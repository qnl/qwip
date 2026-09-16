"""Pure helpers for inspecting and plotting QWIP datastore objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DATASET_FIELDS = (
    "id",
    "timestamp",
    "protocol",
    "comments",
    "sample_id",
    "cooldown_id",
    "user",
    "host",
    "config_db",
    "commit",
    "version",
)


def timestamp_bounds(
    preset: str,
    *,
    now: datetime | None = None,
    start_date: date | None = None,
    start_time: time | None = None,
    end_date: date | None = None,
    end_time: time | None = None,
    timezone_name: str = "America/Los_Angeles",
) -> tuple[datetime | None, datetime | None]:
    """Build inclusive, timezone-aware dataset-search bounds."""
    timezone = ZoneInfo(timezone_name)
    current = now or datetime.now(timezone)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone)
    else:
        current = current.astimezone(timezone)

    if preset == "Any time":
        return None, None
    if preset == "Last 24 hours":
        return current - timedelta(hours=24), current
    if preset == "Today":
        return current.replace(hour=0, minute=0, second=0, microsecond=0), current
    if preset == "This month":
        return current.replace(day=1, hour=0, minute=0, second=0, microsecond=0), current
    if preset == "This year":
        return current.replace(
            month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        ), current
    if preset != "Custom range":
        raise ValueError(f"Unknown timestamp preset: {preset}")

    if None in (start_date, start_time, end_date, end_time):
        raise ValueError("A custom timestamp range needs both dates and times.")
    start = datetime.combine(start_date, start_time, tzinfo=timezone)
    end = datetime.combine(end_date, end_time, tzinfo=timezone)
    if start > end:
        raise ValueError("The start timestamp must be before the end timestamp.")
    return start, end


def normalize_dataset_id(value: str) -> str:
    """Validate a UUID and return the 32-character form used by QWIP."""
    candidate = value.strip()
    compact = candidate.replace("-", "")
    if len(compact) == 31 and all(character in "0123456789abcdefABCDEF" for character in compact):
        raise ValueError(
            "The dataset ID has 31 hexadecimal characters. It is probably missing "
            f"a leading `0`: `0{candidate}`."
        )
    try:
        return UUID(candidate).hex
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Enter a complete dataset UUID, such as "
            "`01a01754-d305-77e7-9f5d-f5e9a0c52422`."
        ) from exc


def display_value(value: Any) -> Any:
    """Convert common QWIP values into compact table-friendly values."""
    if value is None:
        return None
    if hasattr(value, "hex"):
        hex_value = value.hex
        return hex_value() if callable(hex_value) else hex_value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return value


def datasets_to_frame(datasets: list[Any]) -> pd.DataFrame:
    """Make a stable summary table from QWIP Dataset objects."""
    records = []
    for dataset in datasets:
        records.append(
            {
                field: display_value(getattr(dataset, field, None))
                for field in DATASET_FIELDS
            }
        )
    return pd.DataFrame.from_records(records, columns=DATASET_FIELDS)


def commits_to_frame(
    commits: list[Any], user_query: str = "", limit: int = 50
) -> pd.DataFrame:
    """Create a newest-first Dolt commit table with optional user filtering."""
    normalized_user = user_query.strip().casefold()
    records = []
    for commit in commits:
        committer = str(getattr(commit, "committer", "") or "")
        email = str(getattr(commit, "email", "") or "")
        if normalized_user and normalized_user not in f"{committer} {email}".casefold():
            continue

        commit_hash = str(getattr(commit, "hash", "") or "")
        records.append(
            {
                "date": display_value(getattr(commit, "date", None)),
                "committer": committer,
                "email": email,
                "message": str(getattr(commit, "message", "") or ""),
                "short_hash": commit_hash[:8],
                "commit_hash": commit_hash,
            }
        )

    records.sort(key=lambda record: str(record["date"] or ""), reverse=True)
    columns = [
        "date",
        "committer",
        "email",
        "message",
        "short_hash",
        "commit_hash",
    ]
    return pd.DataFrame.from_records(records[: int(limit)], columns=columns)


def filter_datasets_by_protocol(datasets: list[Any], query: str) -> list[Any]:
    """Apply protocol substring filtering missing from QWIP's current search method."""
    if not query:
        return datasets
    normalized = query.casefold()
    return [
        dataset
        for dataset in datasets
        if normalized in str(getattr(dataset, "protocol", "") or "").casefold()
    ]


def dataset_metadata(dataset: Any) -> dict[str, Any]:
    """Return the searchable and provenance metadata for one dataset."""
    metadata = {
        field: display_value(getattr(dataset, field, None))
        for field in DATASET_FIELDS
    }
    metadata["source"] = getattr(dataset, "source", None)
    return metadata


def assets_to_frame(dataset: Any) -> pd.DataFrame:
    """Summarize assets without loading their payloads."""
    records = []
    for name in dataset.assets():
        asset = dataset[name]
        storage = getattr(asset, "storage", None)
        records.append(
            {
                "name": name,
                "serializer": getattr(asset, "serializer", None),
                "format": getattr(asset, "fmt", None),
                "address": getattr(asset, "address", None),
                "storage": type(storage).__name__ if storage is not None else None,
                "parameters": getattr(asset, "params", None),
            }
        )
    return pd.DataFrame.from_records(records)


def extract_dataframes(obj: Any, prefix: str = "") -> dict[str, pd.DataFrame]:
    """Recursively extract DataFrames from common QWIP asset payloads."""
    label = prefix or "data"
    if isinstance(obj, pd.DataFrame):
        return {label: obj}

    data = getattr(obj, "data", None)
    if isinstance(data, pd.DataFrame):
        name = getattr(obj, "name", None) or label
        return {str(name): data}

    if isinstance(obj, Mapping):
        frames: dict[str, pd.DataFrame] = {}
        for key, value in obj.items():
            child_prefix = f"{prefix} / {key}" if prefix else str(key)
            frames.update(extract_dataframes(value, child_prefix))
        if frames:
            return frames
        try:
            return {label: pd.DataFrame([dict(obj)])}
        except (TypeError, ValueError):
            return {}

    if isinstance(obj, tuple):
        frames: dict[str, pd.DataFrame] = {}
        for index, value in enumerate(obj):
            child_prefix = f"{label} [{index}]"
            frames.update(extract_dataframes(value, child_prefix))
        return frames

    return {}


def object_preview(obj: Any) -> Any:
    """Convert an arbitrary loaded asset into a safe display preview."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (Mapping, list, tuple, str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def flatten_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Reset a DataFrame index and return the columns that were index axes."""
    work = df.copy()
    if isinstance(work.columns, pd.MultiIndex):
        work.columns = [
            " / ".join(str(part) for part in column if str(part) != "")
            for column in work.columns
        ]
    else:
        work.columns = [str(column) for column in work.columns]

    if isinstance(work.index, pd.MultiIndex):
        axis_names = [
            str(name) if name is not None else f"index_{index}"
            for index, name in enumerate(work.index.names)
        ]
        work.index = work.index.set_names(axis_names)
    else:
        axis_names = [str(work.index.name) if work.index.name is not None else "index"]
        work.index = work.index.rename(axis_names[0])

    # Avoid reset_index collisions with a dependent column of the same name.
    occupied = set(work.columns)
    unique_axes = []
    for axis in axis_names:
        candidate = axis
        counter = 1
        while candidate in occupied or candidate in unique_axes:
            candidate = f"{axis}_{counter}"
            counter += 1
        unique_axes.append(candidate)
    work.index = work.index.set_names(unique_axes)
    return work.reset_index(), unique_axes


def plottable_columns(frame: pd.DataFrame, axes: list[str]) -> list[str]:
    """Return numeric and complex dependent columns."""
    columns = []
    for column in frame.columns:
        if column in axes:
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or np.iscomplexobj(series.to_numpy()):
            columns.append(str(column))
    return columns


def transform_values(series: pd.Series, transform: str) -> pd.Series:
    """Apply a display transform, especially for complex IQ values."""
    values = series.to_numpy()
    if transform == "Magnitude":
        output = np.abs(values)
    elif transform == "Phase":
        output = np.angle(values)
    elif transform == "Real":
        output = np.real(values)
    elif transform == "Imaginary":
        output = np.imag(values)
    else:
        output = values
    return pd.Series(output, index=series.index, name=series.name)


def line_figure(
    frame: pd.DataFrame,
    *,
    x: str,
    value: str,
    transform: str = "Raw",
    color: str | None = None,
) -> go.Figure:
    """Build an interactive 1D line/scatter plot."""
    plot_frame = frame.copy()
    plot_frame[value] = transform_values(plot_frame[value], transform)
    sort_columns = [column for column in (color, x) if column]
    plot_frame = plot_frame.sort_values(sort_columns)
    labels = {value: f"{value} ({transform.lower()})" if transform != "Raw" else value}
    return px.line(
        plot_frame,
        x=x,
        y=value,
        color=color,
        markers=True,
        labels=labels,
        title=f"{value} vs {x}",
    )


def heatmap_figure(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
    transform: str = "Raw",
) -> go.Figure:
    """Build an interactive 2D sweep heatmap."""
    plot_frame = frame.copy()
    plot_frame[value] = transform_values(plot_frame[value], transform)
    pivot = plot_frame.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    color_label = f"{value} ({transform.lower()})" if transform != "Raw" else value
    figure = go.Figure(
        data=go.Heatmap(
            x=pivot.columns,
            y=pivot.index,
            z=pivot.to_numpy(),
            colorscale="RdBu",
            colorbar={"title": color_label},
            hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<br>{color_label}: %{{z}}<extra></extra>",
        )
    )
    figure.update_layout(title=f"{value} over {x} and {y}", xaxis_title=x, yaxis_title=y)
    return figure
