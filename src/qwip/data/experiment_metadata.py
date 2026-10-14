"""Generic, reproducible measurement metadata for QWIP datasets.

These helpers do not depend on the optional Datastore Explorer and can be used by
measurements, notebooks, and scripts that want to preserve a portable record of a run.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    """Convert common scientific Python values into JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Index, pd.Series)):
        return value.tolist()
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _table_value(value: Any) -> Any:
    """Keep scalars typed and encode structured parameters for a table cell."""
    normalized = _json_value(value)
    if normalized is None or isinstance(normalized, (str, int, float, bool)):
        return normalized
    return json.dumps(normalized, sort_keys=True)


def experiment_parameters(parameters: Mapping[str, Any]) -> pd.DataFrame:
    """Return a one-row overview of fixed timeline or waveform settings."""
    return pd.DataFrame(
        [{str(name): _table_value(value) for name, value in parameters.items()}]
    )


def sweep_parameters(parameters: Mapping[str, Any]) -> pd.DataFrame:
    """Return a table of one-dimensional sweep axes, padding shorter axes."""
    columns: dict[str, pd.Series] = {}
    for name, values in parameters.items():
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"Sweep parameter {name!r} must be one-dimensional.")
        columns[str(name)] = pd.Series(array)
    return pd.DataFrame(columns)


def _builder_description(builder: str | Any | None) -> dict[str, Any] | None:
    if builder is None:
        return None
    if isinstance(builder, str):
        return {"callable": builder}
    description = {"callable": f"{builder.__module__}.{builder.__qualname__}"}
    try:
        description["source"] = inspect.getsource(builder)
    except (OSError, TypeError):
        pass
    return description


def _processor_description(processor: Any) -> Any:
    if isinstance(processor, Mapping):
        return {str(key): _processor_description(value) for key, value in processor.items()}
    if processor is None:
        return None
    if isinstance(processor, type):
        return f"{processor.__module__}.{processor.__qualname__}"
    return _json_value(processor)


def make_measurement_recipe(
    *,
    protocol: str,
    builder: str | Any | None = None,
    parameters: Mapping[str, Any] | None = None,
    sweeps: Mapping[str, Any] | None = None,
    repetitions: int | None = None,
    processor: Any = None,
    compilation: Mapping[str, Any] | None = None,
    backend: Mapping[str, Any] | None = None,
    source_reference: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Create a versioned, human-readable recipe for reconstructing a run."""
    return {
        "schema_version": 1,
        "protocol": protocol,
        "builder": _builder_description(builder),
        "timeline_parameters": _json_value(parameters or {}),
        "sweep_parameters": _json_value(sweeps or {}),
        "acquisition": {
            "repetitions": repetitions,
            "processor": _processor_description(processor),
        },
        "compilation": _json_value(compilation or {}),
        "backend": _json_value(backend or {}),
        "source_reference": _json_value(source_reference),
    }


def make_measurement_data(
    *,
    protocol: str,
    comments: str | None = None,
    builder: str | Any | None = None,
    parameters: Mapping[str, Any] | None = None,
    sweeps: Mapping[str, Any] | None = None,
    repetitions: int | None = None,
    processor: Any = None,
    compilation: Mapping[str, Any] | None = None,
    backend: Mapping[str, Any] | None = None,
    source_reference: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Build QWIP ``data=`` content for any kind of measurement."""
    data: dict[str, Any] = {
        "protocol": protocol,
        "measurement_recipe": make_measurement_recipe(
            protocol=protocol,
            builder=builder,
            parameters=parameters,
            sweeps=sweeps,
            repetitions=repetitions,
            processor=processor,
            compilation=compilation,
            backend=backend,
            source_reference=source_reference,
        ),
    }
    if comments is not None:
        data["comments"] = comments
    if parameters:
        data["experiment_parameters"] = experiment_parameters(parameters)
    if sweeps:
        data["sweep_parameters"] = sweep_parameters(sweeps)
    return data


def run_with_recipe(
    qpu: Any,
    program: Any,
    processor: Any = None,
    *,
    protocol: str,
    builder: str | Any | None = None,
    parameters: Mapping[str, Any] | None = None,
    sweeps: Mapping[str, Any] | None = None,
    repetitions: int = 512,
    compilation: Mapping[str, Any] | None = None,
    backend: Mapping[str, Any] | None = None,
    comments: str | None = None,
    source_reference: Mapping[str, Any] | str | None = None,
    save_sequence: bool = True,
    save: bool = True,
) -> Any:
    """Run a QWIP program while saving one self-consistent measurement recipe."""
    compilation_options = dict(compilation or {})
    backend_options = dict(backend or {})
    data = make_measurement_data(
        protocol=protocol,
        comments=comments,
        builder=builder,
        parameters=parameters,
        sweeps=sweeps,
        repetitions=repetitions,
        processor=processor,
        compilation=compilation_options,
        backend=backend_options,
        source_reference=source_reference,
    )
    if save_sequence:
        data["sequence"] = program
    return qpu.run(
        program,
        processor,
        repetitions=repetitions,
        compilation=compilation_options,
        backend=backend_options,
        data=data,
        save=save,
    )


def make_experiment_data(
    *,
    protocol: str,
    comments: str | None = None,
    fixed: Mapping[str, Any] | None = None,
    sweeps: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Backward-compatible shorthand for simple measurement metadata."""
    return make_measurement_data(
        protocol=protocol, comments=comments, parameters=fixed, sweeps=sweeps
    )


__all__ = [
    "experiment_parameters",
    "make_experiment_data",
    "make_measurement_data",
    "make_measurement_recipe",
    "run_with_recipe",
    "sweep_parameters",
]
