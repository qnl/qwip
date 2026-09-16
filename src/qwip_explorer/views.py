"""Streamlit views for the QWiP Datastore Explorer."""

from __future__ import annotations

import os
import json
from datetime import datetime, time, timedelta
from pprint import pformat
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
import sqlalchemy as sa
import streamlit as st

from qwip_explorer.connections import history_database
from qwip_explorer.core import (
    assets_to_frame,
    commits_to_frame,
    dataset_metadata,
    datasets_to_frame,
    display_value,
    extract_dataframes,
    filter_datasets_by_protocol,
    normalize_dataset_id,
    object_preview,
    timestamp_bounds,
)


def dataset_label(record: dict[str, Any]) -> str:
    name = record.get("protocol") or "No protocol"
    dataset_id = record.get("id") or "unknown ID"
    timestamp = record.get("timestamp") or "unknown time"
    return f"{name} — {timestamp} — {dataset_id}"


SYSTEM_DATABASES = {"information_schema", "mysql", "performance_schema", "sys"}


@st.cache_data(ttl=60, show_spinner=False)
def discover_server_databases(
    host: str,
    seed_database: str,
    username: str,
    password: str,
) -> list[str]:
    """List databases visible to the connected read-only user."""
    with history_database(host, seed_database, username, password) as db:
        with db.session.begin():
            names = [str(name) for name in db.session.execute(sa.text("SHOW DATABASES")).scalars()]
    return sorted(name for name in names if name.casefold() not in SYSTEM_DATABASES)


@st.cache_data(ttl=60, show_spinner=False)
def discover_database_branches(
    host: str,
    database: str,
    username: str,
    password: str,
) -> list[dict[str, Any]]:
    """Return branch heads and their latest commit metadata without checking out a branch."""
    statement = sa.text(
        """
        SELECT name, hash, latest_committer, latest_committer_email,
               latest_commit_date, latest_commit_message
        FROM dolt_branches
        ORDER BY name
        """
    )
    with history_database(host, database, username, password) as db:
        with db.session.begin():
            rows = db.session.execute(statement).mappings().all()
    return [
        {
            "branch": str(row["name"]),
            "head_hash": str(row["hash"]),
            "committer": str(row["latest_committer"] or ""),
            "email": str(row["latest_committer_email"] or ""),
            "date": display_value(row["latest_commit_date"]),
            "message": str(row["latest_commit_message"] or ""),
        }
        for row in rows
    ]


@st.cache_data(ttl=60, show_spinner=False)
def load_branch_commits(
    host: str,
    database: str,
    branch: str,
    username: str,
    password: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Read commits reachable from one branch without changing the active branch."""
    statement = sa.text(
        """
        SELECT commit_hash, date, committer, email, message
        FROM dolt_log(:branch)
        ORDER BY date DESC
        LIMIT :row_limit
        """
    )
    with history_database(host, database, username, password) as db:
        with db.session.begin():
            rows = db.session.execute(
                statement,
                {"branch": branch, "row_limit": int(limit)},
            ).mappings().all()
    return [
        {
            "commit_hash": str(row["commit_hash"]),
            "date": display_value(row["date"]),
            "committer": str(row["committer"] or ""),
            "email": str(row["email"] or ""),
            "message": str(row["message"] or ""),
        }
        for row in rows
    ]


def render_dataset_details(
    datastore: Any,
    dataset_id: str,
    show_plot_controls: Callable[..., None],
    *,
    key_prefix: str,
) -> None:
    """Render metadata, assets, tables, and plots for one dataset."""
    try:
        with st.spinner("Loading dataset metadata..."):
            dataset = datastore.load(dataset_id)
            if dataset is None:
                raise LookupError(f"Dataset `{dataset_id}` was not found.")
            metadata = dataset_metadata(dataset)
            asset_summary = assets_to_frame(dataset)
            asset_names = list(dataset.assets())
    except Exception as exc:
        st.error(f"Dataset loading failed: {exc}")
        return

    source = metadata.pop("source", None)
    st.markdown(f"### {metadata.get('protocol') or 'Unnamed dataset'}")
    st.caption(str(metadata.get("id") or dataset_id))

    overview_tab, parameters_tab, assets_tab, data_tab = st.tabs(
        ["Overview", "Parameters", "Assets", "Plot & data"]
    )

    with overview_tab:
        left, right = st.columns([1, 1.35])
        with left:
            st.subheader("Searchable metadata")
            st.dataframe(
                pd.DataFrame({"field": metadata.keys(), "value": metadata.values()}),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            st.subheader("Source provenance")
            st.json(source or {})

    with parameters_tab:
        parameter_asset_names = [
            name
            for name in ("measurement_recipe", "experiment_parameters", "sweep_parameters")
            if name in asset_names
        ]
        if not parameter_asset_names:
            st.info(
                "No measurement recipe was saved with this dataset. New runs can use "
                "run_with_recipe to save timeline inputs, sweeps, acquisition settings, "
                "compiler settings, and the sequence."
            )
        else:
            parameter_key = f"{key_prefix}:parameters:{dataset_id}"
            state_key = f"{key_prefix}_loaded_parameters"
            if st.button(
                "Load saved parameters",
                type="primary",
                key=f"{key_prefix}_load_parameters",
            ):
                try:
                    with st.spinner("Loading saved parameters..."):
                        parameter_dataset = datastore.load(dataset_id)
                        st.session_state[state_key] = (
                            parameter_key,
                            {
                                name: parameter_dataset[name].load()
                                for name in parameter_asset_names
                            },
                        )
                except Exception as exc:
                    st.error(f"Parameter load failed: {exc}")

            loaded_parameters = st.session_state.get(state_key)
            if not loaded_parameters or loaded_parameters[0] != parameter_key:
                st.caption("Select **Load saved parameters** to retrieve the parameter tables.")
            else:
                parameter_payloads = loaded_parameters[1]
                recipe = parameter_payloads.get("measurement_recipe")
                if isinstance(recipe, dict):
                    st.subheader("Reproducible measurement recipe")
                    st.json(recipe)
                    st.download_button(
                        "Download recipe JSON",
                        json.dumps(recipe, indent=2).encode("utf-8"),
                        file_name=f"{dataset_id}_measurement_recipe.json",
                        mime="application/json",
                        key=f"{key_prefix}_recipe_download",
                    )
                    with st.expander("Copy as Python data"):
                        st.code(
                            f"measurement_recipe = {pformat(recipe, sort_dicts=False)}",
                            language="python",
                        )

                fixed_frame = parameter_payloads.get("experiment_parameters")
                if isinstance(fixed_frame, pd.DataFrame):
                    st.subheader("Fixed experiment settings")
                    fixed_display = fixed_frame.T.reset_index()
                    fixed_display.columns = ["parameter", "value"]
                    st.dataframe(fixed_display, use_container_width=True, hide_index=True)

                sweep_frame = parameter_payloads.get("sweep_parameters")
                if isinstance(sweep_frame, pd.DataFrame):
                    st.subheader("Swept parameters")
                    st.dataframe(sweep_frame, use_container_width=True, hide_index=True)
                    numeric_sweeps = sweep_frame.select_dtypes(include=["number"])
                    if not numeric_sweeps.empty:
                        st.line_chart(numeric_sweeps)

    with assets_tab:
        if not asset_names:
            st.info("This dataset has no stored assets.")
        else:
            st.dataframe(asset_summary, use_container_width=True, hide_index=True)

    with data_tab:
        if not asset_names:
            st.info("There is no asset to load for plotting.")
            return

        selected_asset = st.selectbox(
            "Asset",
            asset_names,
            key=f"{key_prefix}_asset_select",
        )
        load_clicked = st.button(
            "Load asset",
            type="primary",
            key=f"{key_prefix}_asset_load",
        )
        cache_key = f"{key_prefix}_loaded_asset"
        if load_clicked:
            try:
                with st.spinner("Downloading and decoding asset..."):
                    asset_dataset = datastore.load(dataset_id)
                    st.session_state[cache_key] = {
                        "dataset_id": dataset_id,
                        "asset_name": selected_asset,
                        "payload": asset_dataset[selected_asset].load(),
                    }
            except Exception as exc:
                st.session_state.pop(cache_key, None)
                st.error(f"Asset loading failed: {exc}")

        loaded = st.session_state.get(cache_key)
        if not loaded or loaded.get("dataset_id") != dataset_id:
            st.caption("Choose an asset and load it to inspect tables and create plots.")
            return

        payload = loaded["payload"]
        frames = extract_dataframes(payload)
        if frames:
            frame_name = st.selectbox(
                "Result/table",
                list(frames),
                key=f"{key_prefix}_frame_select",
            )
            show_plot_controls(frame_name, frames[frame_name], key_prefix=key_prefix)
        elif hasattr(payload, "savefig"):
            st.pyplot(payload)
        else:
            st.info("This asset is not tabular or directly plottable. Showing its decoded preview instead.")
            preview = object_preview(payload)
            if isinstance(preview, (dict, list, tuple)):
                st.json(preview)
            else:
                st.code(str(preview))


def render_direct_lookup(
    datastore: Any,
    show_plot_controls: Callable[..., None],
) -> None:
    st.subheader("Open a dataset directly")
    st.caption("Paste a dataset UUID to inspect it without searching the catalog first.")

    id_col, button_col = st.columns([4, 1])
    with id_col:
        direct_id = st.text_input(
            "Dataset ID",
            placeholder="01a01754-d305-77e7-9f5d-f5e9a0c52422",
            label_visibility="collapsed",
            key="direct_dataset_id_input",
        )
    with button_col:
        open_direct = st.button(
            "Open dataset",
            use_container_width=True,
            type="primary",
            key="direct_dataset_open",
        )

    if open_direct:
        clean_id = direct_id.strip()
        if not clean_id:
            st.warning("Enter a dataset ID first.")
        else:
            try:
                normalized_id = normalize_dataset_id(clean_id)
                dataset = datastore.load(normalized_id)
                if dataset is None:
                    raise LookupError(f"Dataset `{normalized_id}` was not found.")
                st.session_state.direct_open_dataset_id = normalized_id
            except Exception as exc:
                st.session_state.pop("direct_open_dataset_id", None)
                st.error(f"Dataset lookup failed: {exc}")

    selected_id = st.session_state.get("direct_open_dataset_id")
    if selected_id:
        render_dataset_details(
            datastore,
            selected_id,
            show_plot_controls,
            key_prefix="direct",
        )


def render_database_structure_browser(
    *,
    host: str,
    seed_database: str,
    username: str,
    password: str,
) -> None:
    """Render server database discovery, branch heads, and branch-specific logs."""
    st.subheader("Database and branch browser")
    st.caption(
        f"Starting from the connected catalog `{seed_database}`. All operations are read-only."
    )

    refresh_col, search_col = st.columns([1, 3])
    refresh_clicked = refresh_col.button(
        "Refresh structure",
        use_container_width=True,
        key="structure_refresh",
    )
    database_query = search_col.text_input(
        "Filter databases",
        placeholder="For example: iqm",
        key="structure_database_filter",
    )
    if refresh_clicked:
        discover_server_databases.clear()
        discover_database_branches.clear()
        load_branch_commits.clear()

    try:
        with st.spinner("Discovering accessible databases..."):
            databases = discover_server_databases(
                host,
                seed_database,
                username,
                password,
            )
    except Exception as exc:
        st.error(f"Database discovery failed: {exc}")
        return

    filtered_databases = [
        name for name in databases if database_query.casefold() in name.casefold()
    ]
    if not filtered_databases:
        st.info("No accessible databases match this filter.")
        return

    preferred_database = "iqm" if "iqm" in filtered_databases else filtered_databases[0]
    selected_database = st.selectbox(
        "Database",
        filtered_databases,
        index=filtered_databases.index(preferred_database),
        key="structure_database",
        help="The list is discovered with SHOW DATABASES; no database name is required beforehand.",
    )

    try:
        with st.spinner(f"Reading branches from {selected_database}..."):
            branch_records = discover_database_branches(
                host,
                selected_database,
                username,
                password,
            )
    except Exception as exc:
        st.warning(
            f"`{selected_database}` is visible but does not expose Dolt branches: {exc}"
        )
        return

    branch_query = st.text_input(
        "Filter branches",
        placeholder="main, calibration, experiment...",
        key="structure_branch_filter",
    )
    filtered_branches = [
        record
        for record in branch_records
        if branch_query.casefold()
        in f"{record['branch']} {record['committer']} {record['message']}".casefold()
    ]

    database_metric, branch_metric = st.columns(2)
    database_metric.metric("Accessible databases", len(databases))
    branch_metric.metric("Branches shown", len(filtered_branches))
    if not filtered_branches:
        st.info("No branches match this filter.")
        return

    branch_frame = pd.DataFrame(filtered_branches)
    st.dataframe(
        branch_frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "branch": st.column_config.TextColumn("branch", width="medium"),
            "head_hash": st.column_config.TextColumn("head hash", width="large"),
            "message": st.column_config.TextColumn("latest message", width="large"),
        },
    )

    branch_names = [record["branch"] for record in filtered_branches]
    selected_branch = st.selectbox(
        "Inspect branch commits",
        branch_names,
        key="structure_branch",
    )
    branch_limit = st.number_input(
        "Branch commits to show",
        min_value=1,
        max_value=500,
        value=50,
        step=10,
        key="structure_branch_limit",
    )
    if st.button(
        "Load selected branch commits",
        type="primary",
        key="structure_load_branch_commits",
    ):
        try:
            with st.spinner(f"Reading {selected_branch}..."):
                st.session_state.structure_branch_commits = {
                    "database": selected_database,
                    "branch": selected_branch,
                    "records": load_branch_commits(
                        host,
                        selected_database,
                        selected_branch,
                        username,
                        password,
                        int(branch_limit),
                    ),
                }
        except Exception as exc:
            st.session_state.pop("structure_branch_commits", None)
            st.error(f"Branch commit loading failed: {exc}")

    branch_result = st.session_state.get("structure_branch_commits")
    if (
        branch_result
        and branch_result.get("database") == selected_database
        and branch_result.get("branch") == selected_branch
    ):
        st.markdown(
            f"**{branch_result['database']} → {branch_result['branch']}**"
        )
        branch_commit_frame = pd.DataFrame(branch_result["records"])
        st.dataframe(
            branch_commit_frame,
            use_container_width=True,
            hide_index=True,
            column_config={
                "commit_hash": st.column_config.TextColumn("commit hash", width="large"),
                "message": st.column_config.TextColumn("message", width="large"),
            },
        )
    elif branch_result:
        st.caption("Select **Load selected branch commits** to refresh the commit table.")


def render_commit_history(
    *,
    host: str,
    datastore_database: str,
    configuration_database: str,
    username: str,
    password: str,
) -> None:
    history_view = st.segmented_control(
        "Configuration view",
        ["Browse databases & branches", "My configuration commits"],
        default="Browse databases & branches",
        label_visibility="collapsed",
        key="configuration_view",
    )
    if history_view == "Browse databases & branches":
        render_database_structure_browser(
            host=host,
            seed_database=datastore_database,
            username=username,
            password=password,
        )
    else:
        render_personal_commit_history(
            host=host,
            datastore_database=datastore_database,
            configuration_database=configuration_database,
            username=username,
            password=password,
        )


def render_personal_commit_history(
    *,
    host: str,
    datastore_database: str,
    configuration_database: str,
    username: str,
    password: str,
) -> None:
    st.subheader("Configuration database history")
    st.caption(
        "Calibration commits normally live in the configuration database, not in the "
        "dataset catalog database. This view is read-only."
    )

    history_source = st.radio(
        "Commit source",
        ["Configuration database", "Datastore catalog"],
        horizontal=True,
        help=(
            "Use Configuration database for pulse, calibration, and autocal commits. "
            "The datastore catalog usually contains only initialization/schema commits."
        ),
        key="history_source",
    )
    suggested_database = (
        configuration_database
        if history_source == "Configuration database"
        else datastore_database
    )
    if "history_database_name" not in st.session_state:
        st.session_state.history_database_name = os.getenv(
            "QWIP_HISTORY_DATABASE", suggested_database
        )
        st.session_state.history_source_database = history_source
    elif st.session_state.get("history_source_database") != history_source:
        st.session_state.history_database_name = suggested_database
        st.session_state.history_source_database = history_source
        st.session_state.pop("commit_history_result", None)
    database_name = st.text_input(
        "Database name",
        help="Choose the exact Dolt database whose commits you want to inspect.",
        key="history_database_name",
    )
    st.caption(
        f"Connected datastore catalog: `{datastore_database}` · "
        f"Notebook ConfigDB: `{configuration_database}`"
    )

    filter_col, mode_col, count_col = st.columns([2, 2, 1])
    with filter_col:
        default_committer = os.getenv("QWIP_COMMITTER", username.split("@")[0])
        committer = st.text_input(
            "Committer",
            value=default_committer,
            placeholder="name or email",
            key="history_committer",
        )
    with mode_col:
        match_mode = st.selectbox(
            "Match mode",
            ["Name or email contains", "Exact committer"],
            key="history_match_mode",
        )
    with count_col:
        row_limit = st.number_input(
            "Maximum",
            min_value=1,
            max_value=1000,
            value=50,
            step=10,
            key="history_row_limit",
        )

    load_clicked = st.button("Load commit history", type="primary", key="history_load")
    if load_clicked or "commit_history_result" not in st.session_state:
        target_database = database_name.strip()
        if not target_database:
            st.warning("Enter a database name first.")
        else:
            try:
                with st.spinner(f"Reading commits from {target_database}..."):
                    with history_database(
                        host, target_database, username, password
                    ) as history_db:
                        branch = history_db.current_branch()
                        if match_mode == "Exact committer" and committer.strip():
                            commits = history_db.log(committer=committer.strip())
                            frame = commits_to_frame(commits, user_query="", limit=int(row_limit))
                        else:
                            commits = history_db.log()
                            frame = commits_to_frame(
                                commits,
                                user_query=committer.strip(),
                                limit=int(row_limit),
                            )

                    st.session_state.commit_history_result = {
                        "database": target_database,
                        "branch": display_value(getattr(branch, "name", branch)),
                        "latest_branch_commit": display_value(
                            getattr(getattr(branch, "latest", None), "hash", None)
                        ),
                        "records": frame.to_dict(orient="records"),
                    }
            except Exception as exc:
                st.session_state.pop("commit_history_result", None)
                st.error(f"Commit history loading failed: {exc}")

    result = st.session_state.get("commit_history_result")
    if not result:
        st.info("Choose the database and load its commit history.")
        return

    commit_frame = pd.DataFrame(result.get("records", []))
    database_col, branch_col, latest_col, count_metric_col = st.columns(4)
    database_col.metric("Database", result.get("database") or "Unknown")
    branch_col.metric("Current branch", result.get("branch") or "Unknown")
    latest_commit = result.get("latest_branch_commit") or "Unknown"
    latest_col.metric("Latest branch commit", str(latest_commit)[:8])
    count_metric_col.metric("Commits shown", len(commit_frame.index))
    if latest_commit != "Unknown":
        st.caption(f"Full latest branch commit: `{latest_commit}`")

    if commit_frame.empty:
        st.info("No commits matched this committer filter.")
        return

    display_columns = [
        "commit_hash",
        "date",
        "committer",
        "email",
        "message",
    ]
    st.dataframe(
        commit_frame[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "commit_hash": st.column_config.TextColumn("commit_hash", width="large"),
            "message": st.column_config.TextColumn("message", width="large"),
        },
    )
    st.download_button(
        "Download commit history as CSV",
        data=commit_frame.to_csv(index=False),
        file_name=f"{result['database'].replace('/', '_')}_commit_history.csv",
        mime="text/csv",
        key="history_csv_download",
    )
    latest_hash = commit_frame.iloc[0].get("commit_hash")
    if latest_hash:
        st.caption(f"Newest displayed commit: {str(latest_hash)[:8]}")


def render_dataset_browser(
    datastore: Any,
    show_plot_controls: Callable[..., None],
) -> None:
    st.subheader("Browse datasets")

    with st.sidebar:
        st.divider()
        st.header("Dataset filters")
        timezone_name = os.getenv("QWIP_TIMEZONE", "America/Los_Angeles")
        local_now = datetime.now(ZoneInfo(timezone_name))
        timestamp_preset = st.selectbox(
            "Timestamp range",
            [
                "Any time",
                "Today",
                "Last 24 hours",
                "This month",
                "This year",
                "Custom range",
            ],
        )
        custom_start_date = custom_end_date = None
        custom_start_time = custom_end_time = None
        if timestamp_preset == "Custom range":
            custom_start_date = st.date_input(
                "Start date", value=(local_now - timedelta(days=7)).date()
            )
            custom_start_time = st.time_input("Start time", value=time(0, 0))
            custom_end_date = st.date_input("End date", value=local_now.date())
            custom_end_time = st.time_input(
                "End time", value=local_now.time().replace(microsecond=0)
            )
        if timestamp_preset != "Any time":
            st.caption(f"Timezone: {timezone_name}")
        protocol_filter = st.text_input("Protocol contains")
        comments_filter = st.text_input("Comments contain")
        user_filter = st.text_input("User contains")
        host_filter = st.text_input("Exact host")
        result_limit = st.number_input(
            "Maximum results", min_value=1, max_value=2000, value=100, step=25
        )
        search_clicked = st.button("Search datasets", type="primary", use_container_width=True)

    try:
        search_start, search_end = timestamp_bounds(
            timestamp_preset,
            now=local_now,
            start_date=custom_start_date,
            start_time=custom_start_time,
            end_date=custom_end_date,
            end_time=custom_end_time,
            timezone_name=timezone_name,
        )
    except ValueError as exc:
        st.sidebar.error(str(exc))
        search_start = search_end = None
        if search_clicked:
            return

    if search_clicked or "dataset_records" not in st.session_state:
        try:
            with st.spinner("Searching the datastore..."):
                found_datasets = datastore.search(
                    start=search_start,
                    end=search_end,
                    host=host_filter or None,
                    user=user_filter or None,
                    comments=comments_filter or None,
                    limit=max(int(result_limit), 2000) if protocol_filter else int(result_limit),
                )
                filtered_datasets = filter_datasets_by_protocol(
                    list(found_datasets), protocol_filter
                )[: int(result_limit)]
                st.session_state.dataset_records = datasets_to_frame(
                    filtered_datasets
                ).to_dict("records")
        except Exception as exc:
            st.error(f"Dataset search failed: {exc}")
            return

    records = st.session_state.get("dataset_records", [])
    if not records:
        st.info("No datasets found. Adjust the filters and search again.")
        return

    summary = pd.DataFrame.from_records(records)
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Datasets", f"{len(records):,}")
    metric_b.metric("Protocols", f"{summary['protocol'].nunique(dropna=True):,}")
    metric_c.metric("Users", f"{summary['user'].nunique(dropna=True):,}")

    with st.expander("Dataset catalog", expanded=True):
        catalog_columns = ["timestamp", "protocol", "comments", "sample_id", "user", "id"]
        st.dataframe(
            summary[catalog_columns], use_container_width=True, height=330, hide_index=True
        )
        st.download_button(
            "Download catalog as CSV",
            data=summary.to_csv(index=False),
            file_name="qwip_dataset_catalog.csv",
            mime="text/csv",
            key="catalog_csv_download",
        )

    record_map = {dataset_label(record): record for record in records}
    selected_label = st.selectbox("Dataset", list(record_map.keys()), key="browse_dataset_select")
    selected_id = str(record_map[selected_label]["id"])
    render_dataset_details(
        datastore,
        selected_id,
        show_plot_controls,
        key_prefix="browse",
    )
