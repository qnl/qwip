"""Streamlit session-state helpers for the Explorer."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


CONNECTION_RESULT_KEYS = (
    "browse_loaded_asset",
    "direct_loaded_asset",
    "dataset_records",
    "direct_open_dataset_id",
    "commit_history_result",
    "history_database_name",
    "history_source_database",
    "structure_branch_commits",
    "connected_username",
    "connected_datastore_database",
    "connected_configuration_database",
)


def clear_connection_results(state: MutableMapping[str, Any]) -> None:
    """Remove results tied to a connection before attempting a replacement."""
    for key in CONNECTION_RESULT_KEYS:
        state.pop(key, None)
