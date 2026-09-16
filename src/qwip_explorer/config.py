"""Configuration defaults for the local QWIP Datastore Explorer."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ExplorerDefaults:
    """Connection form defaults supplied by a user's environment."""

    host: str = ""
    datastore_database: str = ""
    configuration_database: str = ""
    username: str = ""
    password: str = ""
    storage_mode: str = "HTTP"
    http_storage_url: str = ""
    local_storage_directory: str = ""

    @classmethod
    def from_environment(cls) -> "ExplorerDefaults":
        storage_mode = os.getenv("QWIP_STORAGE_MODE", "HTTP")
        if storage_mode not in {"HTTP", "Local folder"}:
            storage_mode = "HTTP"
        return cls(
            host=os.getenv("QWIP_DB_HOST", ""),
            datastore_database=os.getenv("QWIP_DATASTORE_DB", ""),
            configuration_database=os.getenv("QWIP_CONFIG_DB", ""),
            username=os.getenv("QWIP_DB_USERNAME", ""),
            password=os.getenv("QWIP_DB_PASSWORD", ""),
            storage_mode=storage_mode,
            http_storage_url=os.getenv("QWIP_STORAGE_URL", ""),
            local_storage_directory=os.getenv("QWIP_LOCAL_DATASTORE", ""),
        )
