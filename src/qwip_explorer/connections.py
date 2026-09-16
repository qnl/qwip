"""Per-browser-session datastore connections for the Explorer."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def connect_datastore(
    host: str,
    datastore_database: str,
    username: str,
    password: str,
    storage_mode: str,
    storage_location: str,
) -> Any:
    """Create a new datastore connection for one Streamlit browser session.

    This intentionally is not a Streamlit cached resource. QWIP database objects own
    mutable SQLAlchemy sessions, which must never be shared by separate browser users.
    """
    from qwip.data import Datastore, HTTPStorageBackend, LocalStorageBackend

    if storage_mode == "HTTP":
        storage = HTTPStorageBackend.from_url(storage_location)
    elif storage_mode == "Local folder":
        storage = LocalStorageBackend(directory=Path(storage_location).expanduser())
    else:
        raise ValueError("Asset storage must be HTTP or Local folder.")

    datastore = Datastore.from_parameters(
        host=host,
        database=datastore_database,
        username=username,
        password=password,
        storage=storage,
    )
    try:
        datastore.connect()
    except Exception:
        close_datastore(datastore)
        raise
    return datastore


def close_datastore(datastore: Any | None) -> None:
    """Release the database engine and optional HTTP client for a datastore."""
    if datastore is None:
        return
    try:
        if getattr(datastore, "session", None) is not None:
            datastore.disconnect()
    finally:
        close = getattr(getattr(datastore, "storage", None), "close", None)
        if callable(close):
            close()


@contextmanager
def history_database(
    host: str, database: str, username: str, password: str
) -> Iterator[Any]:
    """Yield a short-lived read-only Dolt connection for one history operation."""
    from qwip.database.database import DoltDB

    db = DoltDB.from_parameters(
        host=host,
        database=database,
        username=username,
        password=password,
    )
    try:
        db.connect()
        yield db
    finally:
        if getattr(db, "session", None) is not None:
            db.disconnect()
