from pathlib import Path

import pendulum
import sqlalchemy as sa
from attrs import field
from sqlalchemy import Column, types
from uuid6 import UUID, uuid7

from qwip.attrs import qdefine
from qwip.database.dolt import DoltTable
from qwip.database.metadata import QWIP_DB_METADATA, QWIP_DB_REGISTRY
from qwip.config.models import VersionControlled


class GUID(types.TypeDecorator):
    impl = types.BINARY(16)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return value.bytes

    def process_result_value(self, value, dialect):
        return UUID(bytes=value)


class PendulumDateTime(types.TypeDecorator):
    impl = types.DateTime

    def process_result_value(self, value, dialect):
        return pendulum.instance(value).in_tz("local")


class FilePath(types.TypeDecorator):
    impl = types.String

    def process_result_value(self, value, dialect):
        return Path(value)


@qdefine(slots=False)
class DatastoreEntry(VersionControlled):
    id: UUID = field(factory=uuid7, repr=lambda uid: uid.hex)
    timestamp: pendulum.DateTime = field(
        repr=lambda dt: dt.in_tz("local").isoformat()
        if isinstance(dt, pendulum.DateTime)
        else repr(dt),
        factory=pendulum.now,
    )
    host: str
    filename: Path
    user: str | None = None
    config_db: str | None = None
    commit: str | None = None
    diff: dict = field(factory=dict)
    sample_id: str | None = None
    cooldown_id: str | None = None
    protocol: str | None = None
    sequence: dict = field(factory=dict)
    comments: str | None = None


datastore_entry_table = DoltTable(
    "datastore_entries",
    QWIP_DB_METADATA,
    Column(
        "entry_id",
        GUID,
        primary_key=True,
        default=lambda: uuid7().bytes,
        nullable=False,
    ),
    Column("timestamp", PendulumDateTime),
    Column("host", sa.String(255)),
    Column("filename", FilePath(255)),
    Column("user", sa.String(255)),
    Column("config_db", sa.String(255)),
    Column("config_commit", sa.String(32)),
    Column("config_diff", sa.JSON),
    Column("sample_id", sa.String(16)),
    Column("cooldown_id", sa.String(16)),
    Column("protocol", sa.String(255)),
    Column("sequence", sa.JSON),
    Column("comments", sa.String(4096)),
)

QWIP_DB_REGISTRY.map_imperatively(
    DatastoreEntry,
    datastore_entry_table,
    properties=dict(
        id=datastore_entry_table.c.entry_id,
        commit=datastore_entry_table.c.config_commit,
        diff=datastore_entry_table.c.config_diff,
    ),
)

datastore_tables = [datastore_entry_table]

for table in datastore_tables:
    if isinstance(table, DoltTable):
        table.create_system_tables()
