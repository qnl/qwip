from pathlib import Path

import pendulum
import sqlalchemy as sa
from attrs import field
from sqlalchemy import Column
from uuid6 import UUID, uuid7

from qwip.attrs import qdefine
from qwip.database.database import VersionControlled
from qwip.database.dolt import DoltTable
from qwip.database.metadata import QWIP_DB_METADATA, QWIP_DB_REGISTRY
from qwip.database.utils import GUID, FilePath, PendulumDateTime


def _sequence_repr(seq: dict) -> str:
    try:
        return f"Sequence(shape={seq['shape']}, names=({seq['names']})"
    except KeyError:
        return repr(seq)


@qdefine(slots=False)
class Dataset(VersionControlled):
    id: UUID = field(factory=uuid7, repr=lambda uid: uid.hex)
    timestamp: pendulum.DateTime = field(
        repr=lambda dt: dt.in_tz("local").isoformat()
        if isinstance(dt, pendulum.DateTime)
        else repr(dt),
        factory=pendulum.now,
    )
    host: str
    filename: Path | None = field(repr=lambda p: p.as_posix(), default=None)
    fmt: str | None = None
    user: str | None = None
    config_db: str | None = None
    commit: str | None = None
    diff: dict = field(factory=dict)
    sample_id: str | None = None
    cooldown_id: str | None = None
    protocol: str | None = None
    sequence: dict = field(
        factory=dict,
        repr=_sequence_repr,
    )
    comments: str | None = None


dataset_table = DoltTable(
    "datasets",
    QWIP_DB_METADATA,
    Column(
        "dataset_id",
        GUID,
        primary_key=True,
        default=lambda: uuid7().bytes,
        nullable=False,
    ),
    Column("timestamp", PendulumDateTime),
    Column("host", sa.String(255)),
    Column("filename", FilePath(255)),
    Column("fmt", sa.String(32)),
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
    Dataset,
    dataset_table,
    properties=dict(
        id=dataset_table.c.dataset_id,
        commit=dataset_table.c.config_commit,
        diff=dataset_table.c.config_diff,
    ),
)

datastore_tables = [dataset_table]

for table in datastore_tables:
    if isinstance(table, DoltTable):
        table.create_system_tables()
