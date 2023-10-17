import platform
from collections.abc import Iterable

import pendulum
import sqlalchemy as sa
from attrs import field
from sqlalchemy import Column, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
from uuid6 import UUID, uuid7

import qwip
from qwip.attrs import qdefine
from qwip.database.database import VersionControlled
from qwip.database.dolt import DoltTable
from qwip.database.metadata import QWIP_DB_METADATA, QWIP_DB_REGISTRY
from qwip.database.utils import GUID, PendulumDateTime


def _get_source() -> dict:
    return qwip.converter.unstructure(qwip.qsettings["src"])


@qdefine(slots=False)
class Dataset(VersionControlled):
    id: UUID = field(factory=uuid7, repr=lambda uid: uid.hex)
    timestamp: pendulum.DateTime = field(
        repr=lambda dt: dt.in_tz("local").isoformat()
        if isinstance(dt, pendulum.DateTime)
        else repr(dt),
        factory=pendulum.now,
    )
    host: str = field(factory=platform.node)
    user: str | None = None
    version: str = qwip.qsettings["version"]
    source: dict = field(factory=_get_source)
    config_db: str | None = None
    commit: str | None = None
    sample_id: str | None = None
    cooldown_id: str | None = None
    protocol: str | None = None
    comments: str | None = None

    def add(self, assets: "Asset | Iterable[Asset]"):
        match assets:
            case Asset():
                assets = [assets]

        for asset in assets:
            self._assets[asset.name] = asset

    def __getitem__(self, key: str) -> "Asset":
        return self._assets[key]


@qdefine(slots=False)
class Asset(VersionControlled):
    name: str
    dataset_id: UUID | None = field(repr=lambda uid: uid.hex, default=None)
    backend: str | None = None
    address: str = "/"
    serializer: dict = field(factory=dict)


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
    Column("user", sa.String(255)),
    Column("qwip_version", sa.String(255)),
    Column("qwip_source", sa.JSON),
    Column("config_db", sa.String(255)),
    Column("config_commit", sa.String(32)),
    Column("sample_id", sa.String(16)),
    Column("cooldown_id", sa.String(16)),
    Column("protocol", sa.String(255)),
    Column("comments", sa.String(4096)),
)

asset_table = DoltTable(
    "assets",
    QWIP_DB_METADATA,
    Column("asset_id", sa.Integer, primary_key=True, autoincrement=True),
    Column(
        "dataset_id",
        GUID,
        ForeignKey(
            "datasets.dataset_id",
            name="fk_assets_datasets",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        nullable=False,
    ),
    Column("name", sa.String(255)),
    Column("storage_backend", sa.String(255)),
    Column("address", sa.String(1024)),
    Column("serializer", sa.JSON),
    UniqueConstraint("name", "dataset_id", name="uq_assets_name_dataset_id"),
    UniqueConstraint("dataset_id", "asset_id", name="uq_assets_asset_id_dataset_id"),
)

QWIP_DB_REGISTRY.map_imperatively(
    Dataset,
    dataset_table,
    properties=dict(
        id=dataset_table.c.dataset_id,
        version=dataset_table.c.qwip_version,
        source=dataset_table.c.qwip_source,
        commit=dataset_table.c.config_commit,
        _assets=relationship(
            Asset,
            cascade="all, delete-orphan",
            back_populates="dataset",
            collection_class=attribute_mapped_collection("name"),
        ),
    ),
)

QWIP_DB_REGISTRY.map_imperatively(
    Asset,
    asset_table,
    properties=dict(
        dataset=relationship(
            Dataset,
            back_populates="_assets",
        )
    ),
)

datastore_tables = [dataset_table]

for table in datastore_tables:
    if isinstance(table, DoltTable):
        table.create_system_tables()
