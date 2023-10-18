import platform
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pendulum
import sqlalchemy as sa
from attrs import field
from sqlalchemy import Column, ForeignKey, UniqueConstraint, types
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
from uuid6 import UUID, uuid7

import qwip
from qwip.attrs import qdefine
from qwip.data.filesystem import add_extension
from qwip.data.serializers import get_serializer
from qwip.data.storage import StorageBackend
from qwip.database.database import VersionControlled
from qwip.database.dolt import DoltTable
from qwip.database.metadata import QWIP_DB_METADATA, QWIP_DB_REGISTRY
from qwip.database.utils import GUID, PendulumDateTime


def _get_source() -> dict:
    return qwip.converter.unstructure(qwip.qsettings["src"])


class StorageBackendType(types.TypeDecorator):
    impl = types.JSON
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return value

        return qwip.converter.unstructure(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value

        return qwip.converter.structure(value, StorageBackend)


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
    user: str = ""
    version: str = qwip.qsettings["version"]
    source: dict = field(factory=_get_source)
    config_db: str = ""
    commit: str = ""
    sample_id: str = ""
    cooldown_id: str = ""
    protocol: str = ""
    comments: str = ""

    def add(self, assets: "Asset | Iterable[Asset]"):
        match assets:
            case Asset():
                assets = [assets]

        for asset in assets:
            asset.dataset_id = self.id
            self._assets[asset.name] = asset

    def __getitem__(self, key: str) -> "Asset":
        return self._assets[key]


@qdefine(slots=False)
class Asset(VersionControlled):
    name: str
    dataset_id: UUID | None = field(repr=lambda uid: uid.hex, default=None)
    storage: StorageBackend | None = None
    serializer: str = "default"
    params: dict = field(factory=dict)
    obj: Any = None

    @property
    def fmt(self) -> str | None:
        if "fmt" in self.params:
            return self.params["fmt"]

        serializer = get_serializer(self.serializer)
        return serializer.default_format

    @property
    def address(self) -> str:
        if self.dataset_id is None:
            raise ValueError(
                "Cannot compute address for asset with no associated dataset."
            )

        filename = add_extension(self.name, self.fmt or "").lstrip("/")
        return f"/{self.dataset_id.hex}/{filename}"

    def save(self, **kwargs):
        if not self.storage:
            raise ValueError("No storage backend specified, cannot save!")

        params = {"serializer": self.serializer, **self.params} | kwargs
        self.storage.save(self.address, self.obj, **params)

    def load(self, **kwargs) -> Any:
        if not self.storage:
            raise ValueError("No storage backend specified, cannot load!")

        params = {"serializer": self.serializer, **self.params} | kwargs
        self.obj = self.storage.load(self.address, **params)
        return self.obj


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
    Column("storage_backend", StorageBackendType),
    Column("address", sa.String(1024)),
    Column("serializer", sa.String(255)),
    Column("params", sa.JSON),
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
            lazy="selectin",
        ),
    ),
)

QWIP_DB_REGISTRY.map_imperatively(
    Asset,
    asset_table,
    properties=dict(
        storage=asset_table.c.storage_backend,
        dataset=relationship(
            Dataset,
            back_populates="_assets",
        ),
    ),
)

datastore_tables = [dataset_table]

for table in datastore_tables:
    if isinstance(table, DoltTable):
        table.create_system_tables()
