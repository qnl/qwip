import platform
from collections.abc import Iterable
from typing import Any

import attrs
import pendulum
import sqlalchemy as sa
from attrs import field
from sqlalchemy import Column, ForeignKey, UniqueConstraint, types
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
from typing_extensions import Self
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
    """A dataset record.

    A dataset is a record that contains searchable metadata about a collection of data
    assets or resources derived from some measurement data taken at a point in time.

    Attributes:
        id: A dataset ID. These are generated using uuid7 so that the identifiers can
            be time-ordered with a lexicographic sort.
        timestamp: A datetime instance specifying when the data was saved.
        host: The hostname of the computer that the python client code is being run
            on.
        user: The user saving the data. This is determined by the database connection
            user.
        source: Information about the QWiP source code, if it is being run from an
            editable install.
        config_db: The name of the configuration database being used, or `None` if no
            configuration database is specified.
        commit: The commit hash of the configuration database, or `None` if no
            configuration database is specified.
        sample_id: A sample identifier specifying the sample that the data was taken on.
        cooldown_id: A cooldown identifier specifying which fridge and cooldown the
            sample was being measured in.
        protocol: A QWiP protocol that was used to take the data, or `None`, if no
            protocol was used.
        comments: Additional comments to attach to the dataset. This is stored in the
            database so that datasets are searchable by comment.
    """

    id: UUID = field(factory=uuid7, repr=lambda uid: uid.hex if uid else str(uid))
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
        """Adds an asset or list of assets to the dataset.

        Use `update` to update an existing asset attached to the dataset.

        Args:
            assets: An asset or an iterable of assets to add to the dataset.
        """
        match assets:
            case Asset():
                assets = [assets]

        for asset in assets:
            asset.dataset_id = self.id
            self._assets[asset.name] = asset

    def update(self, asset: "Asset"):
        """Update an existing asset in the database.

        Note that this only updates the asset parameters and does not save the asset
        data to the storage backend. Another call to `Asset.save` is needed to update
        the storage backend. If an asset with the same name does not already exist,
        a new one is created.

        Args:
            asset: The asset to update/add.
        """
        if asset.name not in self:
            self.add(asset)
            return asset

        orig = self[asset.name]
        for f in attrs.fields(Asset):
            if f.name == "dataset_id":
                continue
            setattr(orig, f.name, getattr(asset, f.name))

        return orig

    def __getitem__(self, key: str) -> "Asset":
        asset = self._assets[key]
        # repr fails if obj is not present due to ORM weirdness
        if not hasattr(asset, "obj"):
            asset.obj = None

        return asset

    def __contains__(self, key: str) -> bool:
        return key in self._assets

    def assets(self) -> tuple[str]:
        """Returns the names of all assets attached to the dataset."""
        return tuple(asset.name for asset in self._assets.values())


@qdefine(slots=False)
class Asset(VersionControlled):
    """A data asset record.

    An asset is a specific piece of data that is attached to a dataset. Each asset has
    some metadata specifying how to save and retrieve the data. The actual data itself
    is serialized to some binary representation and saved using a storage backend.

    Attributes:
        name: The asset name. Two assets attached to the same dataset cannot share the
            same name.
        dataset_id: The id for the dataset that the asset is attached to.
        storage: A `StorageBackend` used to persist the binary object data to save.
        serializer: The `serializer` to use when saving the asset data.
        params: Any parameters to pass to the storage backend when saving the asset
            data.
        obj: The object to persist. This is the actual data to be saved/reloaded.
    """

    name: str
    dataset_id: UUID | None = field(
        repr=lambda uid: uid.hex if uid else str(uid), default=None
    )
    storage: StorageBackend | None = None
    address: str | None = None
    serializer: str = "default"
    params: dict = field(factory=dict)
    obj: Any = None

    @property
    def fmt(self) -> str | None:
        if "fmt" in self.params:
            return self.params["fmt"]

        serializer = get_serializer(key=self.serializer, obj=self.obj)
        return serializer.default_format

    def default_address(self) -> str:
        if self.dataset_id:
            dataset_id = self.dataset_id
        elif self.dataset:
            dataset_id = self.dataset.id
        else:
            return None

        if dataset_id is None:
            return None

        filename = add_extension(self.name, self.fmt or "").lstrip("/")
        return f"/{dataset_id.hex}/{filename}"

    @classmethod
    def create(cls, obj: Any, /, name: str | None = None, **kwargs) -> Self:
        serializer = get_serializer(key=kwargs.get("serializer"), obj=obj)
        name = name or serializer.get_name(obj)

        kwargs = dict(serializer=serializer.key) | kwargs

        return cls(name=name, obj=obj, **kwargs)

    def save(self, **kwargs):
        """Saves the asset data to the storage backend.

        Data is serialized using the specified serializer to a binary representation
        and then saved to the storage backend.

        Args:
            **kwargs: Keyword arguments are passed to `storage.save` and can be used to
                override any default parameters attached to the asset.
        """
        if not self.storage:
            raise ValueError("No storage backend specified, cannot save!")

        self.address = self.address or self.default_address()
        if not self.address:
            raise ValueError(f"Asset {self} cannot be saved without an address.")

        params = {"serializer": self.serializer, **self.params} | kwargs
        self.storage.save(self.address, self.obj, **params)

    def load(self, **kwargs) -> Any:
        """Loads the asset data from the storage backend.

        Data is loaded in its binary form from the storage backend and then deserialized
        with the speccified serializer. In some cases (e.g. matplotlib figures), the
        reloaded data may not be in the same form as it was when it was initially saved.

        Args:
            **kwargs: Keyword arguments are passed to `storage.load` and can be used to
                override any default parameters attached to the asset.

        Returns:
            The reloaded and deserialized object from the storage backend.
        """
        if not self.storage:
            raise ValueError("No storage backend specified, cannot load!")

        if not self.address:
            raise ValueError(f"Asset {self} cannot be loaded without an address.")

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
    Column("sample_id", sa.String(255)),
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

datastore_tables = [dataset_table, asset_table]

for table in datastore_tables:
    if isinstance(table, DoltTable):
        table.create_system_tables()

__all__ = ["Asset", "Dataset"]
