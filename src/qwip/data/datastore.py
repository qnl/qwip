import platform
from typing import Any

import sqlalchemy as sa
from attrs import field
from loguru import logger
from pendulum import DateTime
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import lazyload
from uuid6 import UUID

from qwip.attrs import qdefine
from qwip.config.interface import OfflineConfigDB
from qwip.data.models import Asset, Dataset
from qwip.data.storage import StorageBackend
from qwip.database.database import Database, DoltDB, session_context


@qdefine
class OfflineDatastore(Database):
    """An interface to a local datastore backend.

    The OfflineDatastore can be used with a SQLite database backend, with the caveat
    that no database version control features are available.

    Attributes:
        url: The database connection url.
        engine: The SQLAlchemy engine that maintains the database connection.
        session: The database session associated with the entine.

    """

    storage: StorageBackend

    @session_context
    def save(
        self,
        *unnamed,
        config_db: OfflineConfigDB | None = None,
        sample_id: str | None = None,
        cooldown_id: str | None = None,
        protocol: str | None = None,
        comments: str | None = None,
        **named: Any | Asset,
    ) -> Dataset:
        """Saves a set of assets to a new dataset.

        Args:
            *unnamed: A variable number of unnamed assets. If these are not instances of
                `Asset`, an `Asset` will be created with `Asset.create`.
            config_db: A `ConfigDB` instance to pull searchable metadata from.
            sample_id: A sample identifier. This will override the `sample_id` in the
                configuration database.
            cooldown_id: A cooldown identifier. This will override the `cooldown_id` in
                the configuration database.
            protocol: A measurement protocol name.
            comments: A comment to attach to the dataset.
            **named: Any additional keyword arguments are taken to be assets to add to
                the dataset. If these are not instances of `Asset`, an `Asset` will be
                created, and the key will be passed in as the asset name.

        Returns:
            A new dataset containing the assets.
        """
        config_data = {}
        if config_db:
            config_data.update(
                config_db=f"{config_db.host}/{config_db.database}",
                sample_id=config_db.config["sample_id"],
                cooldown_id=config_db.config["cooldown_id"],
            )

            try:
                config_data["commit"] = config_db.log()[-1].hash
            except AttributeError:
                ...

        if sample_id is not None:
            config_data.update(sample_id=sample_id)

        if cooldown_id is not None:
            config_data.update(cooldown_id=cooldown_id)

        dataset = Dataset(
            host=platform.node(),
            user=self.username,
            protocol=protocol,
            comments=comments,
            **config_data,
        )

        assets = self._make_assets(*unnamed, **named)
        dataset.add(assets)
        for asset in assets:
            asset.save()
        self.session.add(dataset)

        return dataset

    def _make_assets(self, *unnamed, **named):
        assets = []
        for obj in unnamed:
            match obj:
                case Asset(storage=sb) if sb is None:
                    obj.storage = self.storage
                case _:
                    obj = Asset.create(obj, storage=self.storage)

            assets.append(obj)

        for name, obj in named.items():
            match obj:
                case Asset(name=n, storage=sb):
                    if n != name:
                        logger.warning(
                            f"Keyword name '{name}' differs from asset name "
                            f"'{n}'. Using '{n}'."
                        )
                    if sb is None:
                        obj.storage = self.storage
                case _:
                    obj = Asset.create(obj, name, storage=self.storage)

            assets.append(obj)

        return assets

    @session_context
    def add_asset(
        self,
        dataset_id: UUID | str,
        obj: Any,
        name: str | None = None,
        overwrite: bool = False,
    ) -> Asset:
        """Add an asset to an existing dataset.

        Args:
            dataset_id: The identifier for the dataset to add the asset to.
            obj: The object to save.
            name: An optional name for the asset. If no name is provided and the object
                is not already wrapped in an Asset, a generic name will be generated.
                See `Asset.create` for the naming behavior.
            overwrite: If `False`, will raise an exception if an asset with the same
                name already exists on the dataset.

        Returns:
            The created asset that was added to the dataset and saved.
        """
        if isinstance(dataset_id, str):
            dataset_id = UUID(dataset_id)

        dataset = self.load(dataset_id)
        if dataset is None:
            raise KeyError(f"Could not find dataset '{dataset_id.hex}'")

        match obj:
            case Asset(storage=sb):
                if sb is None:
                    obj.storage = self.storage

                if obj.name != name:
                    logger.warning(
                        f"Name '{name}' differs from asset name '{obj.name}'. Using "
                        f"'{obj.name}'."
                    )
            case _:
                obj = Asset.create(obj, name, storage=self.storage)

        if obj.name in dataset and not overwrite:
            raise FileExistsError(
                f"There already exists an asset with name '{obj.name}'. Set "
                f"`overwrite=True` to update the asset."
            )

        obj = dataset.update(obj)
        obj.save()

        return obj

    @session_context
    def load(self, dataset_id: UUID | str) -> Dataset | None:
        """Loads a dataset from the datastore.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            The corresponding dataset if one exists or None.
        """
        if isinstance(dataset_id, str):
            dataset_id = UUID(dataset_id)

        stmt = sa.select(Dataset).where(Dataset.id == dataset_id)
        dataset = self.session.scalars(stmt).one_or_none()

        return dataset

    @classmethod
    def _add_equals(cls, stmt: sa.Selectable, **kwargs) -> sa.Selectable:
        """Adds where clauses to a statement.

        Args:
            stmt: The statement to add a where clause to.
            **kwargs: Column names and values should to search for should be passed in
                as keyword arguments.

        Returns:
            The modified statement.
        """
        for col, value in kwargs.items():
            stmt = stmt.where(getattr(Dataset, col) == value)

        return stmt

    @classmethod
    def _add_substring_search(cls, stmt: sa.Selectable, **kwargs) -> sa.Selectable:
        """Adds LIKE clauses to a statement to do a case-insensitive search.

        Args:
            stmt: The statement to add a like clause to.
            **kwargs: Column names and values should to search for should be passed in
                as keyword arguments.

        Returns:
            The modified statement.
        """
        for col, value in kwargs.items():
            search_value = f"%{value}%" if "%" not in value else value
            stmt = stmt.where(getattr(Dataset, col).ilike(search_value))

        return stmt

    @session_context
    def search(
        self,
        *,
        start: DateTime | None = None,
        end: DateTime | None = None,
        host: str | None = platform.node(),
        user: str | None = None,
        config_db: str | None = None,
        commit: str | None = None,
        sample_id: str | None = None,
        cooldown_id: str | None = None,
        comments: str | None = None,
        limit: int = 50,
        offset: int | None = None,
        order_desc: bool = True,
    ) -> list[Dataset]:
        """Search the datastore for a list of datasets.

        Note that assets are not loaded automatically in order to speed up queries. To
        load the assets for a specific dataset, reload the dataset with the
        `Datastore.load` method.

        Args:
            start: The start time to filter the datasets by. All returned datasets will
                have a timestamp greater than or equal to `start`.
            end: The end time to filter the datasets by. All returned datasets will have
                a timestamp less than or equal to `end`.
            host: The host computer on which the dataset was taken from.
            user: The user who saved the data.
            config_db: The databse url (`hostname:port/db_name`) for the configuration
                database associated with the dataset.
            commit: The configdb commit at the time the dataset was taken.
            sample_id: The sample id associated with the dataset.
            cooldown_id: The cooldown id associated with the dataset.
            comments: A substring to search for in the comments field of the dataset.
            limit: The maximum number of dataset entries to return.
            offset: The offset number to start searching from. Used for pagination in
                conjunction with the limit argument.
            order_desc: If `True`, order the datasets such that their timestamps go from
                most recent to least recent.

        Returns:
            A list of datasets that fit the search criteria.
        """
        stmt = sa.select(Dataset).options(lazyload(Dataset._assets))

        if order_desc:
            stmt = stmt.order_by(Dataset.id.desc())
        else:
            stmt = stmt.order_by(Dataset.id)

        if start:
            stmt = stmt.where(Dataset.timestamp >= start)
        if end:
            stmt = stmt.where(Dataset.timestamp <= end)

        exact = dict(host=host, sample_id=sample_id, cooldown_id=cooldown_id)
        substring = dict(
            user=user, config_db=config_db, commit=commit, comments=comments
        )

        stmt = type(self)._add_equals(
            stmt, **{c: v for c, v in exact.items() if v is not None}
        )
        stmt = type(self)._add_substring_search(
            stmt, **{c: v for c, v in substring.items() if v is not None}
        )

        if limit is not None:
            stmt = stmt.limit(limit)

        if offset is not None:
            stmt = stmt.offset(offset)

        datasets = self.session.scalars(stmt).all()

        return datasets

    @session_context
    def tail(self, limit: int = 50, **kwargs) -> list[Dataset]:
        """Get the n most recent datasets that fit the search criteria.

        Note that datasets are always ordered from most recent to least recent.

        Args:
            limit: The maximum number of datasets to return.
            **kwargs: See `datstore.search` for allowed keywords.

        Returns:
            A list of the n most recent datasets that fit the search criteria.
        """
        kwargs |= dict(limit=limit, order_desc=True)

        return self.search(**kwargs)

    @session_context
    def last(self, **kwargs) -> Dataset:
        """Get the most recent ddataset that fits the search criteria.

        This is equivalent to calling `datastore.tail(limit=1)[0]`, except assets are
        loaded eagerly.

        Args:
            **kwargs: See `datastore.search` for allowed keywords.

        Returns:
            The most recent dataset that fits the search criteria.
        """
        dataset = self.tail(1, **kwargs)[-1]

        # Force load of assets before session ends.
        dataset._assets
        return dataset


@qdefine
class Datastore(OfflineDatastore, DoltDB):
    """An interface to a datastore backend using Dolt.

    Attributes:
        url: The database connection url.
        engine: The SQLAlchemy engine that maintains the database connection.
        session: The database session associated with the entine.
    """

    url: URL = field(
        converter=lambda s: make_url(s) if isinstance(s, str) else s,
    )

    @url.validator
    def _validate_url(self, attribute, value):
        if value.get_backend_name() != "mysql":
            raise ValueError("Dolt database must use mysql driver")

        if value.database is None:
            raise ValueError("Database name must be provided for Datastore.")


__all__ = ["Datastore", "OfflineDatastore"]
