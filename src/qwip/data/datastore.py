import inspect
import platform
from typing import Literal

import pendulum
import sqlalchemy as sa
from attrs import field
from pendulum import DateTime
from sqlalchemy.engine import URL, make_url
from uuid6 import UUID

import qwip
from qwip.attrs import qdefine
from qwip.config.interface import OfflineConfigDB
from qwip.data.filesystem import DataFormat, DataSaver
from qwip.data.models import Dataset
from qwip.database.database import Database, DoltDB
from qwip.processing.data_processor import DataProcessor, MeasurementResult
from qwip.sequencer.sequence import Sequence


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

    datasaver: DataSaver

    def save(
        self,
        results: list[dict[str, MeasurementResult]]
        | dict[str, MeasurementResult]
        | MeasurementResult,
        fmt: DataFormat | Literal[".csv", ".parquet", ".feather"] = DataFormat.parquet,
        config_db: OfflineConfigDB | None = None,
        seq: Sequence | None = None,
        comments: str | None = None,
    ) -> Dataset:
        """Saves a set of measurement results.
        
        Args:
            results: A list of result sets. Each result set is a dictionary mapping 
                measurement keys to `MeasurementResult`. A single result set or a single
                `MeasurementResult` can also be saved individually.
            fmt: The data format to use.
            config_db: The configuration database used to acquire the measurement.
            seq: The sequence used to acquire the measurement.
            comments: A string attached to the dataset. Can be used to tag datasets with
                additional information.

        Returns:
            The dataset that was saved to the datastore.
        """
        match results:
            case dict():
                results = [results]
            case MeasurementResult(name=name):
                results = [{name: results}]

        with self.session.begin():
            sequence_data = {} if seq is None else qwip.converter.unstructure(seq)

            config_data = {}
            if config_db:
                config_data.update(
                    config_db=str(config_db.url).split("@")[-1],
                    commit=config_db.log()[-1].hash,
                    diff={},
                    sample_id=config_db.config["sample_id"],
                    cooldown_id=config_db.config["cooldown_id"],
                )

            dataset = Dataset(
                fmt=fmt,
                host=platform.node(),
                user=self.username,
                comments=comments,
                sequence=sequence_data,
                **config_data,
            )

            for result in results:
                save_path = self.datasaver.save(dataset.id.hex, result, fmt=fmt)

            dataset.filename = save_path.parent
            self.session.add(dataset)

        return dataset

    def load(
        self, dataset_id: UUID | str, result_type: type[DataProcessor] | None = None
    ) -> tuple[Dataset, dict[MeasurementResult]]:
        """Loads a dataset and associated measurement results from the datastore.

        Args:
            dataset_id: The dataset identifier.
            result_type: The final processor for the measurement result.

        Returns:
            A tuple `(dataset, measurement_result)`.
        """
        dataset = self.load_dataset(dataset_id)

        filename = dataset.filename / self.datasaver.get_filename(
            result_type, fmt=dataset.fmt
        )
        return dataset, self.datasaver.load(filename)

    def load_dataset(self, dataset_id: UUID | str) -> Dataset:
        """Gets a dataset from the datastore by id.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            The dataset with the corresponding identifier.
        """
        if isinstance(dataset_id, str):
            dataset_id = UUID(dataset_id)

        with self.session.begin():
            stmt = sa.select(Dataset).where(Dataset.id == dataset_id)
            dataset = self.session.scalars(stmt).one()

        return dataset

    def files(self, dataset_id: UUID | str) -> set[type[DataProcessor]]:
        dataset = self.get_dataset(dataset_id)
        return [
            f.name for f in dataset.filename.iterdir() if f.suffix[1:] == dataset.fmt
        ]

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

    def search(
        self,
        *,
        start: DateTime | None = None,
        end: DateTime | None = None,
        host: str | None = None,
        fmt: DataFormat | str | None = None,
        user: str | None = None,
        config_db: str | None = None,
        commit: str | None = None,
        sample_id: str | None = None,
        cooldown_id: str | None = None,
        comments: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_desc: bool = True,
    ) -> list[Dataset]:
        """Search the datastore for a list of datasets.
        
        Args:
            start: The start time to filter the datasets by. All returned datasets will
                have a timestamp greater than or equal to `start`.
            end: The end time to filter the datasets by. All returned datasets will have
                a timestamp less than or equal to `end`.
            host: The host computer on which the dataset was taken from.
            fmt: The data format used to store the data.
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
        stmt = sa.select(Dataset)

        if order_desc:
            stmt = stmt.order_by(Dataset.timestamp.desc())
        else:
            stmt = stmt.order_by(Dataset.timestamp)

        if start:
            stmt = stmt.where(Dataset.timestamp >= start)
        if end:
            stmt = stmt.where(Dataset.timestamp <= end)

        exact = dict(host=host, fmt=fmt, sample_id=sample_id, cooldown_id=cooldown_id)
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

        with self.session.begin():
            datasets = self.session.scalars(stmt).all()

        return datasets

    def tail(self, limit: int, **kwargs):
        kwargs |= dict(limit=limit)

        return self.search(**kwargs)


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
