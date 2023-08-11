import functools
import re

import attrs
import pendulum
import sqlalchemy as sa
from attrs import field
from loguru import logger
from sqlalchemy import event
from sqlalchemy.engine import URL, Engine, make_url
from sqlalchemy.orm import Session
from typing_extensions import Self

import qwip
from qwip.attrs import qdefine, qfrozen
from qwip.database.dolt import (
    DoltBranch,
    DoltCommit,
    DoltLog,
    DoltStatus,
    dolt_add,
    dolt_branch,
    dolt_checkout,
    dolt_commit,
    dolt_reset,
)

SHORT_HASH_LEN = 8
USERNAME_REGEX = re.compile(r"(?P<name>[^@]*)(?P<domain>@.*)?")


@qfrozen
class Commit:
    hash: str = field(repr=lambda h: h[:SHORT_HASH_LEN])
    committer: str
    email: str
    date: pendulum.DateTime = field(repr=lambda dt: dt.in_tz("local").isoformat())
    message: str

    @classmethod
    def from_orm(cls, model: DoltCommit):
        return cls(
            hash=model.commit_hash,
            committer=model.committer,
            email=model.email,
            date=model.date,
            message=model.message,
        )

    @property
    def short_hash(self):
        return self.hash[:SHORT_HASH_LEN]


@qfrozen
class Branch:
    name: str
    latest: Commit

    @classmethod
    def from_orm(cls, model: DoltBranch) -> Self:
        commit = Commit(
            hash=model.hash,
            committer=model.latest_committer,
            email=model.latest_committer_email,
            date=model.latest_commit_date,
            message=model.latest_commit_message,
        )

        return cls(name=model.name, latest=commit)


@qfrozen
class Status:
    table: str
    staged: bool
    status: str

    @classmethod
    def from_orm(cls, model: DoltStatus) -> Self:
        return cls(table=model.table_name, staged=model.staged, status=model.status)


@qdefine(slots=False)
class VersionControlled:
    @property
    def table(self) -> sa.Table:
        return self.__table__

    def history(self, connection) -> list[str, Self]:
        table = self.table
        history_cols = table._dolt_history.columns

        cond = sa.and_(
            *(
                getattr(history_cols, col.name) == getattr(self, col.name)
                for col in table.primary_key.columns
            )
        )

        mapper = self.__mapper__

        aliases = dict()
        for f in attrs.fields(type(self)):
            columns = getattr(mapper.get_property(f.name), "columns", None)
            if columns and f.init:
                aliases[f.name] = columns[0].name

        # Everything before here can maybe be cached in the future.

        stmt = sa.select(table._dolt_history).where(cond)
        results = connection.execute(stmt)

        obj_results = []
        for row in results:
            obj = type(self)(
                **{
                    attribute: row._mapping.get(column)
                    for attribute, column in aliases.items()
                }
            )
            commit_hash = row._mapping.get("commit_hash")

            obj_results.append((commit_hash, obj))

        return obj_results


def session_context(func):
    @functools.wraps(func)
    def decorated(inst, *args, **kwargs):
        if inst.session.in_transaction():
            return func(inst, *args, **kwargs)
        else:
            with inst.session.begin():
                return func(inst, *args, **kwargs)

    return decorated


# Savepoint support for sqlite
# See https://docs.sqlalchemy.org/en/latest/dialects/sqlite.html#pysqlite-serializable
def sqlite_connect(dbapi_connection, connection_record):
    # disable pysqlite's emitting of the BEGIN statement entirely.
    # also stops it from emitting COMMIT before any DDL.
    dbapi_connection.isolation_level = None


def sqlite_begin(connection):
    # emit our own BEGIN
    connection.exec_driver_sql("BEGIN")


@qdefine
class Database:
    """An interface to a database backend.

    Attributes:
        url: The database connection url.
        engine: The SQLAlchemy engine that maintains the database connection.
        session: The database session associated with the engine.
    """

    url: URL = field(
        default=make_url("sqlite://"),
        converter=lambda s: make_url(s) if isinstance(s, str) else s,
    )
    engine: sa.engine.Engine | None = field(init=False, default=None)
    session: sa.orm.Session | None = field(init=False, default=None)

    def _get_engine(self, url, **kwargs) -> Engine:
        """Get engine with backend specific parameters.

        Args:
            url: The database URL.

        Return:
            A SQLAlchemy engine.
        """

        match url.get_backend_name():
            case "mysql":
                connect_args = dict(connect_timeout=kwargs.get("timeout"))
            case "sqlite":
                connect_args = dict()
            case backend:
                raise ValueError(
                    f"Backend {backend} is not supported. Must be mysql or sqlite."
                )

        engine = sa.create_engine(url, connect_args=connect_args)

        match url.get_backend_name():
            case "sqlite":
                event.listens_for(engine, "connect")(sqlite_connect)
                event.listens_for(engine, "begin")(sqlite_begin)

        return engine

    def connect(self, test: bool = True, timeout: int = 2):
        """Connect to the database.

        Args:
            test: If `True`, tests the database connection.
            timeout: The timeout for the database connection.

        Return:
            The SQLAlchemy engine.
        """
        self.engine = self._get_engine(self.url, timeout=timeout)
        self.session = Session(self.engine, autobegin=False, expire_on_commit=False)

        if test:
            with self.engine.begin():
                ...

        return self.engine

    def disconnect(self):
        """Disconnects from the database.

        This method closes the SQLAlchemy ORM session associated with the database and
        disposes of the associated engine.
        """
        self.session.close()
        self.engine.dispose()
        self.engine = self.session = None

    @classmethod
    def from_parameters(
        cls,
        driver: str = "sqlite+pysqlite",
        username: str | None = None,
        password: str | None = None,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        **kwargs,
    ) -> Self:
        url = URL(
            drivername=driver,
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
            query=dict(),
        )

        return cls(url=url, **kwargs)

    @property
    def username(self) -> str:
        return self.url.username

    @property
    def backend(self) -> str:
        return self.url.get_backend_name()

    @property
    def database(self) -> str:
        return self.url.database

    @session_context
    def tables(self) -> set[str]:
        """Returns the tables present in the database."""
        meta = sa.MetaData()
        meta.reflect(bind=self.engine)

        return set(meta.tables)


@qdefine
class DoltDB(Database):
    """An interface to a dolt database backend.

    See https://docs.dolthub.com/introduction/what-is-dolt

    Attributes:
        url: The database connection url.
        engine: The SQLAlchemy engine that maintains the database connection.
        session: The database session associated with the engine.
    """

    @session_context
    def current_branch(self) -> Branch:
        # with self.session.begin():
        name = self.session.scalars(sa.func.active_branch()).one()
        stmt = sa.select(DoltBranch).where(DoltBranch.name == name)

        result = self.session.scalars(stmt).one()
        branch = Branch.from_orm(result)

        return branch

    @session_context
    def get_branch(self, name: str) -> Branch | None:
        stmt = sa.select(DoltBranch).where(DoltBranch.name == name)

        result = self.session.scalars(stmt).one_or_none()

        if result is None:
            return result

        return Branch.from_orm(result)

    @session_context
    def add(self, tables: list[str] | None = None) -> None:
        dolt_add(self.session, tables=tables)

    @session_context
    def branch(
        self,
        branch: str | None = None,
        other_branch: str | None = None,
        action: str = "create",
        force: bool = False,
    ) -> None:
        dolt_branch(self.session, branch, other_branch, action, force)

    @session_context
    def checkout(self, name: str, new_branch: bool = False) -> Branch:
        dolt_checkout(self.session, name, new_branch=new_branch)

        return self.current_branch()

    @session_context
    def log(
        self,
        committer: str | None = None,
        date: pendulum.DateTime | None = None,
        date_filter: str = "after",
    ) -> list[Commit]:
        stmt = sa.select(DoltLog).order_by(DoltLog.date)

        if committer:
            stmt = stmt.where(DoltLog.committer == committer)

        results = self.session.scalars(stmt)

        commits = [Commit.from_orm(commit) for commit in results]
        return commits

    @session_context
    def commit(
        self,
        message: str,
        add: str | None = None,
        date: pendulum.DateTime | None = None,
        author: str | None = None,
        allow_empty: bool = False,
    ) -> Commit:
        author = author or self.author

        dolt_commit(self.session, message, add, date, author, allow_empty)

        result = self.session.scalars(
            sa.select(DoltLog).where(DoltLog.commit_hash == sa.func.hashof("HEAD"))
        ).one()

        return Commit.from_orm(result)

    @session_context
    def get_commit(self, commit_hash: str | None = None) -> Commit | None:
        commit_hash = commit_hash or sa.func.hashof("HEAD")
        result = self.session.scalars(
            sa.select(DoltLog).where(DoltLog.commit_hash == commit_hash)
        ).one()

        return Commit.from_orm(result)

    @session_context
    def reset(self, branch_or_commit: str | None = None, hard: bool = False) -> None:
        dolt_reset(self.session, branch_or_commit, hard)

        return self.get_commit()

    @session_context
    def status(
        self,
        staged: bool | None = None,
        status: str | None = None,
    ) -> list[Status]:
        stmt = sa.select(DoltStatus)

        if staged is not None:
            stmt = stmt.where(DoltStatus.staged == staged)

        if status:
            stmt = stmt.where(DoltStatus.status == status)

        results = self.session.scalars(stmt)

        return [Status.from_orm(s) for s in results]

    @property
    def author(self) -> str:
        groups = USERNAME_REGEX.match(self.username).groupdict()

        name = groups["name"]
        domain = groups["domain"] or "@qnl"

        return f"{name} <{name}{domain}>"

    @classmethod
    def from_parameters(
        cls,
        driver: str = "mysql+mysqldb",
        username: str | None = None,
        password: str | None = None,
        host: str = "localhost",
        port: int = 3306,
        database: str | None = None,
        **kwargs,
    ) -> Self:
        return super().from_parameters(
            driver=driver,
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
            **kwargs,
        )


__all__ = [
    "Branch",
    "Commit",
    "Database",
    "DoltDB",
]
