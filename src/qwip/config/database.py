import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import make_url, URL

import re
import pendulum
import functools
from pendulum import DateTime
from attrs import field
from typing import TypeVar, Generic, Any
from typing_extensions import Self
from qwip.settings.settings import Settings, qdefine, qfrozen

from qwip.config.dolt import (
    DoltBranch,
    DoltCommit,
    DoltLog,
    dolt_add,
    dolt_branch,
    dolt_checkout,
    dolt_commit,
)
from qwip.config.models import Parameter

SHORT_HASH_LEN = 8

@qfrozen
class Commit:
    hash: str = field(repr=lambda h: h[:SHORT_HASH_LEN])
    committer: str
    email: str
    date: pendulum.DateTime = field(repr=lambda dt: dt.in_tz('local').isoformat())
    message: str

    @classmethod
    def from_orm(cls, model: DoltCommit):
        return cls(
            hash=model.commit_hash,
            committer=model.committer,
            email=model.email,
            date=model.date,
            message=model.message
        )

@qfrozen
class Branch:
    name: str
    latest: Commit

    @classmethod
    def from_orm(cls, model: DoltBranch):
        commit = Commit(
            hash=model.hash,
            committer=model.latest_committer,
            email=model.latest_committer_email,
            date=model.latest_commit_date,
            message=model.latest_commit_message
        )

        return cls(
            name=model.name,
            latest=commit
        )


@qdefine
class ConfigDB:
    database: str | None = None
    username: str 
    password: str
    host: str
    port: int = 3306

    engine: sa.engine.Engine | None = None
    session: sa.orm.sessionmaker | None = None

    def connect(self):
        url = URL.create(
            drivername='mysql',
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database
        )
        engine = sa.create_engine(url)
        self.engine = engine
        self.session = sessionmaker(engine)
        
        return engine

    def current_branch(self) -> Branch:
        with self.session.begin() as session:
            name = session.execute(sa.func.active_branch()).scalar_one()
            stmt = sa.select(DoltBranch).where(DoltBranch.name == name)

            result = session.execute(stmt).scalar_one()
            branch = Branch.from_orm(result)

        return branch

    def get_branch(self, name: str) -> Branch | None:
        with self.session.begin() as session:
            stmt = sa.select(DoltBranch).where(DoltBranch.name == name)

            result = session.execute(stmt).scalar_one_or_none()

            if result is None:
                return result

            return Branch.from_orm(result)

    def add(
        self,
        tables: list[str] | None = None
    ) -> None:
        with self.session.begin() as connection:
            dolt_add(connection, tables=tables)

    def branch(
        self,
        branch: str | None = None,
        other_branch: str | None = None,
        action: str = 'create',
        force: bool = False,
    ) -> None:
        with self.session.begin() as connection:
            dolt_branch(connection, branch, other_branch, action, force)

    def checkout(self, name: str, new_branch: bool = False) -> Branch:
        with self.session.begin() as connection:
            dolt_checkout(connection, name, new_branch=new_branch)

        return self.get_branch(name)

    def commit(
        self,
        message: str,
        add: str | None = None,
        date: pendulum.DateTime | None = None,
        author: str | None = None,
        allow_empty: bool = False
    ) -> Commit:
        with self.session.begin() as session:
            dolt_commit(session, message, add, date, author, allow_empty)

            result = session.execute(
                sa.select(DoltLog).where(
                    DoltLog.commit_hash == sa.func.hashof('HEAD')
                )
            ).scalar_one()

            return Commit.from_orm(result)

    def get_commit(
        self,
        commit_hash: str
    ) -> Commit | None:
        with self.session.begin() as session:
            result = session.execute(
                sa.select(DoltLog).where(
                    DoltLog.commit_hash == commit_hash
                )
            ).scalar_one()

            return Commit.from_orm(result)

    @classmethod
    def from_url(cls, db_url: str) -> Self:
        url = make_url(db_url)

        return cls(
            database=url.database,
            username=url.username,
            password=url.password,
            host=url.host,
            port=url.port
        )
