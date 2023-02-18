import pytest

import sqlalchemy as sa
from sqlalchemy import MetaData, Table, Column, String, Integer

from qwip.config.dolt import(
    DoltLog,
    DoltBranch,
    DoltCommit,
    DoltDiff,
    DoltTable,
    dolt_add,
    dolt_branch,
    dolt_checkout,
    dolt_commit,
    dolt_reset
)
from qwip.config.database import Branch, Commit


class TestDolt:
    @pytest.fixture
    def new_table(self, doltdb):
        metadata = MetaData()

        test_table = DoltTable(
            'textbooks',
            metadata,
            Column('id', sa.Integer, primary_key=True, autoincrement=True),
            Column('title', sa.Text),
            Column('author', sa.Text),
            Column('edition', sa.Integer),
        )

        test_table.create_system_tables()
        try:
            with doltdb.engine.begin() as connection:
                commit_hash = connection.execute(
                    sa.func.HASHOF('main')
                ).scalars().one()

            metadata.create_all(doltdb.engine, tables=[test_table])
            yield test_table
        finally:
            with doltdb.engine.begin() as connection:
                dolt_reset(connection, commit_hash, hard=True)

            metadata.drop_all(doltdb.engine, tables=[test_table])

    def test_new_table(self, session, new_table):
        log = session.execute(sa.select(DoltLog)).scalars().first()
        assert log.message == 'Initialize data repository'

        branch = session.execute(sa.select(DoltBranch)).scalars().one()
        commit = session.execute(sa.select(DoltCommit)).scalars().first()
        
        assert branch.name == 'main'
        assert branch.hash == log.commit_hash == commit.commit_hash

        nrows = session.execute(sa.select(new_table)).rowcount
        assert nrows == 0

    def test_linear_history(self, session, new_table):
        table = new_table

        stmt = sa.insert(table).values(
            title='Classical Electrodynamics',
            author='Jackson',
            edition=3
        )

        session.execute(stmt)

        dolt_commit(session, 'Add Jackson.', add='all')

        stmt = sa.insert(table).values(
            title='Principles of Quantum Mechanics',
            author='Shankar',
            edition=1
        )

        session.execute(stmt)

        dolt_commit(session, 'Add Shankar.', add='all')

        stmt = (
            sa.update(table)
            .where(table.c.author == 'Shankar')
            .values(edition=2)
        )
        session.execute(stmt)

        dolt_commit(session, 'Update Shankar edition.', add='all')

        result = session.execute(sa.select(DoltLog)).scalars()
        commits = [Commit.from_orm(row) for row in result]

        assert len(commits) == 4
        assert set(c.message for c in commits) == {
            'Initialize data repository',
            'Add Jackson.',
            'Add Shankar.',
            'Update Shankar edition.'
        }