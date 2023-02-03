import pytest
import sqlalchemy as sa

from sqlalchemy import MetaData, Table, Column, String, Integer
from sqlalchemy.orm import registry, Session

import attrs
import pendulum

from qwip.settings.settings import qdefine, Settings
from qwip.config.database import (
    ConfigDB,
    Commit,
    Branch,
    SettingsFolder
)
from qwip.config.models import(
    Parameter,
    Folder,
)
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
from qwip.config.metadata import QWIP_DB_METADATA

@pytest.fixture(scope='module')
def engine():
    engine = ConfigDB(
        database='acadia_v0p1',
        username='qnl',
        password='database',
        host='192.168.1.5'
    ).connect()

    QWIP_DB_METADATA.create_all(engine)

    yield engine


@pytest.fixture
def db_session(engine):
    session = Session(engine)

    yield session

    session.rollback()
    session.close()

@pytest.fixture(scope='module')
def configdb(db_url, test_db):
    configdb = ConfigDB.from_url(f'{db_url}/{test_db}')
    engine = configdb.connect()

    yield configdb

@pytest.fixture
def branch_name():
    return pendulum.now().format('YYYY-MM-DD_HH-mm-ss')

@pytest.fixture(scope='module')
def models(configdb):
    with configdb.session.begin():
        commit_hash = configdb.session.scalars(
            sa.func.HASHOF('main')
        ).one()

    tables = [t for n, t in QWIP_DB_METADATA.tables.items() if not n.startswith('dolt')]
    
    QWIP_DB_METADATA.create_all(configdb.engine, tables=tables)

    configdb.commit('Created models.', add='all')

    yield QWIP_DB_METADATA

    with configdb.session.begin():
        dolt_reset(configdb.session, commit_hash, hard=True)

    QWIP_DB_METADATA.drop_all(configdb.engine, tables=tables)

@pytest.fixture
def reset_models(session, models):
    yield models

    dolt_reset(session, 'main', hard=True)


@pytest.fixture
def session(configdb):
    with configdb.session.begin():
        yield configdb.session

class TestDolt:
    @pytest.fixture
    def new_table(self, configdb):
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
            with configdb.engine.begin() as connection:
                commit_hash = connection.execute(
                    sa.func.HASHOF('main')
                ).scalars().one()

            metadata.create_all(configdb.engine, tables=[test_table])
            yield test_table
        finally:
            with configdb.engine.begin() as connection:
                dolt_reset(connection, commit_hash, hard=True)

            metadata.drop_all(configdb.engine, tables=[test_table])

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


class TestConfigDB:
    def test_current_branch(self, configdb):
        branch = configdb.current_branch()
        assert branch.name == 'main'

    def test_create_delete_branch(self, configdb):
        configdb.branch('new')
        branch = configdb.get_branch('new')
        assert branch.name == 'new'

        configdb.branch('new', action='delete', force=True)
        assert configdb.get_branch('new') is None

    def test_checkout_branch(self, configdb):
        configdb.branch('new')
        branch = configdb.get_branch('new')

        current = configdb.checkout('new')

        assert branch == current

        configdb.checkout('main')
        configdb.branch('new', action='delete', force=True)

class TestFolder:
    def test_select_insert(self, session, reset_models):
        hardware = Folder(name='hardware')
        lo = Folder(name='local_oscillators', parent=hardware)
        dc = Folder(name='dc_sources', parent=hardware)

        session.add(hardware)
        session.flush()

        results = session.scalars(sa.select(Folder)).all()

        assert len(results) == 3
        assert results == [hardware, lo, dc]

    def test_select_none(self, session, reset_models):
        folders = session.scalars(sa.select(Folder)).one_or_none()
        assert folders is None

    def test_select_condition(self, session, reset_models):
        hardware = Folder(name='hardware')
        lo = Folder(name='local_oscillators', parent=hardware)
        dc = Folder(name='dc_sources', parent=hardware)
        yoko = Folder(name='yokos', parent=dc)
        qubits = Folder(name='qubits')

        session.add_all([hardware, qubits])
        session.flush()

        assert session.scalars(sa.select(Folder)).all() == [hardware, qubits, lo, dc, yoko]

        assert session.scalars(
            sa.select(Folder)
            .where(Folder.parent_id == None)
        ).all() == [hardware, qubits]
        assert session.scalars(
            sa.select(Folder)
            .where(Folder.parent_id == hardware.folder_id)
        ).all() == [lo, dc]
        
class TestParameter:
    def test_select(self, session, reset_models):
        params = session.scalars(sa.select(Parameter)).one_or_none()
        assert params is None

class TestSettingsInDB:
    def test_settings(self, session, reset_models):
        params = SettingsFolder(session=session)

        hardware = Folder(name='hardware')
        lo = Folder(name='local_oscillators', parent=hardware)
        dc = Folder(name='dc_sources', parent=hardware)

        session.add(hardware)
        session.flush()

        stmt = sa.select(Folder).where(Folder.parent_id == None)
        result = session.scalars(stmt).all()
        print(result)

        # print(params.folder_id)
        # print(params['hardware'])
        print(params)
        print(params['hardware/local_oscillators'])

        print(list(params.flatitems()))
        print('hardware/local_oscillators' in params)

        # print(list(params.items()))
        # stmt = sa.select(sa.func.count()).select_from(Folder)

        # result = session.scalar(
        #     stmt
        # )
        # print(result)

        # print(params['hardware'].folder.subfolders)
        # result = type(params)._get_root_folders(session).all()
        # print(result)