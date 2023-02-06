import pytest
import sqlalchemy as sa

from qwip.config.dolt import dolt_reset
from qwip.config.database import ConfigDB
from qwip.config.metadata import QWIP_DB_METADATA

@pytest.fixture(scope='module')
def configdb(db_url, test_db):
    configdb = ConfigDB.from_url(f'{db_url}/{test_db}')
    engine = configdb.connect()

    yield configdb

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
def session(configdb):
    with configdb.session.begin():
        yield configdb.session

@pytest.fixture
def reset_models(session, models):
    yield models

    dolt_reset(session, 'main', hard=True)