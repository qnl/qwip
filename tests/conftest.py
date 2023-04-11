import re
from pathlib import Path

import pytest
import sqlalchemy as sa

try:
    from qwip.config.database import DoltDB
    from qwip.config.dolt import dolt_reset
    from qwip.config.metadata import QWIP_DB_METADATA
except ModuleNotFoundError:
    ...


def pytest_addoption(parser):
    parser.addoption("--db_url", action="store", default=None, help="The database url.")

    parser.addoption(
        "--test_db",
        action="store",
        default="test_db",
        help="The name of the test database.",
    )


@pytest.fixture(scope="module")
def db_url(request):
    return request.config.getoption("--db_url")


@pytest.fixture(scope="module")
def test_db(request):
    return request.config.getoption("--test_db")


@pytest.fixture
def data_file(request):
    fspath = Path(request.fspath)

    file = fspath.name
    name = re.match(r"test_(?P<name>.*)\.py", str(file)).group("name")
    datadir = fspath.parent / name

    if request.cls:
        datadir = datadir / request.cls.__name__.lstrip("Test")

    filename = re.match(r"test_(.*)", request.node.name).groups()[0]

    return datadir / f"{filename}.txt"


@pytest.fixture(scope="module")
def doltdb(db_url, test_db):
    doltdb = DoltDB.from_url(f"{db_url}/{test_db}")
    engine = doltdb.connect()

    yield doltdb


@pytest.fixture(scope="module")
def models(doltdb):
    with doltdb.session.begin():
        commit_hash = doltdb.session.scalars(sa.func.HASHOF("main")).one()

    tables = [t for n, t in QWIP_DB_METADATA.tables.items() if not n.startswith("dolt")]

    QWIP_DB_METADATA.create_all(doltdb.engine, tables=tables)

    doltdb.commit("Created models.", add="all")

    yield QWIP_DB_METADATA

    with doltdb.session.begin():
        dolt_reset(doltdb.session, commit_hash, hard=True)

    QWIP_DB_METADATA.drop_all(doltdb.engine, tables=tables)


@pytest.fixture
def session(doltdb):
    with doltdb.session.begin():
        yield doltdb.session


@pytest.fixture
def reset_models(session, models):
    yield models

    dolt_reset(session, "main", hard=True)
