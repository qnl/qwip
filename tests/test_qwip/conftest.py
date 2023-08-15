import re
import sys
from pathlib import Path

import numpy as np
import pendulum
import pytest
from loguru import logger
from numpy.random import default_rng
from sqlalchemy.engine import make_url

try:
    from qtrl.settings import Settings

    Settings.setup = Settings.OFFLINE  # ruff: noqa: E402
except ModuleNotFoundError:
    ...

from qwip.config.interface import ConfigDB, OfflineConfigDB
from qwip.config.schema import ConfigSchema
from qwip.data.models import *
from qwip.database.database import Database, DoltDB
from qwip.database.metadata import QWIP_DB_METADATA
from qwip.qpu.qpu import QPU


def ignore_config_commit(record: dict) -> bool:
    """Ignores warning messages from database config not matching current commit."""
    should_log = not (
        record["module"] == "interface" and record["function"] == "init_config"
    )
    return should_log


logger.remove()
logger.add(sys.stdout, level="WARNING", filter=ignore_config_commit)


def pytest_addoption(parser):
    parser.addoption(
        "--db_url", action="store", default="sqlite://", help="The database url."
    )
    parser.addoption("--seed", action="store", default=0, help="Seed used for all rng.")


@pytest.fixture(scope="session")
def db_url(request):
    return make_url(request.config.getoption("--db_url"))


@pytest.fixture(scope="session")
def seed(request):
    return int(request.config.getoption("--seed"))


@pytest.fixture
def rng(seed):
    return default_rng(seed)


@pytest.fixture
def fixed_time():
    now = pendulum.datetime(2015, 3, 14, 9, 26, 53, tz="America/Los_Angeles")
    with pendulum.test(now):
        yield now


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


@pytest.fixture(scope="session")
def skip_dolt(db_url):
    if db_url.get_backend_name() != "mysql":
        pytest.skip("Skipping dolt tests with offline database.")


@pytest.fixture(scope="session")
def database(db_url):
    if db_url.get_backend_name() == "sqlite":
        db_cls = Database
    else:
        db_cls = DoltDB

    db = db_cls(url=db_url)
    db.connect(test=True)

    yield db


@pytest.fixture(scope="module")
def models(database):
    tables = [t for n, t in QWIP_DB_METADATA.tables.items() if not n.startswith("dolt")]
    QWIP_DB_METADATA.create_all(database.engine, tables=tables)

    yield QWIP_DB_METADATA

    QWIP_DB_METADATA.drop_all(database.engine, tables=tables)


@pytest.fixture(scope="function")
def session(database):
    with database.session.begin_nested():
        yield database.session
        database.session.rollback()


@pytest.fixture(scope="function")
def dolt_session(database):
    with database.session.begin():
        commit_hash = database.get_commit().hash
        yield database.session
        database.reset(commit_hash, hard=True)


@pytest.fixture
def session_with_models(session, models):
    yield session


@pytest.fixture
def configdb_01():
    db_file = Path(__file__).parent / r"sample_configs/config_03.sqlite"
    db = OfflineConfigDB(url=f"sqlite:///{db_file}", schema=ConfigSchema)
    db.connect()

    with db.session.begin_nested():
        yield db
        db.session.rollback()


# config_02: missing many readout parameters
# config_03: added chi, kappa, eta to readouts


@pytest.fixture
def qpu_01(configdb_01):
    qpu = QPU.load(configdb_01)

    # Is this necessary?
    for k, system in qpu.subsystems.items():
        system.modulation_name = (
            "mod_{name}_{mod_key}" if k.startswith("Q") else "mod_{name}"
        )
        qpu.update_modulations()
    return qpu
