import re
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.engine import make_url

try:
    from qtrl.settings import Settings

    Settings.setup = Settings.OFFLINE  # ruff: noqa: E402
except ModuleNotFoundError:
    ...

from qwip.config.database import ConfigDB, Database, DoltDB, OfflineConfigDB
from qwip.config.dolt import dolt_reset
from qwip.config.metadata import QWIP_DB_METADATA
from qwip.processing.data_processor import DATA_PROCESSORS, ReadoutPipeline
from qwip.processing.processors import GMMClassification, IQRotation
from qwip.qpu.qpu import QPU
from qwip.qpu.systems import ReadoutResonator, Transmon
from qwip.sequencer.compilation import ChannelGroup, ChannelInfo, WaveformSequencer


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


## ==================== Processing ==================== ##


@pytest.fixture
def pipeline():
    processors = []
    for processor_cls in DATA_PROCESSORS.data_processors():
        if issubclass(processor_cls, GMMClassification | IQRotation):
            continue
        else:
            processors.append(processor_cls())

    processors.append(
        GMMClassification(
            measurement_key="R0",
            means=np.array([[0, 3], [0, -3]]).astype(float),
            covariances=np.array([1, 1]).astype(float),
            num_states=2,
        )
    )

    processors.append(
        GMMClassification(
            measurement_key="R1",
            means=np.array([[3, 0], [-3, 0]]).astype(float),
            covariances=np.array([1, 1]).astype(float),
            num_states=2,
        )
    )

    return ReadoutPipeline(name="default", processors=processors)


## ==================== Sequencer ==================== ##


@pytest.fixture
def sequencer():
    dac = ChannelGroup.from_channels(
        channels=(
            ChannelInfo("Q0_I", 0),
            ChannelInfo("Q0_Q", 1),
            ChannelInfo("Q1_I", 2),
            ChannelInfo("Q1_Q", 3),
        ),
        sample_rate=2.4e9,
        name="seq",
    )

    adc = ChannelGroup.from_channels(
        channels=(ChannelInfo("RO_I", 0), ChannelInfo("RO_Q", 1)),
        sample_rate=1.8e9,
        name="readout",
    )

    sequencer = WaveformSequencer.from_channel_groups([dac, adc])
    sequencer.readout_qubits = [0, 1]
    return sequencer


## ==================== QuantumSystems ==================== ##


@pytest.fixture
def subsystems():
    subsystems = dict(
        Q0=Transmon(name="Q0", frequency=5.0e9),
        Q1=Transmon(name="Q1", frequency=5.5e9),
        R0=ReadoutResonator(name="R0", frequency=6.0e9),
        R1=ReadoutResonator(name="R1", frequency=6.5e9),
    )

    return subsystems


## ==================== QPU ==================== ##


@pytest.fixture
def qpu(sequencer, pipeline, subsystems):
    qpu = QPU(
        db=ConfigDB.from_parameters(
            username="test", password="test", host="localhost", database="test"
        ),
        subsystems=subsystems,
        sequencer=sequencer,
        pipeline=pipeline,
    )

    return qpu
