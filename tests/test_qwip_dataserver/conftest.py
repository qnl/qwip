import pytest
from fastapi.testclient import TestClient

from qwip_dataserver.settings import get_settings


@pytest.fixture(scope="session")
def settings():
    settings = get_settings()

    yield settings


@pytest.fixture(scope="session")
def client(tmp_path_factory, settings):
    settings.DATASERVER_ROOT = tmp_path_factory.getbasetemp()

    from qwip_dataserver.server import app

    client = TestClient(app)

    yield client
