import httpx
import pytest
from fastapi.testclient import TestClient

from qwip.data.storage import HTTPStorageBackend
from qwip_dataserver.settings import get_settings


@pytest.fixture(scope="class")
def settings(tmp_path_factory):
    settings = get_settings()
    settings.DATASERVER_ROOT = tmp_path_factory.mktemp("pytest")

    yield settings


@pytest.fixture(scope="class")
def client(settings, dataserver):
    if dataserver:
        client = httpx.Client(base_url=dataserver)
    else:
        from qwip_dataserver.server import app

        client = TestClient(app)

    with client:
        yield client


@pytest.fixture(scope="class")
def storage(client, request):
    if client.app is None:
        for marker in request.node.iter_markers():
            if marker.name == "skip_dataserver":
                pytest.skip("Skipping test on live dataserver.")

    yield HTTPStorageBackend(client=client)
