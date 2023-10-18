import httpx
import pytest
from fastapi.testclient import TestClient

from qwip.data.storage import HTTPStorageBackend
from qwip_dataserver.settings import get_settings


@pytest.fixture(scope="session")
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


@pytest.fixture
def storage(client, request):
    for marker in request.node.iter_markers():
        if marker.name != "skip_dataserver":
            continue

        skipif = "app" if hasattr(client, "app") else "live"

        match marker.args:
            case (tp, *args):
                ...
            case ():
                tp = "live"

        if tp.lower() == skipif:
            pytest.skip(f"Skipping test on {tp} dataserver.")

    yield HTTPStorageBackend(client=client)
