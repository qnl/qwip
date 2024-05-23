import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    from qwip_data_api.server import app

    client = TestClient(app)

    yield client
