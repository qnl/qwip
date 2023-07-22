import httpx
import pytest
from fastapi.testclient import TestClient

from qwip_api.server import app


def pytest_addoption(parser):
    parser.addoption(
        "--user", action="store", default="pytest", help="The database user."
    )
    parser.addoption("--password", action="store", help="The database user password.")


@pytest.fixture(scope="session")
def db_user(request):
    return request.config.getoption("--user")


@pytest.fixture(scope="session")
def db_password(request):
    return request.config.getoption("--password")


@pytest.fixture(scope="session")
def auth(db_user, db_password):
    return httpx.BasicAuth(db_user, db_password)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client
