import re
from pathlib import Path

import pytest

def pytest_addoption(parser):
    parser.addoption(
        '--db_url', action='store', default=None, help='The database url.'
    )

    parser.addoption(
        '--test_db', action='store', default='test_db', help='The name of the test database.'
    )

@pytest.fixture(scope='module')
def db_url(request):
    return request.config.getoption('--db_url')

@pytest.fixture(scope='module')
def test_db(request):
    return request.config.getoption('--test_db')

@pytest.fixture
def data_file(request):
    fspath = Path(request.fspath)

    file = fspath.name
    name = re.match(r'test_(?P<name>.*)\.py', str(file)).group('name')
    datadir = fspath.parent / name

    if request.cls:
        datadir = datadir / request.cls.__name__.lstrip('Test')

    return datadir / f'{request.node.name.lstrip("test_")}.txt'