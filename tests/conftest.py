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