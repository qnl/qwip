import pytest
from IPython.testing.globalipapp import get_ipython

from qwip import qsettings


@pytest.fixture(scope="session")
def ipython_session():
    ip = get_ipython()
    yield ip
    ip.run_cell("exit()")
    print("Exiting")


@pytest.fixture
def ipython(ipython_session):
    ipython_session.run_line_magic("load_ext", line="qwip.utils")

    yield ipython_session

    ipython_session.run_line_magic("reset", line="-f")


class TestSlackMagic:
    ...
