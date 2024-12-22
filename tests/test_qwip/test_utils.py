import pytest
from IPython.testing.globalipapp import get_ipython

from qwip.utils import deprecated


@pytest.fixture(scope="session")
def ipython_session():
    ip = get_ipython()
    yield ip
    ip.run_cell("exit()")


@pytest.fixture
def ipython(ipython_session):
    ipython_session.run_line_magic("load_ext", line="qwip.utils")

    yield ipython_session

    ipython_session.run_line_magic("reset", line="-f")


class TestSlackMagic: ...


@deprecated(version="23.2.0", removed="23.3.0")
def import_qtrl(): ...


@deprecated(version="23.2.0", removed="23.3.0")
class ImportQTRL: ...


class TestDeprecated:
    @deprecated(version="23.2.0", removed="23.3.0")
    def import_qtrl(self): ...

    def test_deprecated_function(self):
        with pytest.deprecated_call():
            import_qtrl()

    def test_deprecated_class(self):
        with pytest.deprecated_call():
            ImportQTRL()

    def test_deprecated_method(self):
        with pytest.deprecated_call():
            self.import_qtrl()
