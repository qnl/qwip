import pytest

from qwip.backends.vna import VNABackend
from qwip.processing.processors import IQResult


@pytest.mark.skip_instrument("vna")
class TestVNABackend:
    @pytest.fixture
    def backend(self, instrument_server):
        vna = instrument_server["vna"]

        return VNABackend(vna=vna)

    def test_acquire(self, backend):
        result = backend.acquire()[backend.vna.name]

        assert isinstance(result, IQResult)
