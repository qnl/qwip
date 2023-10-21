import pytest

try:
    from qwip.backends.alazar import AlazarBackend
except ImportError:
    pytest.skip("Alazar dependencies not installed.", allow_module_level=True)


@pytest.fixture
def adc(instrument_server):
    return instrument_server["alazar"]


@pytest.fixture
def backend(adc):
    return AlazarBackend(device=adc)


@pytest.mark.skip_instrument("alazar")
class TestAlazarBackend:
    def test_sample_rate(self, backend):
        backend.sample_rate = 1e9
        assert backend.sample_rate == 1e9
