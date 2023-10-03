import pytest

from qwip.backends.tektronix import (
    TektronixBackend,
    TektronixChannel,
)

@pytest.fixture
def awg(instrument_server):
    return instrument_server["tektronix"]


@pytest.fixture
def backend(awg):
    return TektronixBackend(device=awg)


@pytest.mark.skip_instrument("tektronix")
class TestTektronixChannel:
    @pytest.mark.parametrize("index", list(range(4)))
    def test_write_settings(self, index, awg):
        ch = TektronixChannel.from_awg(awg, index=index)

        ch.amplitude = 1.0
        ch.offset = 0.1
        ch.marker_1 = (0, 1.0)
        ch.write_settings(awg)

        assert ch == TektronixChannel.from_awg(awg, index=index)

    def test_connect(self, awg):
        assert awg.IDN()["vendor"] == "TEKTRONIX"

    def test_amplitudes(self, awg):
        print(awg.num_channels)


@pytest.mark.skip_instrument("tektronix")
class TestTektronixBackend:
    def test_connect(self, backend):
        assert backend.device.IDN()["vendor"] == "TEKTRONIX"

