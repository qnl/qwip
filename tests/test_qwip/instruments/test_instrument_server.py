import pytest

from qwip.instruments.instrument_server import InstrumentServer


class TestInstrumentServer:
    @pytest.mark.skip_instrument("vna")
    def test_skip(self, instrument_server):
        assert "vna" in instrument_server

    def test_get(self, instrument_server):
        assert instrument_server["dummy"] == instrument_server.instruments["dummy"]
