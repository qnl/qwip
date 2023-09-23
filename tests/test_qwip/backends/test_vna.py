import pytest

from qwip.backends.vna import VNABackend, VNAExecutable
from qwip.processing.processors import IQResult


@pytest.mark.skip_instrument("vna")
class TestVNABackend:
    @pytest.fixture
    def backend(self, instrument_server):
        vna = instrument_server["vna"]

        return VNABackend(vna=vna)

    def test_upload_none(self, backend):
        backend.vna.start(6e9)
        backend.vna.stop(7e9)
        backend.vna.points(1001)
        backend.vna.power(-70)
        backend.vna.averages(1)
        backend.vna.electrical_delay(0)
        backend.vna.if_bandwidth(1000)
        backend.vna.trace("S21")

        exe = VNAExecutable()
        backend.upload(exe)

        assert exe == VNAExecutable(
            start=6e9,
            stop=7e9,
            points=1001,
            power=-70,
            averages=1,
            delay=0,
            if_bandwidth=1000,
            meas="S21",
        )

    def test_upload_explicit(self, backend):
        exe = VNAExecutable(
            start=5e9,
            stop=6e9,
            points=501,
            power=-60,
            averages=2,
            delay=100e-9,
            if_bandwidth=500,
            meas="S11",
        )
        backend.upload(exe)

        assert backend.vna.start() == 5e9
        assert backend.vna.stop() == 6e9
        assert backend.vna.points() == 501
        assert backend.vna.power() == -60
        assert backend.vna.averages() == 2
        assert backend.vna.averages_enabled() is True
        assert backend.vna.electrical_delay() == 100e-9
        assert backend.vna.if_bandwidth() == 500
        assert backend.vna.trace() == "S11"

    def test_acquire(self, backend):
        result = backend.acquire()[backend.vna.name]

        assert isinstance(result, IQResult)
        assert list(result.data.index.names) == ["frequency"]
        assert len(result.data) == backend.vna.points()
