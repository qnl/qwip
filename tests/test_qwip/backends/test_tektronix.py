import numpy as np
import pytest

from qwip.backends.tektronix import TektronixBackend, TektronixChannel, TektronixProgram
from qwip.sequencer.compilation import QWiPExecutable


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

    def test_amplitudes(self, awg): ...


@pytest.mark.skip_instrument("tektronix")
class TestTektronixBackend:
    @pytest.fixture
    def program(self):
        program = TektronixProgram(
            device="tektronix",
            channels=[0, 1, 2, 3],
            waveforms=tuple(
                [
                    np.sin(2 * np.pi * ch * np.r_[: 250 * (i + 1)] / 10).astype(
                        np.float32
                    )
                    for i in range(2)
                ]
                for ch in range(1, 5)
            ),
            marker1s=tuple(
                [np.zeros(250 * (i + 1), np.uint8) for i in range(2)] for _ in range(4)
            ),
            marker2s=tuple(
                [np.zeros(250 * (i + 1), np.uint8) for i in range(2)] for _ in range(4)
            ),
            repeats=[1, 1],
            waits=[1, 1],
            go_tos=[1, 0],
            jump_tos=[0, 0],
        )

        return program

    def test_connect(self, backend):
        assert backend.device.IDN()["vendor"] == "TEKTRONIX"

    def test_sample_rate(self, backend):
        backend.sample_rate = 1e9
        assert backend.sample_rate == 1e9
        assert backend.device.clock_freq() == backend.sample_rate

    def test_start(self, backend, program):
        backend.upload(exe=QWiPExecutable(programs=dict(tektronix=program)))
        backend.start()

        assert backend.active is True
        assert [getattr(backend.device, f"ch{i + 1}_state")() for i in range(4)] == [
            1,
            1,
            1,
            1,
        ]

        backend.stop()

    def test_stop(self, backend, program):
        backend.upload(exe=QWiPExecutable(programs=dict(tektronix=program)))
        backend.start()
        backend.stop()

        assert backend.active is False
        assert [getattr(backend.device, f"ch{i + 1}_state")() for i in range(4)] == [
            0,
            0,
            0,
            0,
        ]
