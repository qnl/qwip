import numpy as np
import pytest
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014

from qwip.backends.tektronix import (
    TektronixBackend,
    TektronixChannel,
    TektronixCompiler,
    TektronixProgram,
)
from qwip.sequencer.compilation import ChannelGroup, ChannelInfo, TriggerInfo
from qwip.sequencer.elements import SequenceElement
from qwip.sequencer.phase_tracker import ModulationFrequency
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.waveform import (
    CWWaveform,
    GaussianWaveform,
    ModulatedWaveform,
    SquareWaveform,
    TriggeredWaveform,
    VirtualZWaveform,
)


@pytest.fixture(scope="module")
def backend():
    backend = TektronixBackend.connect(ip="192.168.1.77")
    yield backend

    backend.awg.close()


# class TestTetronixExecutable:
#     @pytest.mark.parametrize("channels", [(0, 1), (0, 1, 2, 3)])
#     def test_init(self, channels):
#         exe = TektronixProgram(device="tektronix", channels=channels)
#         assert (
#             len(exe.waveforms)
#             == len(exe.marker1s)
#             == len(exe.marker2s)
#             == len(channels)
#         )


class TestTektronixChannel:
    @pytest.fixture(scope="class")
    def awg(self, backend):
        return backend.awg

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


class TestTektronixCompiler:
    @pytest.fixture
    def compiler(self):
        dac = ChannelGroup.from_channels(
            channels=(
                ChannelInfo("Q0_I", 0),
                ChannelInfo("Q0_Q", 1),
                ChannelInfo("Q1_I", 0),
                ChannelInfo("Q1_Q", 1),
                ChannelInfo("RO_I", 2),
                ChannelInfo("RO_Q", 3),
                ChannelInfo("RO_marker", 3, subchannel=1),
            ),
            sample_rate=1.0e9,
            name="tektronix",
        )

        adc = ChannelGroup.from_channels(
            channels=(ChannelInfo("RO", 0, read=True),),
            sample_rate=1.0e9,
            trigger=TriggerInfo(device="tektronix", index=3, subchannel=0),
            name="adc",
        )

        demod = ChannelGroup.from_channels(
            channels=(
                ChannelInfo("R0", 0),
                ChannelInfo("R1", 1),
            ),
            sample_rate=1.0e9,
            trigger=TriggerInfo(device="tektronix", index=3, subchannel=0),
            name="demod",
        )

        modulations = dict(
            mod_Q0=ModulationFrequency(200e6),
            mod_Q1=ModulationFrequency(150e6),
            mod_R0=ModulationFrequency(-300e6),
            mod_R1=ModulationFrequency(-400e6),
        )

        return TektronixCompiler.from_channel_groups(
            [dac, adc, demod], modulations=modulations
        )

    @pytest.fixture
    def pulses(self):
        Q0_X90 = ModulatedWaveform(
            name="Q0_X90",
            envelope=GaussianWaveform(width=25e-9, amplitude=0.1),
            modulation=CWWaveform(frequency="mod_Q0", channels=("Q0_I", "Q0_Q")),
        )

        Q1_X90 = ModulatedWaveform(
            name="Q1_X90",
            envelope=GaussianWaveform(width=25e-9, amplitude=0.1),
            modulation=CWWaveform(frequency="mod_Q1", channels=("Q1_I", "Q1_Q")),
        )

        Q0_Z90 = VirtualZWaveform(name="Q0_Z", mod_key="mod_Q0", phase=90)
        Q1_Z90 = VirtualZWaveform(name="Q0_Z", mod_key="mod_Q1", phase=90)

        R0 = ModulatedWaveform(
            name="R0",
            envelope=SquareWaveform(width=1e-6, amplitude=0.2),
            modulation=CWWaveform(channels=("RO_I", "RO_Q"), frequency="mod_R0"),
        )

        R1 = ModulatedWaveform(
            name="R1",
            envelope=SquareWaveform(width=1e-6, amplitude=0.25),
            modulation=CWWaveform(channels=("RO_I", "RO_Q"), frequency="mod_R1"),
        )

        D0 = ModulatedWaveform(
            name="D0",
            envelope=SquareWaveform(width=1e-6, amplitude=1),
            modulation=CWWaveform(channels=("R0",), frequency="mod_R0"),
        )

        D1 = ModulatedWaveform(
            name="D0",
            envelope=SquareWaveform(width=1e-6, amplitude=1),
            modulation=CWWaveform(channels=("R1",), frequency="mod_R1"),
        )

        return dict(
            Q0_X90=Q0_X90,
            Q1_X90=Q1_X90,
            Q0_Z90=Q0_Z90,
            Q1_Z90=Q1_Z90,
            R0=R0,
            R1=R1,
            read=SquareWaveform(width=1e-6, channels=("RO",)),
            D0=D0,
            D1=D1,
        )

    def test_compile(self, compiler, pulses, backend):
        se = SequenceElement()
        se.add_waveform(pulses["Q0_X90"])
        se.add_waveform(pulses["Q0_X90"], pulses["Q0_X90"].width)
        se.add_waveform(pulses["R0"], 2 * pulses["Q0_X90"].width + "delay")

        ro = SequenceElement()
        ro.add_waveform(pulses["read"])
        ro.add_waveform(pulses["D0"])
        ro.add_waveform(pulses["D1"])

        readout = TriggeredWaveform(target=ro, width=2e-9, channels=("RO_marker",))

        se.add_waveform(readout, 2 * pulses["Q0_X90"].width + "delay")

        seq = Sequence.sweep(se, delay=np.linspace(0, 100, 11) * 1e-6)

        exe = compiler.compile(seq)

        backend.upload(exe)


class TestTektronixBackend:
    def test_connect(self, backend):
        assert backend.awg.IDN()["vendor"] == "TEKTRONIX"

    def test_update_channel(self, backend):
        print(backend)
