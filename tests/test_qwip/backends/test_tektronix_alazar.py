import numpy as np
import pytest

from qwip.backends.alazar import AlazarBackend, AlazarCompiler
from qwip.backends.backend import QWiPBackend
from qwip.backends.tektronix import (
    TektronixBackend,
    TektronixChannel,
    TektronixCompiler,
    TektronixProgram,
)
from qwip.processing.processors import IQTraceResult
from qwip.sequencer.compilation import ChannelInfo, DeviceInfo, TriggerInfo
from qwip.sequencer.elements import SequenceElement
from qwip.sequencer.instructions import QWiPCompiler
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


@pytest.fixture
def awg(instrument_server):
    return instrument_server["tektronix"]


@pytest.fixture
def adc(instrument_server):
    return instrument_server["alazar"]


@pytest.fixture
def backend(awg, adc):
    return QWiPBackend(dac=TektronixBackend(device=awg), adc=AlazarBackend(device=adc))


@pytest.fixture
def compiler():
    dac = DeviceInfo.from_channels(
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

    adc = DeviceInfo.from_channels(
        channels=(ChannelInfo("RO", 0, read=True),),
        sample_rate=1.0e9,
        trigger=TriggerInfo(device="tektronix", index=3, subchannel=0),
        name="alazar",
    )

    demod = DeviceInfo.from_channels(
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

    return QWiPCompiler.from_devices(
        [dac, adc, demod],
        modulations=modulations,
        subcompilers=dict(tektronix=TektronixCompiler(), alazar=AlazarCompiler()),
    )


@pytest.fixture
def pulses():
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


@pytest.fixture
def freq_sweep(pulses):
    se = SequenceElement()
    se.add_waveform(pulses["Q0_X90"])
    se.add_waveform(pulses["Q0_X90"], pulses["Q0_X90"].width)
    se.add_waveform(pulses["R0"], 2 * pulses["Q0_X90"].width + 100e-9)

    ro = SequenceElement()
    ro.add_waveform(pulses["read"])
    ro.add_waveform(pulses["D0"])
    ro.add_waveform(pulses["D1"])

    readout = TriggeredWaveform(target=ro, width=50e-9, channels=("RO_marker",))

    se.add_waveform(readout, 2 * pulses["Q0_X90"].width)

    seq = Sequence.sweep(se, mod_R0=np.arange(100, 501, 50) * 1e6)
    return seq


@pytest.fixture
def t1_sweep(pulses):
    se = SequenceElement()
    se.add_waveform(pulses["Q0_X90"])
    se.add_waveform(pulses["Q0_X90"], pulses["Q0_X90"].width)

    ro = SequenceElement()
    ro.add_waveform(pulses["read"])
    ro.add_waveform(pulses["D0"])
    ro.add_waveform(pulses["D1"])

    readout = TriggeredWaveform(target=ro, width=50e-9, channels=("RO_marker",))
    se.add_waveform(readout, 2 * pulses["Q0_X90"].width + "delay")
    se.add_waveform(pulses["R0"], 2 * pulses["Q0_X90"].width + 100e-9 + "delay")

    seq = Sequence.sweep(se, delay=np.linspace(0, 200e-6, 21))
    return seq


@pytest.mark.skip_instrument("alazar", "tektronix")
class TestCompilation:
    @pytest.mark.parametrize("seq", ["freq_sweep", "t1_sweep"])
    def test_compile(self, compiler, seq, request):
        seq = request.getfixturevalue(seq)
        exe = compiler.compile(seq)

        assert (
            len(exe.programs["tektronix"].waveforms[0])
            == len(exe.programs["tektronix"].marker1s[0])
            == len(exe.programs["tektronix"].marker2s[0])
            == len(exe.programs["tektronix"].repeats)
            == len(exe.programs["tektronix"].waits)
            == len(exe.programs["tektronix"].go_tos)
            == len(exe.programs["tektronix"].jump_tos)
            == len(seq.flat)
        )

        assert (
            len(exe.programs["tektronix"].channels)
            == len(exe.programs["tektronix"].waveforms)
            == len(exe.programs["tektronix"].marker1s)
            == len(exe.programs["tektronix"].marker2s)
        )

    @pytest.mark.parametrize("seq", ["freq_sweep", "t1_sweep"])
    def test_upload(self, compiler, seq, request, backend):
        seq = request.getfixturevalue(seq)
        exe = compiler.compile(seq)
        backend.upload(exe)


@pytest.mark.skip_instrument("tektronix", "alazar")
class TestBackend:
    def test_acquire(self, backend, compiler, freq_sweep):
        backend.dac.device.ch4_m1_low(-0.3)
        exe = compiler.compile(freq_sweep)
        backend.upload(exe)
        result = backend.acquire(repetitions=256)

        assert isinstance(result["alazar"], IQTraceResult)
        assert result["alazar"].index.levshape == (256, 9, 1, 1024)
