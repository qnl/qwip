import numpy as np
import pytest

from qwip.backends.backend import QWiPBackend
from qwip.backends.tektronix import (
    TektronixBackend,
    TektronixChannel,
    TektronixCompiler,
    TektronixProgram,
)
from qwip.processing.processors import IQTraceResult
from qwip.sequencer.compilation import (
    ChannelInfo,
    DeviceInfo,
    QWiPCompiler,
    TriggerInfo,
)
from qwip.sequencer.phase_tracker import Frame
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.timeline import Timeline
from qwip.sequencer.waveform import (
    CWWaveform,
    GaussianWaveform,
    ModulatedWaveform,
    SquareWaveform,
    TriggeredWaveform,
    VirtualZWaveform,
)

try:
    from qwip.backends.alazar import AlazarBackend, AlazarCompiler
except ImportError:
    pytest.skip("Alazar dependencies not installed.", allow_module_level=True)


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
        channels=(ChannelInfo("RO", 0, read=True, output=False),),
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

    frames = dict(
        mod_Q0=Frame(200e6),
        mod_Q1=Frame(150e6),
        mod_R0=Frame(-300e6),
        mod_R1=Frame(-400e6),
    )

    return QWiPCompiler.from_devices(
        [dac, adc, demod],
        frames=frames,
        subcompilers=dict(tektronix=TektronixCompiler(), alazar=AlazarCompiler()),
    )


@pytest.fixture
def pulses():
    Q0_X90 = ModulatedWaveform(
        name="Q0_X90",
        envelope=GaussianWaveform(width=25e-9, amplitude=0.1),
        modulation=CWWaveform(frequency="mod_Q0", channel="Q0_IQ"),
    )

    Q1_X90 = ModulatedWaveform(
        name="Q1_X90",
        envelope=GaussianWaveform(width=25e-9, amplitude=0.1),
        modulation=CWWaveform(frequency="mod_Q1", channel="Q1_IQ"),
    )

    Q0_Z90 = VirtualZWaveform(name="Q0_Z", frame="mod_Q0", phase=90)
    Q1_Z90 = VirtualZWaveform(name="Q0_Z", frame="mod_Q1", phase=90)

    R0 = ModulatedWaveform(
        name="R0",
        envelope=SquareWaveform(width=1e-6, amplitude=0.2),
        modulation=CWWaveform(channel="RO_IQ", frequency="mod_R0"),
    )

    R1 = ModulatedWaveform(
        name="R1",
        envelope=SquareWaveform(width=1e-6, amplitude=0.25),
        modulation=CWWaveform(channel="RO_IQ", frequency="mod_R1"),
    )

    D0 = ModulatedWaveform(
        name="D0",
        envelope=SquareWaveform(width=1e-6, amplitude=1),
        modulation=CWWaveform(channel="R0", frequency="mod_R0"),
    )

    D1 = ModulatedWaveform(
        name="D0",
        envelope=SquareWaveform(width=1e-6, amplitude=1),
        modulation=CWWaveform(channel="R1", frequency="mod_R1"),
    )

    return dict(
        Q0_X90=Q0_X90,
        Q1_X90=Q1_X90,
        Q0_Z90=Q0_Z90,
        Q1_Z90=Q1_Z90,
        R0=R0,
        R1=R1,
        read=SquareWaveform(width=1e-6, channel="RO"),
        D0=D0,
        D1=D1,
    )


@pytest.fixture
def freq_sweep(pulses):
    tmln = Timeline()
    tmln.add(pulses["Q0_X90"])
    tmln.add(pulses["Q0_X90"], pulses["Q0_X90"].width)
    tmln.add(pulses["R0"], 2 * pulses["Q0_X90"].width + 100e-9)

    ro = Timeline()
    ro.add(pulses["read"])
    ro.add(pulses["D0"])
    ro.add(pulses["D1"])

    readout = TriggeredWaveform(target=ro, width=50e-9, channel="RO_marker")

    tmln.add(readout, 2 * pulses["Q0_X90"].width)

    seq = Sequence.sweep(se, mod_R0=np.arange(100, 501, 50) * 1e6)
    return seq


@pytest.fixture
def t1_sweep(pulses):
    tmln = Timeline()
    tmln.add(pulses["Q0_X90"])
    tmln.add(pulses["Q0_X90"], pulses["Q0_X90"].width)

    ro = Timeline()
    ro.add(pulses["read"])
    ro.add(pulses["D0"])
    ro.add(pulses["D1"])

    readout = TriggeredWaveform(target=ro, width=50e-9, channel="RO_marker")
    tmln.add(readout, 2 * pulses["Q0_X90"].width + "delay")
    tmln.add(pulses["R0"], 2 * pulses["Q0_X90"].width + 100e-9 + "delay")

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

        assert exe.programs["alazar"].samples == [1001] * len(seq.flat)
        assert exe.num_reads == [1] * len(seq.flat)

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
