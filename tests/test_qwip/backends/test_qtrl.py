import itertools as it

import numpy as np
import pytest
import sympy as sym

from qwip.backends.qtrl import (
    QTRLBackend,
    QTRLCompiler,
    QTRLExecutable,
    format_legacy_IQ,
)
from qwip.sequencer.compilation import ChannelInfo, DeviceInfo, TriggerInfo
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


class TestFormatLegacyIQ:
    def test_reorder(self):
        arr = np.arange(2 * 3 * 4 * 5).astype(float).reshape(2, 3, 4, 5)
        iqdata = format_legacy_IQ(arr)

        assert iqdata.shape == (3, 4, 5)
        assert iqdata.dtype == np.complex128

    def test_float32(self):
        arr = np.arange(2 * 3 * 4 * 5).astype(np.float32).reshape(2, 3, 4, 5)
        iqdata = format_legacy_IQ(arr)

        assert iqdata.shape == (3, 4, 5)


@pytest.fixture
def compiler():
    seq = DeviceInfo.from_channels(
        channels=(
            ChannelInfo("Q0_I", 0),
            ChannelInfo("Q0_Q", 1),
            ChannelInfo("Q1_I", 2),
            ChannelInfo("Q1_Q", 3),
            ChannelInfo("Q2_I", 4),
            ChannelInfo("Q2_Q", 5),
            ChannelInfo("Q3_I", 6),
            ChannelInfo("Q3_Q", 7),
            ChannelInfo("RO_marker", 0, subchannel=1),
        ),
        sample_rate=2.4e9,
        name="seq",
    )

    readout = DeviceInfo.from_channels(
        channels=(
            ChannelInfo("RO_I", 0),
            ChannelInfo("RO_Q", 1),
            ChannelInfo("INT_marker", 0, read=True, subchannel=1),
        ),
        sample_rate=1.8e9,
        trigger=TriggerInfo(device="seq", index=0, subchannel=1),
        name="readout",
    )

    demod = DeviceInfo.from_channels(
        channels=tuple(ChannelInfo(f"R{r}", r) for r in range(4)),
        sample_rate=1.8e9,
        dtype=np.complex64,
        trigger=TriggerInfo(device="readout", index=0, subchannel=1),
        name="demod",
    )

    frames = dict(
        mod_Q0=Frame(400e6),
        mod_Q1=Frame(500e6),
        mod_Q2=Frame(-30e6),
        mod_Q3=Frame(-450e6),
        mod_R0=Frame(100e6),
        mod_R1=Frame(200e6),
        mod_R2=Frame(-50e6),
        mod_R3=Frame(-150e6),
    )

    return QTRLCompiler.from_devices(
        [seq, readout, demod],
        frames=frames,
    )


@pytest.fixture
def pulses():
    X = {
        f"Q{q}_X90": ModulatedWaveform(
            name=f"Q{q}_X90",
            envelope=GaussianWaveform(width=25e-9, amplitude=0.1),
            modulation=CWWaveform(frequency=f"mod_Q{q}", channel=f"Q{q}_IQ"),
        )
        for q in range(4)
    }

    Z = {
        f"Q{q}_Z90": VirtualZWaveform(name=f"Q{q}_Z", frame=f"mod_Q{q}", phase=90)
        for q in range(4)
    }

    readout = {
        f"R{r}": ModulatedWaveform(
            name=f"R{r}",
            envelope=SquareWaveform(width=1e-6, amplitude=0.2),
            modulation=CWWaveform(channel="RO_IQ", frequency=f"mod_R{r}"),
        )
        for r in range(4)
    }

    demod = {
        f"D{r}": ModulatedWaveform(
            name=f"D{r}",
            envelope=SquareWaveform(width=1e-6, amplitude=1),
            modulation=CWWaveform(channel=f"R{r}", frequency=f"mod_R{r}"),
        )
        for r in range(4)
    }

    return X | Z | readout | demod


@pytest.fixture
def freq_sweep(pulses):
    tmln = Timeline()
    tmln.add(pulses["Q0_X90"])
    tmln.add(pulses["Q0_X90"], pulses["Q0_X90"].width)

    ro = Timeline()
    ro.add(pulses["R0"])

    dm = Timeline()
    dm.add(pulses["D0"])
    demod = TriggeredWaveform(target=dm, width=50e-9, channel="INT_marker")
    ro.add(demod)
    readout = TriggeredWaveform(target=ro, width=50e-9, channel="RO_marker")

    tmln.add(readout, 2 * pulses["Q0_X90"].width)

    seq = Sequence.sweep(tmln, mod_R0=np.arange(100, 501, 50) * 1e6)
    return seq


@pytest.fixture
def t1_sweep(pulses):
    tmln = Timeline()
    tmln.add(pulses["Q0_X90"])
    tmln.add(pulses["Q0_X90"], pulses["Q0_X90"].width)
    tmln.add(pulses["Q1_X90"])
    tmln.add(pulses["Q1_X90"], pulses["Q1_X90"].width)

    ro = Timeline()
    ro.add(pulses["R0"])

    dm = Timeline()
    dm.add(pulses["D0"])
    demod = TriggeredWaveform(target=dm, width=50e-9, channel="INT_marker")
    ro.add(demod)
    readout = TriggeredWaveform(target=ro, width=50e-9, channel="RO_marker")

    tmln.add(readout, 2 * pulses["Q0_X90"].width + sym.Symbol("delay"))

    seq = Sequence.sweep(tmln, delay=np.linspace(0, 200e-6, 21))
    return seq


class TestQTRLCompiler:
    @pytest.mark.parametrize("seq,readouts", [("freq_sweep", 9), ("t1_sweep", 1)])
    def test_compile(self, compiler, seq, readouts, request):
        seq = request.getfixturevalue(seq)
        exe = compiler.compile(seq)

        assert isinstance(exe, QTRLExecutable)
        assert exe.sequence is seq
        assert list(exe.waveforms.keys()) == ["seq", "readout", "demod"]
        assert len(exe.get_readout_locations()) == exe.n_elements
        assert exe._readout._readout.n_readouts == len(seq)
        assert exe._readout.shape[1] == readouts
