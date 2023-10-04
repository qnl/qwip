import itertools as it

import numpy as np
import pytest

from qwip.backends.qtrl import (
    QTRLBackend,
    QTRLCompiler,
    QTRLExecutable,
    format_legacy_IQ,
)
from qwip.sequencer.compilation import ChannelInfo, DeviceInfo, TriggerInfo
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
            ChannelInfo("RO_Q", 0),
        ),
        sample_rate=1.8e9,
        trigger=TriggerInfo(device="seq", index=0, subchannel=1),
        name="readout",
    )

    readout_in = DeviceInfo.from_channels(
        channels=tuple(
            ChannelInfo(f"R{r}", r, read=True, output=False) for r in range(4)
        ),
        sample_rate=1.8e9,
        trigger=TriggerInfo(device="seq", index=0, subchannel=1),
        name="readout_in",
    )

    modulations = dict(
        mod_Q0=ModulationFrequency(400e6),
        mod_Q1=ModulationFrequency(500e6),
        mod_Q2=ModulationFrequency(-30e6),
        mod_Q3=ModulationFrequency(-450e6),
        mod_R0=ModulationFrequency(100e6),
        mod_R1=ModulationFrequency(200e6),
        mod_R2=ModulationFrequency(-50e6),
        mod_R3=ModulationFrequency(-150e6),
    )

    return QTRLCompiler.from_devices(
        [seq, readout, readout_in],
        modulations=modulations,
    )


@pytest.fixture
def pulses():
    X = {
        f"Q{q}_X90": ModulatedWaveform(
            name=f"Q{q}_X90",
            envelope=GaussianWaveform(width=25e-9, amplitude=0.1),
            modulation=CWWaveform(
                frequency=f"mod_Q{q}", channels=(f"Q{q}_I", f"Q{q}_Q")
            ),
        )
        for q in range(4)
    }

    Z = {
        f"Q{q}_Z90": VirtualZWaveform(name=f"Q{q}_Z", mod_key=f"mod_Q{q}", phase=90)
        for q in range(4)
    }

    readout = {
        f"R{r}": ModulatedWaveform(
            name=f"R{r}",
            envelope=SquareWaveform(width=1e-6, amplitude=0.2),
            modulation=CWWaveform(channels=("RO_I", "RO_Q"), frequency=f"mod_R{r}"),
        )
        for r in range(4)
    }

    return X | Z | readout


@pytest.fixture
def freq_sweep(pulses):
    se = SequenceElement()
    se.add_waveform(pulses["Q0_X90"])
    se.add_waveform(pulses["Q0_X90"], pulses["Q0_X90"].width)

    ro = SequenceElement()
    ro.add_waveform(pulses["R0"])
    ro.add_waveform(SquareWaveform(width=pulses["R0"].width, channels=("R0",)))
    readout = TriggeredWaveform(target=ro, width=50e-9, channels=("RO_marker",))

    se.add_waveform(readout, 2 * pulses["Q0_X90"].width)

    seq = Sequence.sweep(se, mod_R0=np.arange(100, 501, 50) * 1e6)
    return seq


@pytest.fixture
def t1_sweep(pulses):
    se = SequenceElement()
    se.add_waveform(pulses["Q0_X90"])
    se.add_waveform(pulses["Q0_X90"], pulses["Q0_X90"].width)
    se.add_waveform(pulses["Q1_X90"])
    se.add_waveform(pulses["Q1_X90"], pulses["Q1_X90"].width)

    ro = SequenceElement()
    ro.add_waveform(pulses["R0"])
    ro.add_waveform(SquareWaveform(width=pulses["R0"].width, channels=("R0",)))
    readout = TriggeredWaveform(target=ro, width=50e-9, channels=("RO_marker",))

    se.add_waveform(readout, 2 * pulses["Q0_X90"].width + "delay")

    seq = Sequence.sweep(se, delay=np.linspace(0, 200e-6, 21))
    return seq


class TestQTRLCompiler:
    @pytest.mark.parametrize("seq", ["freq_sweep", "t1_sweep"])
    def test_compile(self, compiler, seq, request):
        seq = request.getfixturevalue(seq)
        exe = compiler.compile(seq)

        assert isinstance(exe, QTRLExecutable)
        assert exe.sequence is seq
        assert list(exe.waveforms.keys()) == ["seq", "readout"]
        assert len(exe.get_readout_locations()) == exe.n_elements
        assert exe._readout._readout.n_readouts == len(seq)
