import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from qwip.sequencer.compilation import (
    ChannelGroup,
    ChannelInfo,
    QuantumExecutable,
    QWiPExecutable,
    QWiPSequencer,
    WaveformBlock,
    WaveformSequencer,
)
from qwip.sequencer.elements import SequenceElement
from qwip.sequencer.phase_tracker import ModulationFrequency
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.waveform import (
    CosineRampWaveform,
    CWWaveform,
    GaussianWaveform,
    ModulatedWaveform,
    ReadoutMarker,
    SquareWaveform,
    TriggeredWaveform,
    VirtualZWaveform,
)


class TestQuantumExecutable:
    def test_equality(self):
        qxe1 = QuantumExecutable(sequence=Sequence([]))
        qxe2 = QuantumExecutable(sequence=Sequence([]))
        qxe3 = QuantumExecutable(sequence=qxe1.seq)

        assert qxe1 != qxe2
        assert qxe1 == qxe3
        assert QuantumExecutable(sequence=None) == QuantumExecutable(sequence=None)


class TestChannelGroup:
    def test_num_channels(self):
        dac = ChannelGroup.from_channels(
            channels=(ChannelInfo("Q0_I", 0), ChannelInfo("Q0_Q", 1)),
            sample_rate=2.4e9,
            name="dac",
        )

        assert dac.name == "dac"
        assert dac.num_channels == 2

    def test_channel_lookup(self):
        dac = ChannelGroup.from_channels(
            channels=(ChannelInfo("Q0_I", 0), ChannelInfo("Q0_Q", 1)),
            sample_rate=2.4e9,
            name="dac",
        )

        assert dac["Q0_I"] is dac.channels[0]
        assert dac["Q0_Q"] == dac.channels[1]

        with pytest.raises(KeyError):
            dac["Q1_I"]

    def test_properties(self):
        dac = ChannelGroup.from_channels(
            channels=(ChannelInfo("Q0_I", 0), ChannelInfo("Q0_Q", 1, subchannel=2)),
            sample_rate=2.4e9,
            name="dac",
        )

        assert dac.num_channels == 2
        assert dac.max_channel_index == 1
        assert dac.num_subchannels == 2
        assert dac.max_subchannel_index == 2


class TestWaveformBlock:
    def test_empty(self):
        wblk = WaveformBlock.empty(
            1024, sample_rate=1e9, channels=[ChannelInfo("Q0", 0), ChannelInfo("Q1", 1)]
        )

        assert list(wblk.waveforms.keys()) == [(0, 0), (1, 0)]

        for arr in wblk.waveforms.values():
            assert_array_equal(arr, np.zeros(1024, dtype=np.float32))

    @pytest.mark.parametrize(
        "channel,dtype",
        [
            (ChannelInfo("CH", index=0, subchannel=0), np.float32),
            (ChannelInfo("MRK", index=0, subchannel=1), np.int8),
        ],
    )
    def test_add_waveform(self, channel, dtype):
        wblk = WaveformBlock(samples=2000, sample_rate=2.4e9)
        assert list(wblk.waveforms.keys()) == []

        wblk.add_waveform(channel, dtype)
        assert_array_equal(
            wblk.waveforms[channel.index, channel.subchannel], np.zeros(2000, dtype)
        )

        old = wblk.waveforms[channel.index, channel.subchannel]
        wblk.add_waveform(channel, dtype)
        assert old is not wblk.waveforms[channel.index, channel.subchannel]

    def test_getitem(self):
        wblk = WaveformBlock.empty(
            512,
            sample_rate=2.4e9,
            channels=[ChannelInfo("Q0", 0), ChannelInfo("Q1", 2)],
        )

        assert wblk[ChannelInfo("Q0", 0)] is wblk[0, 0]
        assert wblk[0] is wblk[0, 0]

    def test_get_waveform(self):
        wblk = WaveformBlock(samples=256, sample_rate=1.8e9)

        assert_array_equal(
            wblk.get_waveform(ChannelInfo("CH", index=2)), np.zeros(256, np.float32)
        )

        assert wblk.get_waveform(ChannelInfo("CH", index=2)) is wblk.waveforms[2, 0]


class TestQWiPSequencer:
    @pytest.fixture
    def sequencer(self):
        dac = ChannelGroup.from_channels(
            channels=(
                ChannelInfo("Q0_I", 0),
                ChannelInfo("Q0_Q", 1),
                ChannelInfo("Q1_I", 2),
                ChannelInfo("Q1_Q", 3),
                ChannelInfo("RO_marker", 0, subchannel=1),
            ),
            sample_rate=2.4e9,
            name="dac",
        )

        ro_out = ChannelGroup.from_channels(
            channels=(
                ChannelInfo("RO_I", 0),
                ChannelInfo("RO_Q", 1),
                ChannelInfo("demod_marker", 3),
            ),
            sample_rate=1.8e9,
            triggered=True,
            name="ro_out",
        )

        ro_in = ChannelGroup.from_channels(
            channels=(
                ChannelInfo("D0", 0, read=True),
                ChannelInfo("D1", 0, read=True),
            ),
            sample_rate=1.8e9,
            triggered=True,
            name="ro_in",
        )

        modulations = dict(
            mod_Q0=ModulationFrequency(200e6),
            mod_Q1=ModulationFrequency(150e6),
            mod_R0=ModulationFrequency(-300e6),
            mod_R1=ModulationFrequency(-400e6),
        )

        return QWiPSequencer.from_channel_groups(
            [dac, ro_in, ro_out], modulations=modulations
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
            modulation=CWWaveform(channels=("D0",), frequency="mod_R0"),
        )

        D1 = ModulatedWaveform(
            name="D0",
            envelope=SquareWaveform(width=1e-6, amplitude=1),
            modulation=CWWaveform(channels=("D1",), frequency="mod_R1"),
        )

        return dict(
            Q0_X90=Q0_X90,
            Q1_X90=Q1_X90,
            Q0_Z90=Q0_Z90,
            Q1_Z90=Q1_Z90,
            R0=R0,
            R1=R1,
            D0=D0,
            D1=D1,
        )

    @pytest.mark.parametrize(
        "name,expect",
        [
            ("Q0_I", ChannelInfo(name="Q0_I", index=0, group="dac")),
            ("RO_Q", ChannelInfo(name="RO_Q", index=1, group="ro_out")),
            ("random", None),
        ],
    )
    def test_get_channel_info(self, sequencer, name, expect):
        assert sequencer.get_channel_info(name) == expect

    def test_compile_sequence_element(self, sequencer, pulses):
        se = SequenceElement()
        se.add_waveform(pulses["Q0_X90"])
        se.add_waveform(pulses["Q0_X90"], pulses["Q0_X90"].width)
        se.add_waveform(pulses["R0"], 2 * pulses["Q0_X90"].width)
        se.add_waveform(pulses["D0"], 2 * pulses["Q0_X90"].width)

        exe = QWiPExecutable(devices={n: [] for n in sequencer.channels})

        wblk = sequencer.compile_sequence_element(se, exe)

        assert list(wblk["dac"].waveforms.keys()) == [(0, 0), (1, 0)]
        # assert list(wblk["ro_out"].waveforms.keys()) == [(0, 0), (1, 0)]

    def test_compile(self, sequencer, pulses):
        seq = []

        for i in range(2):
            se = SequenceElement()
            se.add_waveform(pulses[f"Q{i}_X90"])
            se.add_waveform(pulses[f"Q{i}_X90"], pulses[f"Q{i}_X90"].width)
            se.add_waveform(pulses[f"R{i}"], 2 * pulses[f"Q{i}_X90"].width)
            se.add_waveform(pulses[f"D{i}"], 2 * pulses[f"Q{i}_X90"].width)
            seq.append(se)

        seq = Sequence(seq)
        exe = sequencer.compile(seq)

        assert list(exe.devices["dac"][0].waveforms.keys()) == [(0, 0), (1, 0)]
        assert list(exe.devices["dac"][1].waveforms.keys()) == [(2, 0), (3, 0)]
        # assert list(exe.devices["ro_out"][0].waveforms.keys()) == [(0, 0), (1, 0)]
        # assert list(exe.devices["ro_in"][0].waveforms.keys()) == [(0, 0)]

    def test_compile_triggered(self, sequencer, pulses):
        seq = []

        se = SequenceElement()
        ro = SequenceElement()
        demod = SequenceElement()
        for i in range(2):
            demod.add_waveform(pulses[f"D{i}"])
            ro.add_waveform(pulses[f"R{i}"])

        demod = TriggeredWaveform(width=2e-9, target=demod, channels=("demod_marker",))
        ro.add_waveform(demod, 100e-9)
        ro = TriggeredWaveform(width=2e-9, target=ro, channels=("RO_marker",))

        for i in range(2):
            se.add_waveform(pulses[f"Q{i}_X90"])
            se.add_waveform(pulses[f"Q{i}_X90"], pulses[f"Q{i}_X90"].width)
            se.add_waveform(ro, 2 * pulses[f"Q{i}_X90"].width)
            seq.append(se)

        seq = Sequence(seq)
        exe = sequencer.compile(seq)

        import matplotlib.pyplot as plt

        exe.plot(1)
        plt.show()

class TestWaveformSequencer:
    @pytest.fixture
    def sequencer(self):
        dac = ChannelGroup.from_channels(
            channels=(
                ChannelInfo("Q0_I", 0),
                ChannelInfo("Q0_Q", 1),
                ChannelInfo("Q1_I", 2),
                ChannelInfo("Q1_Q", 3),
            ),
            sample_rate=2.4e9,
            name="seq",
        )

        adc = ChannelGroup.from_channels(
            channels=(ChannelInfo("RO_I", 0), ChannelInfo("RO_Q", 1)),
            sample_rate=1.8e9,
            name="readout",
        )

        modulations = dict(
            mod_Q0=ModulationFrequency(200e6),
            mod_Q1=ModulationFrequency(150e6),
            mod_R0=ModulationFrequency(-300e6),
            mod_R1=ModulationFrequency(-400e6),
        )

        return WaveformSequencer.from_channel_groups(
            [dac, adc], modulations=modulations
        )

    @pytest.fixture
    def pulses(self):
        Q0_X90 = ModulatedWaveform(
            name="Q0_X90",
            envelope=CosineRampWaveform(width=25e-9, amplitude=0.1),
            modulation=CWWaveform(frequency="mod_Q0", channels=("Q0_I", "Q0_Q")),
        )

        Q1_X90 = ModulatedWaveform(
            name="Q1_X90",
            envelope=CosineRampWaveform(width=25e-9, amplitude=0.1),
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

        return dict(
            Q0_X90=Q0_X90, Q1_X90=Q1_X90, Q0_Z90=Q0_Z90, Q1_Z90=Q1_Z90, R0=R0, R1=R1
        )

    @pytest.mark.parametrize(
        "name,expect",
        [
            ("Q0_I", ChannelInfo(name="Q0_I", index=0, group="seq")),
            ("RO_Q", ChannelInfo(name="RO_Q", index=1, group="readout")),
            ("random", None),
        ],
    )
    def test_get_channel_info(self, sequencer, name, expect):
        assert sequencer.get_channel_info(name) == expect

    def test_end_to_end(self, sequencer, pulses, data_file):
        import matplotlib.pyplot as plt
        import numpy as np

        ro_se = SequenceElement()
        ro_se.add_waveform([pulses[f"R{r}"] for r in range(2)])

        se_0 = SequenceElement()
        se_0.add_waveform([pulses["Q0_X90"], pulses["Q1_X90"]])
        se_0.add_waveform(ReadoutMarker(), 50e-9)

        se_1 = SequenceElement()
        se_1.add_waveform([pulses["Q0_Z90"], pulses["Q1_Z90"]])
        se_1.add_waveform([pulses["Q0_X90"], pulses["Q1_X90"]])
        se_1.add_waveform(ReadoutMarker(), 50e-9)

        seq = Sequence([se_0, se_1])

        cseq = sequencer.compile(seq, readout=ro_se)

        expected = np.loadtxt(data_file).reshape(cseq.array.shape)

        assert_array_almost_equal(expected, cseq.array)
