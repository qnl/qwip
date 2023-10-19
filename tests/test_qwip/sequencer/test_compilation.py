import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import qwip
from qwip.sequencer.compilation import (
    ChannelInfo,
    DeviceInfo,
    QuantumExecutable,
    QWiPCompiler,
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

    @pytest.mark.parametrize(
        "exe,expected",
        [
            (
                QuantumExecutable(),
                dict(__class__="qwip.sequencer.compilation.QuantumExecutable"),
            ),
            (
                QuantumExecutable(sequence=Sequence.empty((10, 2))),
                dict(
                    sequence=dict(
                        names=[None, None],
                        labels={},
                        shape=(10, 2),
                        data=[[{} for _ in range(2)] for _ in range(10)],
                    ),
                    __class__="qwip.sequencer.compilation.QuantumExecutable",
                ),
            ),
        ],
    )
    def test_unstructure(self, exe, expected):
        unstructured = qwip.converter.unstructure(exe)
        assert unstructured == expected

    @pytest.mark.parametrize(
        "exe",
        [QuantumExecutable(), QuantumExecutable(sequence=Sequence.empty((10, 2)))],
    )
    def test_structure(self, exe):
        unstructured = qwip.converter.unstructure(exe)
        structured = qwip.converter.structure(unstructured, QuantumExecutable)

        assert (
            exe.sequence is structured.sequence
            or (exe.sequence == structured.sequence).all()
        )


class TestDeviceInfo:
    def test_num_channels(self):
        dac = DeviceInfo.from_channels(
            channels=(ChannelInfo("Q0_I", 0), ChannelInfo("Q0_Q", 1)),
            sample_rate=2.4e9,
            name="dac",
        )

        assert dac.name == "dac"
        assert dac.num_channels == 2

    def test_channel_lookup(self):
        dac = DeviceInfo.from_channels(
            channels=(ChannelInfo("Q0_I", 0), ChannelInfo("Q0_Q", 1)),
            sample_rate=2.4e9,
            name="dac",
        )

        assert dac["Q0_I"] is dac.channels[0]
        assert dac["Q0_Q"] == dac.channels[1]

        with pytest.raises(KeyError):
            dac["Q1_I"]

    def test_properties(self):
        dac = DeviceInfo.from_channels(
            channels=(ChannelInfo("Q0_I", 0), ChannelInfo("Q0_Q", 1, subchannel=2)),
            sample_rate=2.4e9,
            name="dac",
        )

        assert dac.num_channels == 2
        assert dac.max_channel_index == 1
        assert dac.num_subchannels == 2
        assert dac.max_subchannel_index == 2


class TestQWiPCompiler:
    @pytest.fixture
    def compiler(self):
        dac = DeviceInfo.from_channels(
            channels=(
                ChannelInfo("Q0_I", 0),
                ChannelInfo("Q0_Q", 1),
                ChannelInfo("Q1_I", 2),
                ChannelInfo("Q1_Q", 3),
                ChannelInfo("RO_marker", 0, subchannel=1),
            ),
            sample_rate=2.4e9,
            name="seq",
        )

        adc = DeviceInfo.from_channels(
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

        return QWiPCompiler.from_devices([dac, adc], modulations=modulations)

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
            ("Q0_I", ChannelInfo(name="Q0_I", index=0, device="seq")),
            ("RO_Q", ChannelInfo(name="RO_Q", index=1, device="readout")),
            ("random", None),
        ],
    )
    def test_get_channel_info(self, compiler, name, expect):
        assert compiler.get_channel_info(name) == expect

    def test_end_to_end(self, compiler, pulses, data_file):
        import matplotlib.pyplot as plt
        import numpy as np

        ro_se = SequenceElement()
        ro_se.add_waveform([pulses[f"R{r}"] for r in range(2)])
        readout = TriggeredWaveform(target=ro_se, width=50e-9, channels=("RO_marker",))

        se_0 = SequenceElement()
        se_0.add_waveform([pulses["Q0_X90"], pulses["Q1_X90"]])
        se_0.add_waveform(readout, 50e-9)

        se_1 = SequenceElement()
        se_1.add_waveform([pulses["Q0_Z90"], pulses["Q1_Z90"]])
        se_1.add_waveform([pulses["Q0_X90"], pulses["Q1_X90"]])
        se_1.add_waveform(readout, 50e-9)

        seq = Sequence([se_0, se_1])

        exe = compiler.compile(seq)
