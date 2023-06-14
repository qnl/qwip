import attrs
import numpy as np
import pytest
from distproc.compiler import CompiledProgram
from numpy.testing import assert_array_equal

import qwip
from qwip.backends.qubic import (
    BarrierInstruction,
    DelayInstruction,
    PulseInstruction,
    QubicExecutable,
    QubicSequencer,
)
from qwip.sequencer import (
    CWWaveform,
    GaussianWaveform,
    ModulatedWaveform,
    ModulationFrequency,
    Sequence,
    SequenceElement,
)
from qwip.sequencer.compilation import ChannelGroup, ChannelInfo


class TestQubicInstruction:
    @pytest.mark.parametrize(
        "ins,expect",
        [
            (
                PulseInstruction(
                    env=np.linspace(0, 200, 100),
                    dest="Q0.drv",
                    freq=5e9,
                    twidth=12.5e-9,
                ),
                dict(
                    name="pulse",
                    env=np.linspace(0, 200, 100),
                    dest="Q0.drv",
                    freq=5e9,
                    phase=0,
                    amp=1,
                    twidth=12.5e-9,
                ),
            ),
            (DelayInstruction(t=0.0005), dict(name="delay", t=0.0005)),
        ],
    )
    def test_todict(self, ins, expect):
        d = ins.todict()

        assert set(d.keys()) == set(expect.keys())

        for k in d.keys():
            match k:
                case "env":
                    assert_array_equal(d[k], expect[k])
                case _:
                    assert d[k] == expect[k]


class TestPulseInstruction:
    @pytest.mark.parametrize(
        "p1,p2,expect",
        [
            (
                PulseInstruction(env=np.zeros(10), dest="Q0.drv", freq=0),
                PulseInstruction(env=np.zeros(10), dest="Q0.drv", freq=0),
                False,
            ),
            (
                p := PulseInstruction(env=np.ones(21), dest="Q0.drv", freq=1e9),
                PulseInstruction(env=p.env, dest="Q0.drv", freq=1e9),
                True,
            ),
        ],
    )
    def test_equality_and_hash(self, p1, p2, expect):
        assert (p1 == p2) is expect
        assert (hash(p1) == hash(p2)) is expect


class TestQubicExecutable:
    def test_equality_and_hash(self):
        exe1 = QubicExecutable(
            program=CompiledProgram([]), assembly={}, repetition_delay=1
        )
        exe2 = QubicExecutable(
            program=CompiledProgram([]), assembly={}, repetition_delay=1
        )
        exe3 = QubicExecutable(
            program=exe1.program, assembly=exe1.assembly, repetition_delay=1
        )

        assert exe1 != exe2
        assert exe1 == exe3
        assert hash(exe1) == hash(exe3)

    def test_frozen(self):
        exe1 = QubicExecutable(
            program=CompiledProgram([]), assembly={}, repetition_delay=1
        )

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            exe1.seq = Sequence([])


class TestQubicSequencer:
    @pytest.fixture
    def sequencer(self):
        qubit = [ChannelInfo(f"Q{i}.qdrv", index=i) for i in range(8)]
        readout = [ChannelInfo(f"Q{i}.rdrv", index=i, subchannel=1) for i in range(8)]
        adc = [ChannelInfo(f"Q{i}.rdlo", index=i, subchannel=2) for i in range(8)]

        qubit = ChannelGroup.from_channels(qubit, sample_rate=8e9, name="qubit")
        readout = ChannelGroup.from_channels(readout, sample_rate=0.5e9, name="readout")
        adc = ChannelGroup.from_channels(adc, sample_rate=0.5e9, name="adc")

        modulations = {
            f"Q{i}.freq_GE": ModulationFrequency((5 + 0.1 * i) * 1e9) for i in range(8)
        } | {
            f"Q{i}.readfreq": ModulationFrequency((6.4 + 0.1 * i) * 1e9)
            for i in range(8)
        }

        return QubicSequencer.from_channel_groups(
            [qubit, readout, adc], modulations=modulations
        )

    def test_get_qchip(self, sequencer):
        qchip = sequencer.get_qchip()

        assert qchip.qubit_dict == {
            f"Q{i}": {
                "freq": None,
                "freq_GE": (5 + 0.1 * i) * 1e9,
                "readfreq": (6.4 + 0.1 * i) * 1e9,
            }
            for i in range(8)
        }

    def test_get_channel_config(self, sequencer):
        channel_config = sequencer.get_channel_config()

        expected_elem_params = {
            "qdrv": dict(samples_per_clk=16, interp_ratio=1),
            "rdrv": dict(samples_per_clk=16, interp_ratio=16),
            "rdlo": dict(samples_per_clk=4, interp_ratio=4),
        }

        assert channel_config.pop("fpga_clk_freq") == 5e8

        for k, ch_config in channel_config.items():
            ch_id = ch_config.core_ind
            assert ch_config.group == k[3:]
            assert ch_config.elem_params == expected_elem_params[ch_config.group]
            assert ch_config.env_mem_name == f"{ch_config.group}env{ch_id}"
            assert ch_config.freq_mem_name == f"{ch_config.group}freq{ch_id}"
            assert ch_config.acc_mem_name == f"accbuf{ch_id}"

    def test_compile_instruction(self):
        ...

    def test_compile_sequence_element(self):
        ...
