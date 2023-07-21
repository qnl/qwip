import attrs
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

try:
    from distproc.compiler import CompiledProgram
except ImportError:
    pytest.skip("Qubic dependencies not installed.", allow_module_level=True)

import qwip
from qwip.backends.qubic import (
    BarrierInstruction,
    DelayInstruction,
    PulseInstruction,
    QubicExecutable,
    QubicSequencer,
    VirtualZInstruction,
)
from qwip.sequencer import (
    CWWaveform,
    GaussianWaveform,
    ModulatedWaveform,
    ModulationFrequency,
    Sequence,
    SequenceElement,
    SquareWaveform,
    VirtualZWaveform,
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
        adc = [
            ChannelInfo(f"Q{i}.rdlo", index=i, subchannel=2, read=True)
            for i in range(8)
        ]

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

    @pytest.fixture
    def gates(self):
        gates = dict()
        for q in range(4):
            X90 = SequenceElement()
            X90.add_waveform(VirtualZWaveform(mod_key=f"Q{q}.freq_GE"))
            X90.add_waveform(
                ModulatedWaveform(
                    envelope=GaussianWaveform(width=25e-9),
                    modulation=CWWaveform(
                        channels=(f"Q{q}.qdrv",),
                        frequency=f"Q{q}.freq_GE",
                        hardware_modulation=True,
                    ),
                )
            )
            X90.add_waveform(VirtualZWaveform(mod_key=f"Q{q}.freq_GE"))
            X90.width = 25e-9

            ro = SequenceElement()
            ro.add_waveform(
                ModulatedWaveform(
                    envelope=SquareWaveform(width=2e-6),
                    modulation=CWWaveform(
                        channels=(f"Q{q}.rdrv",),
                        frequency=f"Q{q}.readfreq",
                        hardware_modulation=True,
                    ),
                )
            )
            ro.add_waveform(
                ModulatedWaveform(
                    envelope=SquareWaveform(width=2e-6),
                    modulation=CWWaveform(
                        channels=(f"Q{q}.rdlo",),
                        frequency=f"Q{q}.readfreq",
                        hardware_modulation=True,
                    ),
                ),
                500e-9,
            )
            ro.width = 2.5e-6

            gates[f"Q{q}_X90"] = X90
            gates[f"Q{q}_RO"] = ro

        return gates

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

    def test_compile_sequence_element(self, sequencer, gates):
        se = SequenceElement()
        se.append(gates["Q0_X90"])
        se.append(gates["Q0_X90"], gates["Q0_X90"].width)
        se.append(gates["Q0_RO"], 2 * gates["Q0_X90"].width)

        instructions, reads = sequencer.compile_sequence_element(se.resolve_locations())

        names = [ins.name for ins in instructions]
        pulses = [
            (ins.dest, ins.env.shape)
            for ins in instructions
            if isinstance(ins, PulseInstruction)
        ]

        expected = [
            VirtualZInstruction(qubit=("Q0",), freqname="freq_GE"),
            PulseInstruction(
                env=np.zeros(200, dtype=np.complex64),
                dest="Q0.qdrv",
                freq="Q0.freq_GE",
                phase=0,
                amp=1,
                twidth=25e-9,
            ),
            VirtualZInstruction(qubit=("Q0",), freqname="freq_GE"),
            VirtualZInstruction(qubit=("Q0",), freqname="freq_GE"),
            PulseInstruction(
                env=np.zeros(200, dtype=np.complex64),
                dest="Q0.qdrv",
                freq="Q0.freq_GE",
                phase=0,
                amp=1,
                twidth=25e-9,
            ),
            VirtualZInstruction(qubit=("Q0",), freqname="freq_GE"),
            DelayInstruction(qubits=("Q0.rdrv",), t=50e-9),
            PulseInstruction(
                env=np.zeros(1000, dtype=np.complex64),
                dest="Q0.rdrv",
                freq="Q0.readfreq",
                phase=0,
                amp=1,
                twidth=2e-6,
            ),
            DelayInstruction(qubits=("Q0.rdlo",), t=500e-9),
            PulseInstruction(
                env=np.zeros(1000, dtype=np.complex64),
                dest="Q0.rdlo",
                freq="Q0.readfreq",
                phase=0,
                amp=1,
                twidth=2e-6,
            ),
        ]

        assert len(instructions) == len(expected)

        for ins, exp in zip(instructions, expected):
            match ins:
                case PulseInstruction(env=env):
                    assert exp.env.shape == env.shape
                    assert exp.env.dtype == env.dtype
                    exp = attrs.evolve(exp, env=ins.env)

                case DelayInstruction(t=t):
                    # delay times might be slightly off due to floating point error
                    assert_almost_equal(t, exp.t)
                    ins = attrs.evolve(ins, t=exp.t)

            assert ins == exp
