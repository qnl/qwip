from collections import Counter

import attrs
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

try:
    from distproc.compiler import CompiledProgram
    from distproc.ir_instructions import Pulse, VirtualZ
except ImportError:
    pytest.skip("Qubic dependencies not installed.", allow_module_level=True)


import qwip
from qwip.backends.qubic import QubicCompiler, QubicExecutable  # VirtualZInstruction,
from qwip.sequencer import (
    CWWaveform,
    Frame,
    GaussianWaveform,
    Location,
    ModulatedWaveform,
    Sequence,
    SquareWaveform,
    Timeline,
    VirtualZWaveform,
)
from qwip.sequencer.compilation import ChannelInfo, DeviceInfo


def assert_instructions_almost_equal(ins1, ins2):
    assert type(ins1) == type(ins2)
    for f in attrs.fields(type(ins1)):
        if f.name == "env":
            assert ins1.env.dtype == ins2.env.dtype
            assert_almost_equal(ins1.env, ins2.env)
        else:
            assert getattr(ins1, f.name) == getattr(ins2, f.name)


# # class TestQubicInstruction:
# #     @pytest.mark.parametrize(
# #         "ins,expect",
# #         [
# #             (
# #                 PulseInstruction(
# #                     env=np.linspace(0, 200, 100),
# #                     dest="Q0.drv",
# #                     freq=5e9,
# #                     twidth=12.5e-9,
# #                 ),
# #                 dict(
# #                     name="pulse",
# #                     env=np.linspace(0, 200, 100),
# #                     dest="Q0.drv",
# #                     freq=5e9,
# #                     phase=0,
# #                     amp=1,
# #                     twidth=12.5e-9,
# #                 ),
# #             ),
# #             (DelayInstruction(t=0.0005), dict(name="delay", t=0.0005)),
# #         ],
# #     )
# #     def test_todict(self, ins, expect):
# #         d = ins.todict()

# #         assert set(d.keys()) == set(expect.keys())

# #         for k in d.keys():
# #             match k:
# #                 case "env":
# #                     assert_array_equal(d[k], expect[k])
# #                 case _:
# #                     assert d[k] == expect[k]


# class TestPulseInstruction:
#     @pytest.mark.parametrize(
#         "p1,p2,expect",
#         [
#             (
#                 PulseInstruction(env=np.zeros(10), dest="Q0.drv", freq=0),
#                 PulseInstruction(env=np.zeros(10), dest="Q0.drv", freq=0),
#                 False,
#             ),
#             (
#                 p := PulseInstruction(env=np.ones(21), dest="Q0.drv", freq=1e9),
#                 PulseInstruction(env=p.env, dest="Q0.drv", freq=1e9),
#                 True,
#             ),
#         ],
#     )
#     def test_equality_and_hash(self, p1, p2, expect):
#         assert (p1 == p2) is expect
#         assert (hash(p1) == hash(p2)) is expect


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


class TestQubicCompiler:
    @pytest.fixture
    def compiler(self):
        qubit = [ChannelInfo(f"Q{i}.qdrv", index=i) for i in range(8)]
        readout = [ChannelInfo(f"Q{i}.rdrv", index=i, subchannel=1) for i in range(8)]
        adc = [
            ChannelInfo(f"Q{i}.rdlo", index=i, subchannel=2, read=True)
            for i in range(8)
        ]

        qubit = DeviceInfo.from_channels(
            qubit, sample_rate=8e9, name="qubit", dtype=np.complex64
        )
        readout = DeviceInfo.from_channels(
            readout, sample_rate=0.5e9, name="readout", dtype=np.complex64
        )
        adc = DeviceInfo.from_channels(
            adc, sample_rate=0.5e9, name="adc", dtype=np.complex64
        )

        frames = {f"Q{i}.freq_GE": Frame((5 + 0.1 * i) * 1e9) for i in range(8)} | {
            f"Q{i}.readfreq": Frame((6.4 + 0.1 * i) * 1e9) for i in range(8)
        }

        return QubicCompiler.from_devices([qubit, readout, adc], frames=frames)

    @pytest.fixture
    def gates(self):
        gates = dict()
        for q in range(4):
            X90 = Timeline()
            X90.add(VirtualZWaveform(frame=f"Q{q}.freq_GE"))
            X90.add(
                ModulatedWaveform(
                    envelope=GaussianWaveform(width=30e-9),
                    modulation=CWWaveform(
                        channels=(f"Q{q}.qdrv",),
                        frequency=f"Q{q}.freq_GE",
                        hardware_modulation=True,
                    ),
                )
            )
            X90.add(VirtualZWaveform(frame=f"Q{q}.freq_GE"), 30e-9)
            X90.width = 30e-9

            ro = Timeline()
            ro.add(
                ModulatedWaveform(
                    envelope=SquareWaveform(width=2e-6),
                    modulation=CWWaveform(
                        channels=(f"Q{q}.rdrv",),
                        frequency=f"Q{q}.readfreq",
                        hardware_modulation=True,
                    ),
                )
            )
            ro.add(
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

    def test_get_qchip(self, compiler):
        qchip = compiler.get_qchip()

        assert qchip.qubit_dict == {
            f"Q{i}": {
                "freq": None,
                "freq_GE": (5 + 0.1 * i) * 1e9,
                "readfreq": (6.4 + 0.1 * i) * 1e9,
            }
            for i in range(8)
        }

    def test_get_channel_config(self, compiler):
        channel_config = compiler.get_channel_config()

        expected_elem_params = {
            "qdrv": dict(samples_per_clk=16, interp_ratio=1),
            "rdrv": dict(samples_per_clk=16, interp_ratio=16),
            "rdlo": dict(samples_per_clk=4, interp_ratio=4),
        }

        assert np.round(channel_config.pop("fpga_clk_freq")) == 5e8

        for k, ch_config in channel_config.items():
            ch_id = ch_config.core_ind
            assert ch_config.device == k[3:]
            assert ch_config.elem_params == expected_elem_params[ch_config.device]
            assert ch_config.env_mem_name == f"{ch_config.device}env{ch_id}"
            assert ch_config.freq_mem_name == f"{ch_config.device}freq{ch_id}"
            assert ch_config.acc_mem_name == f"accbuf{ch_id}"

    @pytest.mark.parametrize(
        "location,wave,t0,expected",
        [
            (
                Location(),
                VirtualZWaveform(phase=90, frame="Q0.freq_GE"),
                0,
                [VirtualZ(qubit="Q0", phase=np.pi / 2, freq="freq_GE")],
            ),
            (
                Location(50e-9),
                SquareWaveform(width=50e-9, amplitude=0.5, channels=("Q0.qdrv",)),
                500,
                [
                    Pulse(
                        freq=0,
                        phase=0,
                        amp=1,
                        twidth=50e-9,
                        env=np.array([0] + [0.5] * 399).astype(np.complex64),
                        dest="Q0.qdrv",
                        start_time=525,
                    )
                ],
            ),
            (
                Location(),
                ModulatedWaveform(
                    envelope=SquareWaveform(width=2e-6),
                    modulation=CWWaveform(
                        amplitude=0.5,
                        phase=180,
                        channels=("Q1.rdlo",),
                        frequency="Q1.readfreq",
                        hardware_modulation=True,
                    ),
                ),
                100,
                [
                    Pulse(
                        freq="Q1.readfreq",
                        phase=0,
                        amp=0.5,
                        twidth=2e-6,
                        env=np.array([0] + [np.exp(1j * np.pi)] * 999).astype(
                            np.complex64
                        ),
                        dest="Q1.rdlo",
                        start_time=100,
                    )
                ],
            ),
        ],
    )
    def test_compile_instruction(self, compiler, location, wave, t0, expected):
        reads = Counter()
        instructions = compiler.compile_instruction(
            location,
            wave,
            waveform_cache={},
            reads=reads,
            pulse_kwargs={},
            t0=t0,
        )

        for c in wave.channels:
            if "rdlo" in c:
                assert reads[c] == 1

        assert len(instructions) == len(expected)

        for i1, i2 in zip(instructions, expected):
            assert_instructions_almost_equal(i1, i2)

    def test_compile_timeline(self, compiler, gates):
        tmln = Timeline()
        tmln.add(gates["Q0_X90"])
        tmln.add(gates["Q0_X90"], gates["Q0_X90"].width)
        tmln.add(gates["Q0_RO"], 2 * gates["Q0_X90"].width)

        waveform_cache = {}
        instructions, reads = compiler.compile_timeline(
            tmln.resolve_locations(), waveform_cache=waveform_cache
        )

        expected = [
            VirtualZ(qubit="Q0", phase=0, freq="freq_GE"),
            Pulse(
                freq="Q0.freq_GE",
                phase=0,
                amp=1,
                twidth=30e-9,
                env=waveform_cache[GaussianWaveform(width=30e-9), "qubit"],
                dest="Q0.qdrv",
                start_time=0,
            ),
            VirtualZ(qubit="Q0", phase=0, freq="freq_GE"),
            VirtualZ(qubit="Q0", phase=0, freq="freq_GE"),
            Pulse(
                freq="Q0.freq_GE",
                phase=0,
                amp=1,
                twidth=30e-9,
                env=waveform_cache[GaussianWaveform(width=30e-9), "qubit"],
                dest="Q0.qdrv",
                start_time=15,
            ),
            VirtualZ(qubit="Q0", phase=0, freq="freq_GE"),
            Pulse(
                freq="Q0.readfreq",
                phase=0,
                amp=1,
                twidth=2e-6,
                env=waveform_cache[SquareWaveform(width=2e-6), "readout"],
                dest="Q0.rdrv",
                start_time=30,
            ),
            Pulse(
                freq="Q0.readfreq",
                phase=0,
                amp=1,
                twidth=2e-6,
                env=waveform_cache[SquareWaveform(width=2e-6), "adc"],
                dest="Q0.rdlo",
                start_time=280,
            ),
        ]

        assert instructions == expected
        assert reads == Counter({"Q0.rdlo": 1})

    def test_compile(self, compiler, gates):
        pi = Timeline()
        pi.add(gates["Q0_X90"])
        pi.add(gates["Q0_X90"], gates["Q0_X90"].width)
        pi.add(gates["Q0_RO"], 2 * gates["Q0_X90"].width)

        nopi = Timeline()
        nopi.add(gates["Q0_RO"])

        seq = Sequence([nopi, pi])

        exe = compiler.compile(seq)

        print(exe)
