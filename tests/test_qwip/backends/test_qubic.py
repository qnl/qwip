from collections import Counter

import attrs
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

try:
    from distproc.compiler import CompiledProgram
    from distproc.executable import Executable
    from distproc.ir.instructions import BranchFproc, Idle, Pulse, VirtualZ

    from qwip.backends.qubic import (
        QubicCompiler,
        QubicExecutable,
        _get_board_and_core,
        _get_memory_name,
        find_constant_segments,
    )
except ImportError:
    pytest.skip("Qubic dependencies not installed.", allow_module_level=True)


from qwip.sequencer import (
    CWWaveform,
    Frame,
    GaussianWaveform,
    ModulatedWaveform,
    ResetOperation,
    Sequence,
    SquareWaveform,
    Timeline,
    VirtualZWaveform,
    active_reset,
)
from qwip.sequencer.compilation import ChannelInfo, DeviceInfo


def assert_instructions_almost_equal(ins1, ins2):
    assert type(ins1) == type(ins2)
    for f in attrs.fields(type(ins1)):
        if f.name == "env":
            if isinstance(ins1.env, str) and isinstance(ins2.env, str):
                assert ins1.env == ins2.env
            else:
                assert ins1.env.dtype == ins2.env.dtype
                assert_almost_equal(ins1.env, ins2.env)
        elif f.name == "phase":
            assert_almost_equal(
                ins1.phase % (2 * np.pi), ins2.phase % (2 * np.pi), decimal=6
            )
        else:
            assert getattr(ins1, f.name) == getattr(ins2, f.name)


def test_get_board_and_core(): ...


def test_get_memory_name(): ...


class TestQubicExecutable:
    def test_equality_and_hash(self):
        exe1 = QubicExecutable(
            program=CompiledProgram([]), assembly=Executable(), repetition_delay=1
        )
        exe2 = QubicExecutable(
            program=CompiledProgram([]), assembly=Executable(), repetition_delay=1
        )
        exe3 = QubicExecutable(
            program=exe1.program, assembly=exe1.assembly, repetition_delay=1
        )

        assert exe1 != exe2
        assert exe1 == exe3
        assert hash(exe1) == hash(exe3)

    def test_frozen(self):
        exe1 = QubicExecutable(
            program=CompiledProgram([]), assembly=Executable(), repetition_delay=1
        )

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            exe1.seq = Sequence([])


@pytest.mark.parametrize(
    "arr,locations,values,lengths",
    [
        (
            np.array([0, 1, 2, 2, 3, 3, 3]),
            np.array([0, 1, 2, 4]),
            np.array([0, 1, 2, 3]),
            np.array([1, 1, 2, 3]),
        ),
        (
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0, 1, 2, 3]),
            np.array([0, 1, 2, 3.0]),
            np.array([1, 1, 1, 1]),
        ),
    ],
)
def test_find_constant_segments(arr, locations, values, lengths):
    locs, vals, lens = find_constant_segments(arr)

    assert_equal(locs, locations)
    assert_equal(vals, values)
    assert_equal(lens, lengths)


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
                        channel=f"Q{q}.qdrv",
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
                        channel=f"Q{q}.rdrv",
                        frequency=f"Q{q}.readfreq",
                        hardware_modulation=True,
                    ),
                )
            )
            ro.add(
                ModulatedWaveform(
                    envelope=SquareWaveform(width=2e-6),
                    modulation=CWWaveform(
                        channel=f"Q{q}.rdlo",
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
            device = k[-4:]
            ch_id = ch_config.core_ind
            assert ch_config.elem_params == expected_elem_params[device]
            assert ch_config.elem_type == "rf"
            assert ch_config.env_mem_name == f"{device}env{ch_id}"
            assert ch_config.freq_mem_name == f"{device}freq{ch_id}"

            if device == "rdlo":
                assert ch_config.acc_mem_name == f"accbuf{ch_id}"

    @pytest.mark.parametrize(
        "location,wave,t0,expected",
        [
            (
                0,
                VirtualZWaveform(phase=90, frame="Q0.freq_GE"),
                0,
                [VirtualZ(qubit="Q0", phase=np.pi / 2, freq="freq_GE")],
            ),
            (
                0,
                VirtualZWaveform(phase=90, frame=5e9),
                0,
                [VirtualZ(phase=np.pi / 2, freq=5e9)],
            ),
            (
                50e-9,
                SquareWaveform(width=50e-9, amplitude=0.5, channel="Q0.qdrv"),
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
                0,
                ModulatedWaveform(
                    envelope=SquareWaveform(width=2e-6),
                    modulation=CWWaveform(
                        amplitude=0.5,
                        phase=180,
                        channel="Q1.rdlo",
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
            t0=t0,
            cw_threshold=None,
        )

        if "rdlo" in wave.channel:
            assert reads[wave.channel] == 1

        assert len(instructions) == len(expected)

        for i1, i2 in zip(instructions, expected):
            assert_instructions_almost_equal(i1, i2)

    @pytest.mark.parametrize(
        "wave,threshold,expected",
        [
            (
                ModulatedWaveform(
                    envelope=SquareWaveform(amplitude=0.5, width=100e-9),
                    modulation=CWWaveform(
                        amplitude=0.5,
                        frequency=100,
                        phase=180,
                        channel="Q0.rdrv",
                        hardware_modulation=True,
                    ),
                ),
                16,
                [
                    Pulse(
                        freq=100,
                        phase=0,
                        amp=0.5,
                        twidth=6e-9,
                        env=np.array([0, -0.5, -0.5]).astype(np.complex64),
                        dest="Q0.rdrv",
                        start_time=0,
                    ),
                    Pulse(
                        freq=100,
                        phase=np.pi,
                        amp=0.25,
                        twidth=88e-9,
                        env="cw",
                        dest="Q0.rdrv",
                        start_time=3,
                    ),
                    Pulse(
                        freq=100,
                        phase=0,
                        amp=0.5,
                        twidth=6e-9,
                        env=np.array([-0.5, -0.5, -0.5]).astype(np.complex64),
                        dest="Q0.rdrv",
                        start_time=47,
                    ),
                ],
            )
        ],
    )
    def test_compile_instruction_cw(self, compiler, wave, threshold, expected):
        instructions = compiler.compile_instruction(
            0,
            wave,
            waveform_cache={},
            reads=Counter(),
            t0=0,
            cw_threshold=threshold,
        )

        assert len(instructions) == len(expected)

        for i1, i2 in zip(instructions, expected):
            assert_instructions_almost_equal(i1, i2)

    def test_compile_timeline(self, compiler, gates):
        tmln = Timeline()
        tmln.add(gates["Q0_X90"])
        tmln.add(gates["Q0_X90"], gates["Q0_X90"].width)
        tmln.add(gates["Q0_RO"], 2 * gates["Q0_X90"].width)
        tmln.resolve()

        waveform_cache = {}
        instructions, reads = compiler.compile_timeline(
            tmln, waveform_cache=waveform_cache, cw_threshold=None
        )

        expected = [
            VirtualZ(qubit="Q0", phase=0, freq="freq_GE"),
            Pulse(
                freq="Q0.freq_GE",
                phase=0,
                amp=1,
                twidth=30e-9,
                env=waveform_cache[GaussianWaveform(width=30e-9), int(8e9)],
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
                env=waveform_cache[GaussianWaveform(width=30e-9), int(8e9)],
                dest="Q0.qdrv",
                start_time=15,
            ),
            VirtualZ(qubit="Q0", phase=0, freq="freq_GE"),
            Pulse(
                freq="Q0.readfreq",
                phase=0,
                amp=1,
                twidth=2e-6,
                env=waveform_cache[SquareWaveform(width=2e-6), int(0.5e9)],
                dest="Q0.rdrv",
                start_time=30,
            ),
            Pulse(
                freq="Q0.readfreq",
                phase=0,
                amp=1,
                twidth=2e-6,
                env=waveform_cache[SquareWaveform(width=2e-6), int(0.5e9)],
                dest="Q0.rdlo",
                start_time=280,
            ),
        ]

        assert instructions == expected
        assert reads == Counter({"Q0.rdlo": 1})

    def test_reset_operation_lowering(self, compiler, gates):
        """A `ResetOperation` lowers to one BranchFproc per round, each conditioned
        on the rdlo channel's `func_id`, with the X pulse in the `true` arm."""

        class FakeDB:
            def __init__(self, mapping):
                self._mapping = mapping

            def load_pulse(self, name):
                return self._mapping[name].copy()

        db = FakeDB(
            {"Q0_X90": gates["Q0_X90"], "Q0_measure": gates["Q0_RO"]}
        )
        reset = active_reset(db, "Q0", n_resets=2)

        tmln = Timeline()
        tmln.add(reset)
        tmln.width = reset.width
        tmln.resolve()

        instructions, reads = compiler.compile_timeline(
            tmln, waveform_cache={}, cw_threshold=None
        )

        branches = [ins for ins in instructions if isinstance(ins, BranchFproc)]
        assert len(branches) == 2, "expected one BranchFproc per reset round"

        # Q0.rdlo has index=0 in the compiler fixture, so func_id should be 0.
        for br in branches:
            assert br.cond_lhs == 1
            assert br.alu_cond == "eq"
            assert br.func_id == 0
            assert br.false == []
            # `true` arm contains the X180 — at minimum a Pulse on Q0.qdrv.
            qdrv_pulses_in_true = [
                ins
                for ins in br.true
                if isinstance(ins, Pulse) and ins.dest == "Q0.qdrv"
            ]
            assert len(qdrv_pulses_in_true) >= 1

        # Two measurement rounds => two rdlo pulses at the top level.
        rdlo_pulses = [
            ins
            for ins in instructions
            if isinstance(ins, Pulse) and ins.dest == "Q0.rdlo"
        ]
        assert len(rdlo_pulses) == 2

        # The internal measurements should NOT contribute to the user-visible
        # read counter (matches BranchOperation semantics).
        assert reads.get("Q0.rdlo", 0) == 0

    def test_reset_operation_measure_first_false(self, compiler, gates):
        """With `measure_first=False`, the first round skips its measurement,
        so only n_resets-1 internal measurements are emitted (the first
        BranchFproc reuses a measurement from the parent timeline)."""

        class FakeDB:
            def __init__(self, mapping):
                self._mapping = mapping

            def load_pulse(self, name):
                return self._mapping[name].copy()

        db = FakeDB(
            {"Q0_X90": gates["Q0_X90"], "Q0_measure": gates["Q0_RO"]}
        )
        reset = active_reset(db, "Q0", n_resets=2, measure_first=False)

        tmln = Timeline()
        tmln.add(reset)
        tmln.width = reset.width
        tmln.resolve()

        instructions, _ = compiler.compile_timeline(
            tmln, waveform_cache={}, cw_threshold=None
        )

        branches = [ins for ins in instructions if isinstance(ins, BranchFproc)]
        rdlo_pulses = [
            ins
            for ins in instructions
            if isinstance(ins, Pulse) and ins.dest == "Q0.rdlo"
        ]
        assert len(branches) == 2
        # n_resets=2, measure_first=False: skip round 0's measurement, keep round 1's.
        assert len(rdlo_pulses) == 1

    def test_compile_tight_packs_timelines(self, compiler, gates):
        """With default `reset_delay=0`, timelines pack back-to-back: the batch's
        total duration is the sum of timeline widths, not n * max."""
        short = Timeline()
        short.add(gates["Q0_RO"])
        short.width = 2.5e-6

        long_ = Timeline()
        long_.add(gates["Q0_X90"])
        long_.add(gates["Q0_X90"], gates["Q0_X90"].width)
        long_.add(gates["Q0_RO"], 2 * gates["Q0_X90"].width)
        long_.width = 2 * 30e-9 + 2.5e-6

        seq = Sequence([short, long_])
        exe = compiler.compile(seq, cw_threshold=None)

        # Both timelines in one batch; each gets a slot equal to its own width
        # (reset_delay=0 imposes no floor). Allow a clk-period of rounding slack.
        clk = compiler.fpga_config.fpga_clk_period
        expected = short.width + long_.width
        assert abs(exe[0].repetition_delay - expected) <= clk

    def test_compile_reset_delay_as_min_period(self, compiler, gates):
        """An explicit `reset_delay` acts as a floor: any timeline shorter than
        `reset_delay` gets padded up to it; longer timelines keep their width."""
        short = Timeline()
        short.add(gates["Q0_RO"])
        short.width = 2.5e-6

        seq = Sequence([short, short])
        exe = compiler.compile(seq, reset_delay=10e-6, cw_threshold=None)

        clk = compiler.fpga_config.fpga_clk_period
        # Each timeline (2.5us) gets padded to the 10us floor.
        assert abs(exe[0].repetition_delay - 2 * 10e-6) <= clk

    def test_compile(self, compiler, gates):
        pi = Timeline()
        pi.add(gates["Q0_X90"])
        pi.add(gates["Q0_X90"], gates["Q0_X90"].width)
        pi.add(gates["Q0_RO"], 2 * gates["Q0_X90"].width)

        nopi = Timeline()
        nopi.add(gates["Q0_RO"])

        seq = Sequence([nopi, pi])

        exe = compiler.compile(seq, cw_threshold=None)

        ops = exe.program.program[("Q0.qdrv", "Q0.rdrv", "Q0.rdlo")]

        assert ops[0] == dict(op="phase_reset")
        assert ops[-1] == dict(op="done_stb")

        start_times = [op["start_time"] for op in ops[1:-1]]
        assert start_times == [250_005, 250_255, 500_005, 500_020, 500_035, 500_285]

        dtypes = {op["env"].dtype for op in ops[1:-1]}
        shapes = [len(op["env"]) for op in ops[1:-1]]

        assert shapes == [1000, 1000, 240, 240, 1000, 1000]
        assert dtypes == {np.dtype(np.complex64)}

        dest = [op["dest"] for op in ops[1:-1]]
        assert dest == [
            "Q0.rdrv",
            "Q0.rdlo",
            "Q0.qdrv",
            "Q0.qdrv",
            "Q0.rdrv",
            "Q0.rdlo",
        ]
