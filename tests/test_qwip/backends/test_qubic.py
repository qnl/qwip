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
        _TimedBranch,
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
            qubit,
            sample_rate=8e9,
            envelope_sample_rate=8e9,
            name="b0_c0_qdrv",
            dtype=np.complex64,
        )
        readout = DeviceInfo.from_channels(
            readout,
            sample_rate=8e9,
            envelope_sample_rate=0.5e9,
            name="b0_c0_rdrv",
            dtype=np.complex64,
        )
        adc = DeviceInfo.from_channels(
            adc,
            sample_rate=2e9,
            envelope_sample_rate=0.5e9,
            name="b0_c0_rdlo",
            dtype=np.complex64,
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
            X90.add(VirtualZWaveform(frame=f"Q{q}.freq_GE", channel=f"Q{q}.qdrv"))
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
            X90.add(
                VirtualZWaveform(frame=f"Q{q}.freq_GE", channel=f"Q{q}.qdrv"), 30e-9
            )
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
            assert ch_config.elem_type == ("rf_mix" if device == "rdlo" else "rf")
            assert ch_config.env_mem_name == f"c0_{device}_env{ch_id}"
            assert ch_config.freq_mem_name == f"c0_{device}_freq{ch_id}"

            if device == "rdlo":
                assert ch_config.acc_mem_name == f"c0_accbuf{ch_id}"

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

        branches = [
            ins.branch for ins in instructions if isinstance(ins, _TimedBranch)
        ]
        assert len(branches) == 2, "expected one BranchFproc per reset round"

        # The branch reads the named FPROC channel the compiler registers for
        # Q0's readout; distproc resolves it to a core index later.
        for br in branches:
            assert br.cond_lhs == 1
            assert br.alu_cond == "eq"
            assert br.func_id == "Q0.meas"
            assert br.false == []
            # `true` arm contains the X180 — at minimum a Pulse on Q0.qdrv.
            qdrv_pulses_in_true = [
                ins
                for ins in br.true
                if isinstance(ins, Pulse) and ins.dest == "Q0.qdrv"
            ]
            assert len(qdrv_pulses_in_true) >= 1

        # Two measurement rounds => two rdlo pulses at the top level. The square
        # readout envelope is split into head/CW/tail pieces, so count the CW one.
        rdlo_pulses = [
            ins
            for ins in instructions
            if isinstance(ins, Pulse)
            and ins.dest == "Q0.rdlo"
            and isinstance(ins.env, str)
            and ins.env == "cw"
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

        branches = [
            ins.branch for ins in instructions if isinstance(ins, _TimedBranch)
        ]
        rdlo_pulses = [
            ins
            for ins in instructions
            if isinstance(ins, Pulse)
            and ins.dest == "Q0.rdlo"
            and isinstance(ins.env, str)
            and ins.env == "cw"
        ]
        assert len(branches) == 2
        # n_resets=2, measure_first=False: skip round 0's measurement, keep round 1's.
        assert len(rdlo_pulses) == 1

    def test_compile_tight_packs_timelines(self, compiler, gates):
        """With default `reset_delay=None`, timelines pack back-to-back: the batch's
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

    def test_compile_reset_delay_as_uniform_period(self, compiler, gates):
        """An explicit `reset_delay` gives every timeline a uniform slot."""
        short = Timeline()
        short.add(gates["Q0_RO"])
        short.width = 2.5e-6

        seq = Sequence([short, short])
        exe = compiler.compile(seq, reset_delay=10e-6, cw_threshold=None)

        clk = compiler.fpga_config.fpga_clk_period
        # Each 2.5us timeline gets a 10us slot.
        assert abs(exe[0].repetition_delay - 2 * 10e-6) <= clk

    def test_compile_holds_until_the_end_of_the_last_slot(self, compiler, gates):
        """The gateware starts the next shot as soon as the last instruction
        ends, so the wait after the last timeline must be an explicit idle."""
        short = Timeline()
        short.add(gates["Q0_RO"])
        short.width = 2.5e-6

        exe = compiler.compile(Sequence([short]), reset_delay=10e-6, cw_threshold=None)[0]

        clk = compiler.fpga_config.fpga_clk_period
        ends = [
            ins["end_time"]
            for core in exe.program.program.values()
            for ins in core
            if ins.get("op") == "idle"
        ]
        assert ends
        assert min(ends) * clk >= 10e-6 - clk

    def test_construct_circuit_reset_delay_too_short_raises(self, compiler, gates):
        """An explicit `reset_delay` shorter than the longest timeline raises."""
        long_ = Timeline()
        long_.add(gates["Q0_RO"])
        long_.width = 5e-6

        seq = Sequence([long_])
        with pytest.raises(ValueError, match="reset_delay"):
            compiler.construct_circuit(seq, reset_delay=1e-6, cw_threshold=None)

    @staticmethod
    def _run_distproc_passes(compiler, seq):
        """Compile-only run of distproc's IR passes on a sequence.

        Returns the nominal batch length in seconds, the scheduled program
        length in seconds, and the scheduled IR program.
        """
        import itertools

        from distproc.compiler import Compiler as _QubicInternalCompiler
        from distproc.compiler import CompilerFlags, get_passes

        frame_declarations = compiler.get_frame_declarations(compiler.frame_scopes)
        proc_grouping = compiler.get_proc_grouping()
        passes = get_passes(
            compiler.fpga_config,
            qchip=compiler.get_qchip(),
            compiler_flags=CompilerFlags(
                schedule=False, resolve_gates=False, multi_board=True
            ),
            qubit_grouping=list(itertools.chain(*proc_grouping)),
            proc_grouping=proc_grouping,
        )
        circuit, _, nominal = compiler.construct_circuit(
            seq.flatten(), None, frame_declarations, {}, cw_threshold=None
        )
        qc = _QubicInternalCompiler(circuit, proc_grouping=proc_grouping)
        qc.run_ir_passes(passes)
        end = max(b.get("end_time", 0) for b in qc.ir_prog.blocks.values())
        return nominal, end * compiler.fpga_config.fpga_clk_period, qc.ir_prog

    @staticmethod
    def _reset_db(gates):
        class FakeDB:
            def __init__(self, mapping):
                self._mapping = mapping

            def load_pulse(self, name):
                return self._mapping[name].copy()

        return FakeDB({"Q0_X90": gates["Q0_X90"], "Q0_measure": gates["Q0_RO"]})

    @pytest.mark.parametrize("n_timelines", [1, 2, 4])
    def test_active_reset_schedules_to_nominal_length(
        self, compiler, gates, n_timelines
    ):
        """distproc re-bases every basic block by its predecessor's end. With
        absolute timestamps that made actively reset programs grow roughly
        quadratically with the number of branches; `_rebase_blocks` must make the
        scheduled length equal the nominal length (plus distproc's 5-clock start
        offset) regardless of how many timelines share a batch."""
        db = self._reset_db(gates)

        def proto():
            return Timeline.from_layers(
                [active_reset(db, "Q0", n_resets=2), gates["Q0_X90"], gates["Q0_RO"]]
            )

        seq = Sequence([proto() for _ in range(n_timelines)])
        nominal, scheduled, ir_prog = self._run_distproc_passes(compiler, seq)

        clk = compiler.fpga_config.fpga_clk_period
        assert abs(scheduled - nominal) <= 8 * clk

        # The first conditional X pulse sits exactly `fproc_delay` after the
        # first measurement: 5 (start) + 1250 (2.5 us readout) + 200 (400 ns).
        qdrv = [
            ins
            for block in ir_prog.blocks.values()
            for ins in block.get("instructions", [])
            if isinstance(ins, Pulse) and ins.dest == "Q0.qdrv"
        ]
        assert min(ins.start_time for ins in qdrv) == 5 + 1250 + 200

    @staticmethod
    def _nominal_pulses(compiler, seq):
        """(dest, start clock, twidth) of every pulse, at the absolute times
        `construct_circuit` lays out before `_rebase_blocks` runs."""
        import unittest.mock as mock

        def walk(instrs):
            for ins in instrs:
                if isinstance(ins, Pulse):
                    yield (ins.dest, ins.start_time, ins.twidth)
                elif isinstance(ins, _TimedBranch):
                    yield from walk(ins.branch.true)
                    yield from walk(ins.branch.false)

        with mock.patch.object(QubicCompiler, "_rebase_blocks", lambda self, c: c):
            circuit, _, _ = compiler.construct_circuit(
                seq.flatten(), None, [], {}, cw_threshold=None
            )
        return sorted(walk(circuit))

    @staticmethod
    def _scheduled_pulses(ir_prog):
        return sorted(
            (ins.dest, ins.start_time, ins.twidth)
            for block in ir_prog.blocks.values()
            for ins in block.get("instructions", [])
            if isinstance(ins, Pulse)
        )

    @pytest.mark.parametrize("qubits", [["Q0"], ["Q0", "Q1"]])
    def test_active_reset_pulses_land_at_nominal_times(self, compiler, gates, qubits):
        """After distproc's Schedule pass every pulse, in every arm of every
        branch and in every following timeline, sits at the clock qwip laid it
        out at. Two qubits reset at the same time (as in a two-qubit sweep) use
        two branches on two cores between the same measurements."""

        class FakeDB:
            def load_pulse(self, name):
                q, kind = name.split("_")
                return gates[f"{q}_X90" if kind == "X90" else f"{q}_RO"].copy()

        db = FakeDB()

        def proto():
            return Timeline.from_layers(
                [
                    [active_reset(db, q, n_resets=2) for q in qubits],
                    [gates[f"{q}_X90"] for q in qubits],
                    [gates[f"{q}_RO"] for q in qubits],
                ]
            )

        seq = Sequence([proto() for _ in range(3)])
        nominal = self._nominal_pulses(compiler, seq)
        _, _, ir_prog = self._run_distproc_passes(compiler, seq)
        assert self._scheduled_pulses(ir_prog) == nominal

    def test_reset_arm_folds_virtual_z(self, compiler, gates):
        """Inside a reset arm the X pulse's virtual-Z markers are folded into
        the pulse phases and no `VirtualZ` instruction is emitted, so distproc
        sees identical frame phases on both arms."""
        x90 = gates["Q0_X90"]
        # Give the markers a nonzero phase: 10 degrees before and after the pulse.
        x90 = Timeline.fromtuples(
            [
                (0, VirtualZWaveform(frame="Q0.freq_GE", channel="Q0.qdrv", phase=10)),
                (0, x90.operations[1]),
                (30e-9, VirtualZWaveform(frame="Q0.freq_GE", channel="Q0.qdrv", phase=10)),
            ],
            width=30e-9,
        )

        class FakeDB:
            def load_pulse(self, name):
                return (x90 if name.endswith("X90") else gates["Q0_RO"]).copy()

        tmln = Timeline.from_layers([active_reset(FakeDB(), "Q0", n_resets=1)])
        instructions, _ = compiler.compile_timeline(
            tmln, waveform_cache={}, cw_threshold=None
        )
        (branch,) = [ins.branch for ins in instructions if isinstance(ins, _TimedBranch)]
        assert not any(isinstance(ins, VirtualZ) for ins in branch.true)
        pulses = [ins for ins in branch.true if isinstance(ins, Pulse)]
        assert len(pulses) == 2
        # First pulse: one 10 degree marker before it; second: three.
        assert_almost_equal(pulses[0].phase, np.deg2rad(10))
        assert_almost_equal(pulses[1].phase, np.deg2rad(30))

    @staticmethod
    def _classified_db(gates, ground, excited):
        """FakeDB with Q0's pulses and a stored two-blob readout classification."""

        class FakeDB:
            config = {
                "readout": {
                    "default": {
                        "registers": {
                            "Q0": {"classification": {"means": np.stack([ground, excited])}}
                        }
                    }
                }
            }

            def load_pulse(self, name):
                return (gates["Q0_X90"] if name.endswith("X90") else gates["Q0_RO"]).copy()

        return FakeDB()

    def test_hardware_demod_rotation(self, gates):
        """The rotation aligns the ground-to-excited separation with negative I,
        the threshold is the midpoint I in that frame, and `active_reset`
        applies the rotation to the reset's demod waveform only and carries the
        threshold."""
        from qwip.sequencer import hardware_demod_rotation, hardware_state_threshold

        ground = np.array([-10048.2, 18203.2])
        excited = np.array([45536.1, 7804.5])
        db = self._classified_db(gates, ground, excited)

        rotation = hardware_demod_rotation(db, "Q0")
        # A demod phase change of +phi rotates I/Q by -phi.
        r = np.deg2rad(-rotation)
        rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        g, e = rot @ ground, rot @ excited
        assert e[0] < g[0]
        assert_almost_equal(e[1], g[1], decimal=3)
        assert_almost_equal(g[0] - e[0], np.linalg.norm(excited - ground), decimal=3)

        threshold = hardware_state_threshold(db, "Q0")
        assert threshold == int(round((g[0] + e[0]) / 2))
        assert e[0] < threshold < g[0]
        # With the stored phase kept, the threshold is the stored midpoint I.
        assert hardware_state_threshold(db, "Q0", demod_rotation=0) == int(
            round((ground[0] + excited[0]) / 2)
        )

        reset = active_reset(db, "Q0", n_resets=1)
        assert reset.threshold == threshold
        stored = gates["Q0_RO"].copy()
        for (_, op), (_, ref) in zip(reset.measurement, stored):
            if op.channel == "Q0.rdlo":
                assert_almost_equal(op.modulation.phase - ref.modulation.phase, rotation)
            else:
                assert op.modulation.phase == ref.modulation.phase

    def test_active_reset_without_classification_has_no_threshold(self, gates):
        reset = active_reset(self._reset_db(gates), "Q0", n_resets=1)
        assert reset.threshold is None

    def test_compile_writes_state_threshold(self, compiler, gates):
        """A compiled batch containing an `active_reset` carries the readout
        board's threshold register for the reset's channel, named after the
        device (board b0, core c0, siggen rdlo, channel index 0)."""
        db = self._classified_db(
            gates, np.array([-10048.2, 18203.2]), np.array([45536.1, 7804.5])
        )
        reset = active_reset(db, "Q0", n_resets=1)
        tmln = Timeline.from_layers([reset, gates["Q0_RO"]])
        exe = compiler.compile(Sequence([tmln]), cw_threshold=None)
        assert exe[0].assembly.write_registers == {"b0:thresh_c0_rdlo_0": reset.threshold}

        # An explicit threshold overrides the computed one.
        forced = active_reset(db, "Q0", n_resets=1, threshold=1234)
        tmln = Timeline.from_layers([forced, gates["Q0_RO"]])
        exe = compiler.compile(Sequence([tmln]), cw_threshold=None)
        assert exe[0].assembly.write_registers == {"b0:thresh_c0_rdlo_0": 1234}

        # No stored classification: nothing is written.
        plain = active_reset(self._reset_db(gates), "Q0", n_resets=1)
        tmln = Timeline.from_layers([plain, gates["Q0_RO"]])
        exe = compiler.compile(Sequence([tmln]), cw_threshold=None)
        assert exe[0].assembly.write_registers == {}

    def test_compile_conflicting_state_thresholds_raise(self, compiler, gates):
        db = self._classified_db(
            gates, np.array([-10048.2, 18203.2]), np.array([45536.1, 7804.5])
        )
        a = Timeline.from_layers([active_reset(db, "Q0", n_resets=1, threshold=1), gates["Q0_RO"]])
        b = Timeline.from_layers([active_reset(db, "Q0", n_resets=1, threshold=2), gates["Q0_RO"]])
        with pytest.raises(ValueError, match="different state thresholds"):
            compiler.compile(Sequence([a, b]), cw_threshold=None)

    def test_active_reset_gap_too_short_raises(self, compiler, gates):
        """The branch needs fproc_meas_clks + pulse_load_clks + jump_fproc_clks
        (150 ns) after the demod window; a shorter `fproc_delay` is an error
        that names the required timing rather than a silently late X pulse."""
        db = self._reset_db(gates)
        tmln = Timeline.from_layers(
            [active_reset(db, "Q0", n_resets=1, fproc_delay=100e-9), gates["Q0_RO"]]
        )
        with pytest.raises(ValueError, match="before its basic block can begin"):
            compiler.construct_circuit(
                Sequence([tmln]).flatten(), None, [], {}, cw_threshold=None
            )

    def test_pulse_overlapping_branch_raises(self, compiler, gates):
        """distproc cannot schedule a pulse that spans a branch."""
        db = self._reset_db(gates)
        tmln = Timeline.from_layers(
            [active_reset(db, "Q0", n_resets=1), gates["Q0_RO"]]
        )
        # A drive on another qubit that starts during the reset's readout and
        # runs through the branch.
        tmln.add(
            ModulatedWaveform(
                envelope=SquareWaveform(width=3e-6),
                modulation=CWWaveform(
                    channel="Q1.qdrv", frequency="Q1.freq_GE", hardware_modulation=True
                ),
            ),
            1e-6,
        )
        with pytest.raises(ValueError, match="branch"):
            compiler.construct_circuit(
                Sequence([tmln]).flatten(), None, [], {}, cw_threshold=None
            )

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
