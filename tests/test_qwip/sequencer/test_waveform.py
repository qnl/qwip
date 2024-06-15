from contextlib import nullcontext
from copy import copy, deepcopy

import attrs
import matplotlib.pyplot as plt
import numpy as np
import pytest
import sympy as sym
from numpy.testing import assert_allclose, assert_almost_equal

import qwip
from qwip.sequencer.phase_tracker import Frame, PhaseJump, PhaseTracker
from qwip.sequencer.timeline import Timeline
from qwip.sequencer.utils import Location
from qwip.sequencer.waveform import (  # update_fields,
    DRAG,
    BasicWaveform,
    ConvolvedWaveform,
    CWWaveform,
    DCWaveform,
    GaussianWaveform,
    Marker,
    ModulatedWaveform,
    Operation,
    PhaseResetWaveform,
    SquareWaveform,
    TriggeredWaveform,
    VirtualZWaveform,
    Waveform,
)


class TestOperation:
    @pytest.mark.parametrize("op", [Operation(), Operation(name="op")])
    def test_copy(self, op):
        assert copy(op) is op
        assert deepcopy(op) is op

    def test_getattribute(self):
        w = BasicWaveform(width="tau")

        assert isinstance(w.name, str)
        assert isinstance(w.amplitude, int | float)
        assert isinstance(w.width, sym.Expr)

    @pytest.mark.parametrize(
        "wave,updates,expect",
        [
            (
                SquareWaveform(),
                dict(width=10),
                SquareWaveform(width=10),
            ),
            (
                DCWaveform(amplitude=1),
                dict(width=10),
                DCWaveform(amplitude=1),
            ),
            (
                DRAG(envelope=SquareWaveform()),
                dict(width=10, lmbda=1),
                DRAG(envelope=SquareWaveform(width=10), lmbda=1),
            ),
            (
                DRAG(envelope=SquareWaveform()),
                dict(name="new_name"),
                DRAG(name="new_name", envelope=SquareWaveform()),
            ),
            (
                DRAG(envelope=SquareWaveform()),
                dict(envelope_name="new_name"),
                DRAG(envelope=SquareWaveform(name="new_name")),
            ),
            (
                ModulatedWaveform(
                    envelope=DRAG(envelope=SquareWaveform()),
                    modulation=CWWaveform(frequency="f"),
                ),
                dict(envelope_width=10),
                ModulatedWaveform(
                    envelope=DRAG(envelope=SquareWaveform(width=10)),
                    modulation=CWWaveform(frequency="f"),
                ),
            ),
            (
                DRAG(envelope=SquareWaveform()),
                dict(envelope=GaussianWaveform()),
                DRAG(envelope=GaussianWaveform()),
            ),
            (
                ModulatedWaveform(
                    envelope=SquareWaveform(), modulation=CWWaveform(frequency="f")
                ),
                dict(modulation_frequency="f1"),
                ModulatedWaveform(
                    envelope=SquareWaveform(), modulation=CWWaveform(frequency="f1")
                ),
            ),
        ],
    )
    def test_evolve(self, wave, updates, expect):
        assert wave.evolve(**updates) == expect

    @pytest.mark.parametrize(
        "op,variables",
        [
            (BasicWaveform(), set()),
            (
                BasicWaveform(amplitude="amplitude", width="width", t0="t0"),
                {"amplitude", "width", "t0"},
            ),
            (BasicWaveform(amplitude="A"), {"A"}),
        ],
    )
    def test_variables(self, op, variables):
        assert op.variables() == variables

    @pytest.mark.parametrize(
        "op,expect",
        [
            (BasicWaveform(), True),
            (BasicWaveform(amplitude=0.5), True),
            (BasicWaveform(amplitude="amp"), False),
        ],
    )
    def test_resolved(self, op, expect):
        assert op.resolved is expect

    @pytest.mark.parametrize(
        "op,mapping,expect",
        [
            (
                BasicWaveform(amplitude="amp"),
                dict(amp=0.5),
                BasicWaveform(amplitude=0.5),
            ),
            (
                BasicWaveform(width="tau0"),
                dict(tau0="tau1"),
                BasicWaveform(width="tau1"),
            ),
        ],
    )
    def test_resolve_basic(self, op, mapping, expect):
        assert op.resolve(**mapping) == expect

    def test_resolve_nonlinear(self):
        op = VirtualZWaveform(frame="Q0.GE", phase="360*frequency*time")

        assert op.resolve(frequency=5e9) == VirtualZWaveform(
            frame="Q0.GE", phase="360 * 5e9 * time"
        )
        assert op.resolve(time=10e-9) == VirtualZWaveform(
            frame="Q0.GE", phase="360 * frequency * 10e-9"
        )
        assert op.resolve(frequency=5e9, time=10e-9) == VirtualZWaveform(
            frame="Q0.GE", phase=360 * 50
        )

    def test_resolve_no_update(self):
        op = BasicWaveform(amplitude="amp")

        assert op.resolve() is op
        assert op.resolve(width="width") is op

    @pytest.mark.parametrize(
        "frequency,var_map,expect",
        [
            ("frequency", dict(frequency=5e9), Frame(5e9)),
            ("(f_a - f_b) / 2", dict(f_a=5e9), (Frame(5e9) - Frame("f_b")) / 2),
            (
                "frequency",
                dict(frequency=sym.parse_expr("(f_a - f_b) / 2")),
                Frame.from_string("0.5 * (f_a - f_b)"),
            ),
        ],
    )
    def test_resolve_frame(self, frequency, var_map, expect):
        op = CWWaveform(frequency=frequency)
        resolved = op.resolve(**var_map)

        assert resolved.frequency == expect

    # @pytest.mark.parametrize(
    #     "wave,variables,expect",
    #     [
    #         (
    #             Marker(name="m", channel="ch"),
    #             dict(m="ro_marker", ch="ro_channel"),
    #             Marker(name="ro_marker", channel="ro_channel"),
    #         ),
    #         (
    #             Marker(name="m", channel="ch"),
    #             dict(name="ro_marker", channel="ro_channel"),
    #             Marker(name="m", channel="ch"),
    #         ),
    #         (
    #             Marker(name="name", channel="channel"),
    #             dict(name="ro_marker", channel="ro_channel"),
    #             Marker(name="ro_marker", channel="ro_channel"),
    #         ),
    #     ],
    # )
    # def test_resolve_str(self, wave, variables, expect):
    #     resolved = wave.resolve(**variables)

    #     assert resolved == expect


class TestBasicWaveform:
    @pytest.mark.parametrize("name,expect", [(None, "BasicWaveform"), ("name", "name")])
    def test_name(self, name, expect):
        w = BasicWaveform(name=name) if name else BasicWaveform()
        assert w.name == expect

    def test_create(self):
        w = BasicWaveform()

        assert w.name == "BasicWaveform"
        assert w.channel == ""
        assert w.width == 0
        assert w.amplitude == 1
        assert w.t0 == 0

        w = BasicWaveform(name="name")
        assert w.name == "name"

    def test_convert(self):
        w = BasicWaveform(channel=0, width=Location(1), amplitude="A")

        assert w.channel == "0"
        assert w.width == 1
        assert w.amplitude == sym.Symbol("A")

    @pytest.mark.parametrize(
        "kwargs,expect",
        [
            (
                dict(),
                dict(width=sym.Symbol("w"), amplitude=sym.Symbol("amp"), phase=0, t0=0),
            ),
            (dict(w=10, amp=20), dict(width=10.0, amplitude=20, phase=0, t0=0)),
            (
                dict(width=10, amplitude=20, t0=1),
                dict(width=10, amplitude=20, phase=0, t0=1),
            ),
            (
                dict(random=10),
                dict(
                    width=sym.Symbol("w"),
                    amplitude=sym.Symbol("amp"),
                    phase=0,
                    t0=0,
                    random=10,
                ),
            ),
        ],
    )
    def test_update_fields(self, kwargs, expect):
        w = BasicWaveform(width="w", amplitude="amp")

        assert w._update_fields(**kwargs) == expect

    def test_variables(self):
        assert BasicWaveform().variables() == frozenset()

        hits = Waveform.variables.cache_info().hits

        wave = BasicWaveform(width="w")
        assert wave.variables() == frozenset({"w"})
        assert wave.variables() is wave.variables()

        assert wave.variables.cache_info().hits == hits + 2

    @pytest.mark.parametrize(
        "wave,vmap,new",
        [
            (
                BasicWaveform(width="w", amplitude="a"),
                dict(w=1),
                BasicWaveform(width=1, amplitude="a"),
            ),
            (
                BasicWaveform(width="w", amplitude="a"),
                dict(c=1),
                BasicWaveform(width="w", amplitude="a"),
            ),
            (
                BasicWaveform(amplitude="name"),
                dict(name=0.2),
                BasicWaveform(amplitude=0.2),
            ),
        ],
    )
    def test_resolve(self, wave, vmap, new):
        assert wave.resolve(**vmap) == new

    def test_assign_channel(self):
        wave = BasicWaveform(channel="channel").assign_channel("new_channel")
        assert wave.channel == "new_channel"

    def test_legacy_serialization(self):
        unstruct = dict(
            width=4e-8, amplitude=0.5, channels=["I", "Q"], __class__="GaussianWaveform"
        )
        structured = qwip.converter.structure(unstruct, Waveform)

        wave = GaussianWaveform(width=40e-9, amplitude=0.5, channel="IQ")

        assert structured == wave


class TestConvolvedWaveform:
    @pytest.mark.parametrize(
        "a,b,should_raise",
        [
            (SquareWaveform(), SquareWaveform(), False),
            (SquareWaveform(channel="Q0"), SquareWaveform(), False),
            (SquareWaveform(), SquareWaveform(channel="Q1"), False),
            (SquareWaveform(channel="Q0"), SquareWaveform(channel="Q1"), True),
        ],
    )
    def test_init(self, a, b, should_raise):
        maybe_raise = pytest.raises(ValueError) if should_raise else nullcontext()

        with maybe_raise:
            c = ConvolvedWaveform(a=a, b=b)

            assert c.a == a
            assert c.b == b

    @pytest.mark.parametrize(
        "a,b,expect",
        [
            (
                SquareWaveform(width=20e-9, phase=15, t0=-20e-9),
                SquareWaveform(width=10e-9, phase=30, t0=10e-9),
                dict(width=10e-9 + 20e-9, phase=45, t0=-10e-9),
            ),
            (
                ModulatedWaveform(
                    envelope=SquareWaveform(width="width", phase="phase", t0="t0"),
                    modulation=CWWaveform(frequency="frequency"),
                ),
                GaussianWaveform(width=10e-9, phase=0, t0=0),
                dict(
                    width=sym.Symbol("width") + 10e-9,
                    phase=sym.Symbol("phase"),
                    t0=sym.Symbol("t0"),
                ),
            ),
        ],
    )
    def test_properties(self, a, b, expect):
        c = a * b

        for attribute, value in expect.items():
            assert getattr(c, attribute) == value

    @pytest.mark.parametrize(
        "ts,a,b",
        [
            (
                np.arange(200) / 1e9,
                GaussianWaveform(width=20e-9, amplitude=1 / 8, phase=45),
                SquareWaveform(width=100e-9, phase=45),
            ),
            (
                np.arange(200) / 1e9 - 80e-9,
                GaussianWaveform(width=20e-9, amplitude=1 / 8, phase=45),
                SquareWaveform(width=100e-9, phase=45),
            ),
            (
                np.arange(120) / 1e9,
                GaussianWaveform(width=20e-9, amplitude=1 / 8, t0=50e-9),
                SquareWaveform(width=100e-9, t0=-50e-9),
            ),
        ],
    )
    def test_evaluate(self, ts, a, b, data_file):
        expected = np.loadtxt(str(data_file), dtype=np.complex64)

        wave = a * b
        w_t = wave(ts)

        assert_allclose(w_t, expected, atol=5e-7)

    def test_assign_channel(self):
        wave = GaussianWaveform() * SquareWaveform()

        new_wave = wave.assign_channel("new_channel")
        assert new_wave.channel == "new_channel"
        assert new_wave.a.channel == "new_channel"
        assert new_wave.b.channel == "new_channel"


class TestCWWaveform:
    @pytest.mark.parametrize(
        "ts,phase_jumps",
        [
            (np.linspace(0, 5, 21), np.array([(0, 0)])),
            (np.linspace(0, 5, 21), np.array([(0, 0), (1.25, 90), (2.5, -180)])),
            (np.linspace(1.25, 6.25, 21), np.array([(0, 0), (1, -90)])),
        ],
    )
    def test_modulation(self, ts, phase_jumps, data_file):
        expected = np.loadtxt(str(data_file))

        w = CWWaveform(frequency=Frame(0.2), channel="IQ")

        phase_tracker = PhaseTracker(
            phases={Frame(0.2): [PhaseJump(t, pj) for t, pj in phase_jumps]}
        )

        wave = w(ts, phase_tracker=phase_tracker, phase_unit="degrees")
        assert_allclose(wave, expected[0] + 1j * expected[1], atol=2e-6)

    @pytest.mark.parametrize(
        "ts,phase_jumps",
        [
            (np.linspace(0, 15e-9, 61), np.array([(0, 0), (10e-9, 90), (25e-9, -180)])),
            (np.linspace(0, 20e-9, 81), np.array([(0, 0), (5e-9, -90), (15e-9, 45)])),
        ],
    )
    def test_hardware_modulation(self, ts, phase_jumps, data_file):
        expected = np.loadtxt(str(data_file), dtype=np.complex64)
        w = CWWaveform(
            frequency=Frame(5e9),
            channel="Q0.drv",
            hardware_modulation=True,
        )

        phase_tracker = PhaseTracker(
            phases={Frame(5e9): [PhaseJump(t, pj) for t, pj in phase_jumps]}
        )

        wave = w(
            ts, phase_tracker=phase_tracker, phase_unit="degrees", complex_out=True
        )
        assert_allclose(wave, expected)

    @pytest.mark.parametrize(
        "ts,phase_jumps",
        [
            (np.linspace(0, 10, 51), np.array([(0, 0), (1, 90)])),
        ],
    )
    def test_units(self, ts, phase_jumps):
        pj_deg = phase_jumps
        pj_rad = phase_jumps * np.array([[1, np.pi / 180]])

        mod_freq = Frame(0.5)

        phase_tracker_deg = PhaseTracker(
            phases={mod_freq: [PhaseJump(t, pj) for t, pj in pj_deg]}
        )

        phase_tracker_rad = PhaseTracker(
            phases={mod_freq: [PhaseJump(t, pj) for t, pj in pj_rad]}
        )

        w = CWWaveform(frequency=mod_freq, channel="IQ")

        wave_d = w(ts, phase_tracker=phase_tracker_deg, phase_unit="degrees")
        wave_r = w(ts, phase_tracker=phase_tracker_rad, phase_unit="radians")

        assert_allclose(wave_d, wave_r)


class TestModulatedWaveform:
    @pytest.mark.parametrize(
        "env,mod,expected",
        [
            (
                dict(t0=1, width=10),
                dict(amplitude=2),
                dict(t0=1, width=10, amplitude=2),
            ),
            (
                dict(width=20, amplitude=2),
                dict(amplitude=0.5),
                dict(t0=0, width=20, amplitude=1),
            ),
        ],
    )
    def test_properties(self, env, mod, expected):
        env = SquareWaveform(**env)
        mod = CWWaveform(frequency="f", **mod)

        wave = ModulatedWaveform(envelope=env, modulation=mod)

        assert dict((p, getattr(wave, p)) for p in expected.keys()) == expected

    @pytest.mark.parametrize(
        "env,vmap,new_env",
        [
            (
                dict(width="w", amplitude="a", t0="t"),
                dict(w=1, a=2, t=3),
                dict(width=1, amplitude=2, t0=3),
            ),
        ],
    )
    def test_resolve(self, env, vmap, new_env):
        env = SquareWaveform(**env)
        new_env = SquareWaveform(**new_env)

        mod = CWWaveform(frequency="f")

        wave = ModulatedWaveform(envelope=env, modulation=mod)
        expected = ModulatedWaveform(envelope=new_env, modulation=mod)

        assert wave.resolve(**vmap) == expected

    @pytest.mark.parametrize(
        "ts,phase_jumps",
        [
            (np.arange(240) / 2.4e9, np.array([(0, 0)])),
            (np.arange(240) / 2.4e9, np.array([(0, 0), (20e-9, 180)])),
            (np.arange(240) / 2.4e9, np.array([(0, 0), (60e-9, 90)])),
        ],
    )
    def test_IQ_modulation(self, ts, phase_jumps, data_file):
        expected = np.loadtxt(str(data_file))

        freq = Frame(100e6)
        env = SquareWaveform(width=40e-9)
        mod = CWWaveform(frequency=freq, channel="IQ")

        phase_tracker = PhaseTracker(
            phases={freq: [PhaseJump(t, pj) for t, pj in phase_jumps]}
        )

        w = ModulatedWaveform(envelope=env, modulation=mod)

        ts = np.arange(240) / 2.4e9
        wave = w(ts, t0=40e-9, phase_tracker=phase_tracker)

        assert_allclose(wave, expected[0] + 1j * expected[1], atol=5e-6)

    def test_hardware_modulation(self):
        env = GaussianWaveform(width=50e-9)
        mod = CWWaveform(frequency=5e9, amplitude=0.5, hardware_modulation=True)

        wave = ModulatedWaveform(envelope=env, modulation=mod)

        ts = np.arange(400) / 8e9
        assert_almost_equal(wave(ts, complex_out=True).real, wave.envelope(ts))

    @pytest.mark.parametrize(
        "wave,val",
        [
            (
                ModulatedWaveform(
                    name="X90",
                    envelope=GaussianWaveform(width=40e-9, amplitude=0.5),
                    modulation=CWWaveform(frequency="f", channel="IQ"),
                ),
                dict(
                    name="X90",
                    envelope=dict(
                        width=4e-8, amplitude=0.5, __class__="GaussianWaveform"
                    ),
                    modulation=dict(
                        channel="IQ",
                        frequency="f",
                        __class__="CWWaveform",
                    ),
                    __class__="ModulatedWaveform",
                ),
            ),
        ],
    )
    def test_serialization(self, wave, val):
        unstruct = qwip.converter.unstructure(wave)
        restruct = qwip.converter.structure(unstruct, ModulatedWaveform)

        assert unstruct == val
        assert restruct == wave

    def test_assign_channel(self):
        wave = ModulatedWaveform(
            envelope=SquareWaveform(),
            modulation=CWWaveform(frequency=5e9, channel="IQ"),
        )

        new_wave = wave.assign_channel("I_Q")
        assert new_wave.channel == "I_Q"
        assert new_wave.modulation.channel == "I_Q"
        assert new_wave.envelope.channel == ""


class TestDRAGWaveform:
    def test_suppression(self):
        f0 = 500e6
        f1 = -100e6
        sample_rate = 2.4e9
        env = DRAG(
            envelope=GaussianWaveform(width=20e-9), lmbda=1 / (2 * np.pi * f1)
        )

        freq = CWWaveform(frequency=Frame(f0), channel="IQ")

        wave_drag = ModulatedWaveform(envelope=env, modulation=freq)
        wave_nodrag = ModulatedWaveform(envelope=env.envelope, modulation=freq)

        ts = np.arange(480) / sample_rate

        # Compute fft and check that drag waveform is suppressed in a 40 MHz
        # window around the target frequency.
        ks, fs_drag = wave_drag.fft(ts, variables=dict(t0=100e-9))
        ks, fs_nodrag = wave_nodrag.fft(ts, variables=dict(t0=100e-9))

        window = (f0 + f1 - 20e6 < ks) & (ks < f0 + f1 + 20e6)

        assert (np.abs(fs_drag[window]) < np.abs(fs_nodrag[window])).all()

    def test_assign_channel(self):
        wave = DRAG(envelope=SquareWaveform())

        new_wave = wave.assign_channel("I_Q")
        assert new_wave.channel == "I_Q"
        assert new_wave.envelope.channel == "I_Q"


class TestVirtualZWaveform:
    @pytest.mark.parametrize(
        "frame,z_gates,phase_jumps",
        [
            (
                "mod_Q0",
                [
                    (0, VirtualZWaveform(frame="mod_Q0", phase=45)),
                    (0, VirtualZWaveform(frame="mod_Q0", phase=45)),
                    (1.5, VirtualZWaveform(frame="mod_Q1", phase=-90)),
                    (2, VirtualZWaveform(frame="mod_Q0", phase=-180)),
                ],
                [PhaseJump(0, 90), PhaseJump(2, -180)],
            ),
            (
                "mod_Q0 - mod_Q1",
                [
                    (0, VirtualZWaveform(frame="mod_Q0", phase=45)),
                    (0, VirtualZWaveform(frame="mod_Q0", phase=45)),
                    (1.5, VirtualZWaveform(frame="mod_Q1", phase=-90)),
                    (2, VirtualZWaveform(frame="mod_Q0", phase=-180)),
                ],
                [PhaseJump(0, 90), PhaseJump(1.5, 90), PhaseJump(2, -180)],
            ),
        ],
    )
    def test_update_phase_tracker(self, frame, z_gates, phase_jumps):
        phase_tracker = PhaseTracker.from_frames(["mod_Q0", "mod_Q1"])

        for t, z in z_gates:
            z.update_phase_tracker(t, phase_tracker)

        assert phase_tracker.compressed(frame) == phase_jumps


class TestPhaseResetWaveform:
    @pytest.mark.parametrize(
        "frame,resets,expected",
        [
            (
                Frame("Q0.mod"),
                [
                    (10, PhaseResetWaveform(frame="Q0.mod")),
                    (0, PhaseResetWaveform(frame="Q0.mod")),
                ],
                [10, 0],
            ),
            (
                Frame("Q1.mod"),
                [
                    (10, PhaseResetWaveform(frame="Q0.mod")),
                    (0, PhaseResetWaveform(frame="Q0.mod")),
                ],
                [],
            ),
        ],
    )
    def test_update_phase_tracker(self, frame, resets, expected):
        pt = PhaseTracker()

        for t, r in resets:
            r.update_phase_tracker(t, pt)

        assert pt.resets.get(frame, []) == expected


class TestTriggeredWaveform:
    @pytest.fixture
    def wave(self):
        wave = TriggeredWaveform(
            width="width",
            target=Timeline().add_waveform(SquareWaveform(amplitude="amp")),
        )
        return wave

    def test_equality_by_id(self):
        assert TriggeredWaveform(target=Timeline()) != TriggeredWaveform(
            target=Timeline()
        )

        w1 = TriggeredWaveform(target=Timeline())
        w2 = attrs.evolve(w1)

        assert w1 == w2

    def test_resolve(self, wave):
        resolved = wave.resolve(amp=0.5)

        tmln = wave.target.copy()
        tmln.resolve(amp=0.5)
        assert resolved.target == tmln

    def test_resolve_copy(self, wave):
        # No variables in target get resolved so no copy.
        w1 = wave.resolve(width=1)
        w2 = wave.resolve(width=1)
        assert w1 == w2

        w1 = wave.resolve(amp="amp")
        w2 = wave.resolve(amp="amp")
        assert w1 != w2
        assert w1.target == w2.target

    def test_variables(self, wave):
        assert wave.variables() == {"amp", "width"}
