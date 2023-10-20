from copy import copy, deepcopy

import attrs
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

import qwip
from qwip.sequencer.elements import Timeline
from qwip.sequencer.phase_tracker import Frame, PhaseJump, PhaseTracker
from qwip.sequencer.utils import Location
from qwip.sequencer.waveform import (
    DRAG,
    BasicWaveform,
    CWWaveform,
    GaussianWaveform,
    ModulatedWaveform,
    PhaseResetWaveform,
    SquareWaveform,
    TriggeredWaveform,
    VirtualZWaveform,
    Waveform,
    update_fields,
)


class TestWaveform:
    @pytest.mark.parametrize("wave", [Waveform(), Waveform(name="wave")])
    def test_copy(self, wave):
        assert copy(wave) is wave
        assert deepcopy(wave) is wave

    @pytest.mark.parametrize(
        "wave,updates,expect",
        [
            (
                SquareWaveform(),
                dict(width=10),
                SquareWaveform(width=10),
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


class TestBasicWaveform:
    @pytest.mark.parametrize("name,expect", [(None, "BasicWaveform"), ("name", "name")])
    def test_name(self, name, expect):
        w = BasicWaveform(name=name) if name else BasicWaveform()
        assert w.name == expect

    def test_create(self):
        w = BasicWaveform()

        assert w.name == "BasicWaveform"
        assert w.channels == tuple()
        assert w.width == Location()
        assert w.amplitude == 1
        assert w.t0 == 0

        w = BasicWaveform(name="name")
        assert w.name == "name"

    def test_convert(self):
        w = BasicWaveform(channels=(0, "Q1"), width=1, amplitude="A")

        assert w.channels == ("0", "Q1")
        assert w.width == Location(1)
        assert w.amplitude == "A"

    @pytest.mark.parametrize(
        "kwargs,expect",
        [
            (dict(), dict(width="w", amplitude="amp", t0=0)),
            (dict(w=10, amp=20), dict(width=10.0, amplitude=20, t0=0, w=10)),
            (dict(width=10, amplitude=20, t0=1), dict(width=10.0, amplitude=20, t0=1)),
            (dict(random=10), dict(width="w", amplitude="amp", t0=0, random=10)),
        ],
    )
    def test_update_fields(self, kwargs, expect):
        w = BasicWaveform(width="w", amplitude="amp")

        assert update_fields(w, **kwargs) == expect

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
                BasicWaveform(amplitude="BasicWaveform"),
                dict(BasicWaveform=0.2),
                BasicWaveform(amplitude=0.2),
            ),
        ],
    )
    def test_resolve(self, wave, vmap, new):
        assert wave.resolve(**vmap) == new


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

        w = CWWaveform(frequency=Frame(0.2), channels=("I", "Q"))

        phase_tracker = PhaseTracker(
            phases={Frame(0.2): [PhaseJump(t, pj) for t, pj in phase_jumps]}
        )

        wave = w(ts, phase_tracker=phase_tracker, phase_unit="degrees")
        assert_allclose(wave, expected, atol=1e-7)

        w_single_channel = w.evolve(channels=("I",))
        wave = w_single_channel(ts, phase_tracker=phase_tracker, phase_unit="degrees")
        assert_allclose(wave, expected[0], atol=1e-7)

        w_three_channel = w.evolve(channels=("a", "b", "c"))
        wave = w_three_channel(ts, phase_tracker=phase_tracker, phase_unit="degrees")
        assert_allclose(wave, np.stack([expected[0] for i in range(3)]), atol=1e-7)

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
            channels=("Q0.drv",),
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

        w = CWWaveform(frequency=mod_freq, channels=("I", "Q"))

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
                dict(t0=1, width=Location(10), amplitude=2),
            ),
            (
                dict(width=20, amplitude=2),
                dict(amplitude=0.5),
                dict(t0=0, width=Location(20), amplitude=1),
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
        mod = CWWaveform(frequency=freq, channels=("I", "Q"))

        phase_tracker = PhaseTracker(
            phases={freq: [PhaseJump(t, pj) for t, pj in phase_jumps]}
        )

        w = ModulatedWaveform(envelope=env, modulation=mod)

        ts = np.arange(240) / 2.4e9
        wave = w(ts, t0=40e-9, phase_tracker=phase_tracker)

        assert_allclose(wave, expected, atol=1e-7)

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
                    modulation=CWWaveform(frequency="f", channels=("I", "Q")),
                ),
                dict(
                    name="X90",
                    envelope=dict(
                        width=4e-8, amplitude=0.5, __class__="GaussianWaveform"
                    ),
                    modulation=dict(
                        channels=["I", "Q"],
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


class TestDRAGWaveform:
    def test_suppression(self):
        f0 = 500e6
        f1 = -100e6
        env = DRAG(envelope=GaussianWaveform(width=20e-9), lmbda=1 / (2 * np.pi * f1))

        freq = CWWaveform(frequency=Frame(f0), channels=("I", "Q"))

        wave_drag = ModulatedWaveform(envelope=env, modulation=freq)
        wave_nodrag = ModulatedWaveform(envelope=env.envelope, modulation=freq)

        ts = np.arange(480) / 2.4e9

        # Compute fft and check that drag waveform is suppressed in a 40 MHz
        # window around the target frequency.
        ks, fs_drag = wave_drag.fft(ts, t0=100e-9)
        ks, fs_nodrag = wave_nodrag.fft(ts, t0=100e-9)

        window = (f0 + f1 - 20e6 < ks) & (ks < f0 + f1 + 20e6)

        assert (np.abs(fs_drag[window]) < np.abs(fs_nodrag[window])).all()


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

        se = wave.target.copy()
        se.resolve_waveforms(amp=0.5)
        assert resolved.target == se

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
