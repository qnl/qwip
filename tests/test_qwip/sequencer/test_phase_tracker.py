import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from qwip.sequencer.phase_tracker import Frame, PhaseJump, PhaseTracker
from qwip.testing import UnorderedList


class TestFrame:
    @pytest.mark.parametrize(
        "frame,phase,expect",
        [
            (Frame(), 10, {Frame(): 10}),
            (Frame("f0"), 90, {Frame("f0"): 90}),
            (Frame("f") + 5e9, 90, {Frame("f"): 90}),
            (-Frame("f"), 40, {Frame("f"): -40}),
            (Frame.from_string("Q0 + Q1"), 90, {Frame("Q0"): 45, Frame("Q1"): 45}),
            (
                Frame.from_string("0.5 * (Q0 - Q1)"),
                -90,
                {Frame("Q0"): -90, Frame("Q1"): 90},
            ),
        ],
    )
    def test_distribute_phase(self, frame, phase, expect):
        assert frame.distribute_phase(phase) == expect


class TestPhaseJump:
    @pytest.mark.parametrize(
        "a,b",
        [
            (PhaseJump(0, 1), PhaseJump(0, 2)),
            (PhaseJump(0, 1), PhaseJump(0.3, 2)),
            (PhaseJump(0.4, 1), PhaseJump(0.3, 2)),
            (PhaseJump(0, 1), PhaseJump(0, 0.5)),
            (PhaseJump(0, 1), PhaseJump(0.3, 0.5)),
            (PhaseJump(0.4, 1), PhaseJump(0.3, 0.5)),
        ],
    )
    def test_order(self, a, b):
        assert (a < b) == (a.t < b.t)
        assert (a > b) == (a.t > b.t)
        assert (a <= b) == (a.t <= b.t)
        assert (a >= b) == (a.t >= b.t)


class TestPhaseTracker:
    @pytest.fixture
    def phi_tracker(self):
        phi_tracker = PhaseTracker.from_frames(["mod_Q0", "mod_Q1"])

        phi_tracker.append("mod_Q0", PhaseJump(0, 1))
        phi_tracker.append("mod_Q0", PhaseJump(0, 0.5))
        phi_tracker.append("mod_Q1", PhaseJump(0, 1.5))
        phi_tracker.append("mod_Q1", PhaseJump(1.2, 2))
        phi_tracker.append("mod_Q0", PhaseJump(2, 1))
        return phi_tracker

    def test_from_frames(self):
        phi_tracker = PhaseTracker.from_frames(["a", "b", "a - b + 0.5 * c"])

        assert list(phi_tracker.keys()) == [
            Frame("a"),
            Frame("b"),
            Frame("c"),
        ]

    def test_getitem(self, phi_tracker):
        assert UnorderedList(phi_tracker["mod_Q0"]) == [
            PhaseJump(0, 0),
            PhaseJump(0, 1),
            PhaseJump(0, 0.5),
            PhaseJump(2, 1),
        ]

        assert UnorderedList(phi_tracker["0.5 * (mod_Q0 - mod_Q1)"]) == [
            PhaseJump(0, 0),
            PhaseJump(0, 0),
            PhaseJump(0, 0.5),
            PhaseJump(0, 0.25),
            PhaseJump(0, -0.75),
            PhaseJump(1.2, -1),
            PhaseJump(2, 0.5),
        ]

        assert phi_tracker["R0.mod"] == [PhaseJump(0, 0)]

    def test_contains(self):
        phi_tracker = PhaseTracker.from_frames(["a", "b", "c"])

        assert "a" in phi_tracker
        assert "a + b" in phi_tracker
        assert "0.5 * (a - b)" in phi_tracker
        assert "d" not in phi_tracker
        assert "a + d" not in phi_tracker

    def test_compress(self, phi_tracker):
        phis = PhaseTracker.compress(phi_tracker["0.5 * (mod_Q0 - mod_Q1)"])

        assert phis == [PhaseJump(0, 0), PhaseJump(1.2, -1), PhaseJump(2, 0.5)]

    def test_accumulate(self, phi_tracker):
        phis = phi_tracker.accumulated("0.5 * (mod_Q0 - mod_Q1)")

        assert phis == [PhaseJump(0, 0), PhaseJump(1.2, -1), PhaseJump(2, -0.5)]

    def test_reset(self):
        phi_tracker = PhaseTracker()

        phi_tracker.reset("Q0.mod", 0)
        assert phi_tracker.resets["Q0.mod"] == [0]

        phi_tracker.reset("Q0.mod", 10e-9)
        assert phi_tracker.resets["Q0.mod"] == [0, 10e-9]

    @pytest.mark.parametrize(
        "phase_jumps,resets,expect",
        [
            ([], [], np.array([0])),
            (
                [PhaseJump(0, 90), PhaseJump(25e-9, 90), PhaseJump(100e-9, 180)],
                [200e-9],
                np.array([90, 180, 360]),
            ),
            (
                [PhaseJump(0, 90), PhaseJump(25e-9, 90), PhaseJump(100e-9, 180)],
                [25e-9, 80e-9],
                np.array([90, 90, 180]),
            ),
            ([PhaseJump(5, 90), PhaseJump(25, 90)], [], np.array([0, 90, 180])),
        ],
    )
    def test_integrate_phase(self, phase_jumps, resets, expect):
        pt = PhaseTracker()

        t_expect = []
        for pj in phase_jumps:
            pt.append("Q0.mod", pj)
            t_expect.append(pj.t)

        if not t_expect:
            t_expect = [0]

        if t_expect[0] != 0:
            t_expect = [0] + t_expect

        for t in resets:
            pt.reset("Q0.mod", t)

        t_jump, accumulated_phase = pt.integrate_phase("Q0.mod")
        assert_array_equal(t_jump, np.unique(np.sort(t_expect)))
        assert_array_equal(accumulated_phase, expect)

    def test_integrate_phase_cache(self):
        pt = PhaseTracker()

        phase_jumps = [PhaseJump(0, 90), PhaseJump(25e-9, 90), PhaseJump(100e-9, 180)]

        for pj in phase_jumps:
            pt.append("Q0.mod", pj)
            pt.append("Q1.mod", pj)

        pt.integrate_phase("Q0.mod")
        assert pt.integrate_phase.cache_info().hits == 0
        pt.integrate_phase("Q0.mod")
        assert pt.integrate_phase.cache_info().hits == 1
        pt.integrate_phase("Q1.mod")
        assert pt.integrate_phase.cache_info().hits == 1

        pt.append("Q0.mod", PhaseJump(0, 180))
        pt.integrate_phase("Q0.mod")
        assert pt.integrate_phase.cache_info().hits == 0

    @pytest.mark.parametrize(
        "ts,phase_jumps,expected",
        [
            (np.arange(10), [], np.zeros(10)),
            (
                np.arange(10),
                [PhaseJump(0, 0), PhaseJump(4.5, 1)],
                np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            ),
            (
                np.arange(10) / 2,
                [PhaseJump(0, 0), PhaseJump(3, 1)],
                np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
            ),
            (np.arange(10) * 5, [PhaseJump(0, 0), PhaseJump(51, 1)], np.zeros(10)),
            (
                np.arange(5),
                [PhaseJump(0, 0), PhaseJump(4, 1)],
                np.array([0, 0, 0, 0, 1]),
            ),
            (
                np.arange(10),
                [
                    PhaseJump(0, 0),
                    PhaseJump(1.9, 1),
                    PhaseJump(4.7, -2),
                    PhaseJump(20, -1),
                ],
                np.array([0, 0, 1, 1, 1, -1, -1, -1, -1, -1]),
            ),
            (
                np.arange(10) + 10,
                [
                    PhaseJump(0, 0),
                    PhaseJump(9, 1),
                    PhaseJump(10, -2),
                    PhaseJump(15, 2),
                    PhaseJump(19, -3),
                ],
                np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, -2]),
            ),
            (
                np.arange(10) + 10,
                [
                    PhaseJump(0, 0),
                    PhaseJump(9, 1),
                    PhaseJump(14.5, -2),
                    PhaseJump(20, -1),
                ],
                np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]),
            ),
            (
                np.arange(10) + 10,
                [PhaseJump(0, 0), PhaseJump(9, 1), PhaseJump(20, -3)],
                np.ones(10),
            ),
            (
                np.arange(10),
                [PhaseJump(5, 1), PhaseJump(20, -3)],
                np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            ),
        ],
    )
    def test_compute_integrated_phase(self, ts, phase_jumps, expected):
        phase_tracker = PhaseTracker.from_frames(["Q0"])

        for pj in phase_jumps:
            phase_tracker.append("Q0", pj)

        integrated_phase = phase_tracker.compute_integrated_phase("Q0", ts)
        assert_almost_equal(integrated_phase, expected)

        mod = Frame(0)
        integrated_phase = phase_tracker.compute_integrated_phase(mod, ts)
        assert_almost_equal(integrated_phase, np.zeros_like(expected))

    @pytest.mark.parametrize(
        "ts,frame,pj1,pj2,expected",
        [
            (
                np.arange(10),
                0.5 * (Frame("Q0") - Frame("Q1")),
                [PhaseJump(5, 90), PhaseJump(10, 90)],
                [],
                np.array([0, 0, 0, 0, 0, 45, 45, 45, 45, 45]),
            ),
            (
                np.arange(10),
                0.5 * (Frame("Q0") - Frame("Q1")),
                [],
                [PhaseJump(5, 90), PhaseJump(10, 90)],
                np.array([0, 0, 0, 0, 0, -45, -45, -45, -45, -45]),
            ),
            (
                np.arange(10),
                0.5 * (Frame("Q0") - Frame("Q1")),
                [PhaseJump(0, 180), PhaseJump(5, -90)],
                [PhaseJump(3, 90), PhaseJump(9, 90)],
                np.array([90, 90, 90, 45, 45, 0, 0, 0, 0, -45]),
            ),
        ],
    )
    def test_compute_integrated_phase_dependent(self, ts, frame, pj1, pj2, expected):
        phase_tracker = PhaseTracker.from_frames(["Q0", "Q1"])

        for pj in pj1:
            phase_tracker.append("Q0", pj)

        for pj in pj2:
            phase_tracker.append("Q1", pj)

        integrated_phase = phase_tracker.compute_integrated_phase(frame, ts)
        assert_almost_equal(integrated_phase, expected)

    @pytest.mark.parametrize(
        "resets,ts,frequency,expected",
        [
            ([], np.r_[2:20] / 1e9, 100e6, 2 * np.pi * 100e6 * np.r_[2:20] / 1e9),
            (
                [-2e-9],
                np.r_[2:20] / 1e9,
                100e6,
                2 * np.pi * 100e6 * (np.r_[2:20] + 2) / 1e9,
            ),
            (
                [3e-9, 5e-9],
                np.arange(10) / 1e9,
                1e9 / (2 * np.pi),
                np.array([0, 1, 2, 0, 1, 0, 1, 2, 3, 4]),
            ),
            (
                [2.5e-9, 20e-9],
                np.arange(5) / 1e9,
                1e9 / (2 * np.pi),
                np.array([0, 1, 2, 0.5, 1.5]),
            ),
        ],
    )
    def test_compute_oscillator_phase(self, resets, ts, frequency, expected):
        pt = PhaseTracker(resets=dict(Q0=resets))

        frames = dict(Q0=Frame(frequency))
        phis = pt.compute_oscillator_phase(Frame("Q0"), ts, frames)

        assert_almost_equal(phis, expected)

    @pytest.mark.parametrize(
        "ts,frame,detuning,expected",
        [
            (
                ts := np.linspace(0, 10, 11),
                0.5 * (Frame("Q0") - Frame("Q1")),
                Frame(0.2),
                ts * -0.3,
            ),
            (
                ts := np.linspace(10, 20, 11),
                0.5 * (Frame("Q0") - Frame("Q1")),
                Frame(0.2),
                ts[0] * -0.5 + (ts - ts[0]) * -0.3,
            ),
        ],
    )
    def test_dependent_compute_oscillator_phase(self, ts, frame, detuning, expected):
        pt = PhaseTracker()
        frames = dict(Q0=Frame(1), Q1=Frame(2))

        phis = pt.compute_oscillator_phase(frame, ts, frames, detuning) / (2 * np.pi)

        assert_almost_equal(phis, expected)
