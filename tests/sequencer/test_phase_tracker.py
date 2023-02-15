import pytest
import numpy as np
from numpy.testing import assert_allclose

from qwip.testing import UnorderedList
from qwip.sequencer.phase_tracker import (
    ModulationFrequency,
    PhaseJump,
    PhaseTracker,
)

class TestPhaseJump:
    @pytest.mark.parametrize(
        'a,b',
        [
            (PhaseJump(0, 1), PhaseJump(0, 2)),
            (PhaseJump(0, 1), PhaseJump(0.3, 2)),
            (PhaseJump(0.4, 1), PhaseJump(0.3, 2)),
            (PhaseJump(0, 1), PhaseJump(0, 0.5)),
            (PhaseJump(0, 1), PhaseJump(0.3, 0.5)),
            (PhaseJump(0.4, 1), PhaseJump(0.3, 0.5))
        ]
    )
    def test_order(self, a, b):
        assert (a < b) == (a.t < b.t)
        assert (a > b) == (a.t > b.t)
        assert (a <= b) == (a.t <= b.t)
        assert (a >= b) == (a.t >= b.t)


class TestPhaseTracker:
    @pytest.fixture
    def phi_tracker(self):
        phi_tracker = PhaseTracker.from_modulations(['mod_Q0', 'mod_Q1'])

        phi_tracker.append("mod_Q0", PhaseJump(0, 1))
        phi_tracker.append("mod_Q0", PhaseJump(0, 0.5))
        phi_tracker.append("mod_Q1", PhaseJump(0, 1.5))
        phi_tracker.append("mod_Q1", PhaseJump(1.2, 2))
        phi_tracker.append("mod_Q0", PhaseJump(2, 1))
        return phi_tracker

    def test_from_modulations(self):
        phi_tracker = PhaseTracker.from_modulations(["a", "b", "a - b + 0.5 * c"])

        assert list(phi_tracker.keys()) == [
            ModulationFrequency("a"),
            ModulationFrequency("b"),
            ModulationFrequency("c")
        ]

    def test_getitem(self, phi_tracker):
        assert UnorderedList(phi_tracker["mod_Q0"]) == [
            PhaseJump(0, 1),
            PhaseJump(0, 0.5),
            PhaseJump(2, 1)
        ]

        assert UnorderedList(phi_tracker['0.5 * (mod_Q0 - mod_Q1)']) == [
            PhaseJump(0, 0.5),
            PhaseJump(0, 0.25),
            PhaseJump(0, -0.75),
            PhaseJump(1.2, -1),
            PhaseJump(2, 0.5)
        ]

    def test_contains(self):
        phi_tracker = PhaseTracker.from_modulations(['a', 'b', 'c'])

        assert 'a' in phi_tracker
        assert 'a + b' in phi_tracker
        assert '0.5 * (a - b)' in phi_tracker
        assert 'd' not in phi_tracker
        assert 'a + d' not in phi_tracker

    def test_compress(self, phi_tracker):
        phis = PhaseTracker.compress(phi_tracker["0.5 * (mod_Q0 - mod_Q1)"])

        assert phis == [
            PhaseJump(0, 0),
            PhaseJump(1.2, -1),
            PhaseJump(2, 0.5)
        ]

    def test_accumulate(self, phi_tracker):
        phis = phi_tracker.accumulated("0.5 * (mod_Q0 - mod_Q1)")

        assert phis == [
            PhaseJump(0, 0),
            PhaseJump(1.2, -1),
            PhaseJump(2, -0.5)
        ]

    @pytest.mark.parametrize(
        'ts,phase_jumps,expected',
        [
            (
                np.arange(10),
                [],
                np.zeros(10)
            ),
            (
                np.arange(10),
                [PhaseJump(0, 0), PhaseJump(4.5, 1)],
                np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
            ),
            (
                np.arange(10) / 2,
                [PhaseJump(0, 0), PhaseJump(3, 1)],
                np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
            ),
            (
                np.arange(10) * 5,
                [PhaseJump(0, 0), PhaseJump(51, 1)],
                np.zeros(10)
            ),
            (
                np.arange(5),
                [PhaseJump(0, 0), PhaseJump(4, 1)],
                np.array([0, 0, 0, 0, 1])
            ),
            (
                np.arange(10),
                [
                    PhaseJump(0, 0),
                    PhaseJump(1.9, 1),
                    PhaseJump(4.7, -2),
                    PhaseJump(20, -1)
                ],
                np.array([0, 0, 1, 1, 1, -1, -1, -1, -1, -1])
            ),
            (
                np.arange(10) + 10,
                [
                    PhaseJump(0, 0),
                    PhaseJump(9, 1),
                    PhaseJump(10, -2),
                    PhaseJump(15, 2),
                    PhaseJump(19, -3)
                ],
                np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, -2])
            ),
            (
                np.arange(10) + 10,
                [
                    PhaseJump(0, 0),
                    PhaseJump(9, 1),
                    PhaseJump(14.5, -2),
                    PhaseJump(20, -1)
                ],
                np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
            ),
            (
                np.arange(10) + 10,
                [
                    PhaseJump(0, 0),
                    PhaseJump(9, 1),
                    PhaseJump(20, -3)
                ],
                np.ones(10)
            )
        ]
    )
    def test_integrated_phase(self, ts, phase_jumps, expected):
        phase_tracker = PhaseTracker.from_modulations(['Q0'])

        for pj in phase_jumps:
            phase_tracker.append('Q0', pj)

        integrated_phase = phase_tracker.compute_integrated_phase('Q0', ts)

        assert_allclose(integrated_phase, expected)

