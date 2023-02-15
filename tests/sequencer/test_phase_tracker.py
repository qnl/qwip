import pytest

from qwip.sequencer.phase_tracker import (
    ModulationFrequency,
    PhaseJump,
    PhaseTracker
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
        assert phi_tracker["mod_Q0"] == [
            PhaseJump(0, 1),
            PhaseJump(0, 0.5),
            PhaseJump(2, 1)
        ]

        assert phi_tracker['0.5 * (mod_Q0 - mod_Q1)'] == [
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


