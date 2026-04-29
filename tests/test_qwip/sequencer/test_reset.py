import pytest

from qwip.sequencer.reset import active_reset
from qwip.sequencer.timeline import Timeline
from qwip.sequencer.waveform import (
    REGISTERED_OPERATIONS,
    BasicWaveform,
    ResetOperation,
    SquareWaveform,
)


class FakeDB:
    """Minimal stand-in for a ConfigDB with `load_pulse(name)`."""

    def __init__(self, pulses: dict[str, Timeline]):
        self._pulses = pulses

    def load_pulse(self, name: str) -> Timeline:
        # Return a copy so callers can't mutate the source
        return self._pulses[name].copy()


def _make_pulse(channel: str, width: float = 30e-9) -> Timeline:
    tmln = Timeline()
    tmln.add(SquareWaveform(width=width, channel=channel))
    tmln.width = width
    return tmln


@pytest.fixture
def db():
    return FakeDB(
        {
            "Q0_X90": _make_pulse("Q0.qdrv", width=30e-9),
            "Q0_measure": _make_pulse("Q0.rdrv", width=2e-6).add(
                SquareWaveform(width=2e-6, channel="Q0.rdlo")
            ),
            "Q1_measure_no_rdlo": _make_pulse("Q1.rdrv", width=2e-6),
        }
    )


def test_reset_operation_is_registered():
    assert "ResetOperation" in REGISTERED_OPERATIONS
    assert REGISTERED_OPERATIONS["ResetOperation"] is ResetOperation


def test_reset_operation_construct():
    meas = _make_pulse("Q0.rdlo", 2e-6)
    x = _make_pulse("Q0.qdrv", 60e-9)
    op = ResetOperation(
        name="r",
        channel="Q0.rdlo",
        measurement=meas,
        x_pulse=x,
        n_resets=3,
        measure_first=False,
        width=3 * (meas.width + x.width),
    )
    assert op.measurement is meas
    assert op.x_pulse is x
    assert op.n_resets == 3
    assert op.measure_first is False
    assert op.channel == "Q0.rdlo"


def test_active_reset_helper_loads_pulses(db):
    op = active_reset(db, "Q0", n_resets=2)

    assert isinstance(op, ResetOperation)
    assert op.n_resets == 2
    assert op.measure_first is True
    assert op.channel == "Q0.rdlo"
    assert op.measurement is not None
    # X180 = two X90s composed, so the resulting Timeline width is twice X90's
    assert op.x_pulse.width == 2 * 30e-9
    # Total width should match n_resets * (meas + x180)
    assert op.width == 2 * (op.measurement.width + op.x_pulse.width)


def test_active_reset_defaults():
    pulses = {
        "Q0_measure": _make_pulse("Q0.rdrv").add(
            SquareWaveform(width=30e-9, channel="Q0.rdlo")
        ),
        "Q0_X90": _make_pulse("Q0.qdrv"),
    }
    op = active_reset(FakeDB(pulses), "Q0")
    assert op.n_resets == 2
    assert op.measure_first is True


def test_active_reset_raises_when_no_rdlo_channel(db):
    bad_db = FakeDB(
        {
            "Q1_measure": db.load_pulse("Q1_measure_no_rdlo"),
            "Q1_X90": _make_pulse("Q1.qdrv"),
        }
    )
    with pytest.raises(ValueError, match="rdlo channel"):
        active_reset(bad_db, "Q1")
