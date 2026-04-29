"""Helpers for constructing measurement-conditioned reset operations."""

from qwip.sequencer.timeline import Timeline
from qwip.sequencer.waveform import ResetOperation


def active_reset(
    db,
    qubit: str,
    n_resets: int = 2,
    measure_first: bool = True,
) -> ResetOperation:
    """Build a `ResetOperation` for ``qubit`` from calibrated pulses in ``db``.

    Loads ``{qubit}_measure`` and ``{qubit}_X90`` from the database and composes
    two X90s into an X180 (no native X180 calibration exists). The readout
    acquisition channel is auto-detected from the measurement timeline by
    matching the ``{qubit}.rdlo`` naming convention used by the Qubic backend.

    Args:
        db: A `ConfigDB` (or anything implementing ``load_pulse(name) -> Timeline``).
        qubit: Qubit name, e.g. ``"Q1"``.
        n_resets: Number of measurement+conditional-X rounds. Each round multiplicatively
            suppresses residual excited-state population.
        measure_first: If False, the first round skips its measurement and reuses the
            discrimination result from a measurement that occurred earlier in the parent
            timeline. Useful when chaining reset onto an existing end-of-circuit readout.

    Returns:
        A `ResetOperation` ready to drop into a `Timeline`.
    """
    meas = db.load_pulse(f"{qubit}_measure")
    x90 = db.load_pulse(f"{qubit}_X90")
    x180 = Timeline.from_layers([x90, x90])

    rdlo_channels = [c for c in meas.channels if c.endswith(".rdlo")]
    if len(rdlo_channels) != 1:
        raise ValueError(
            f"Expected exactly one .rdlo channel in {qubit}_measure, "
            f"found {sorted(rdlo_channels)}"
        )
    rdlo = rdlo_channels[0]

    return ResetOperation(
        name=f"{qubit}_reset",
        measurement=meas,
        x_pulse=x180,
        n_resets=n_resets,
        measure_first=measure_first,
        channel=rdlo,
        width=n_resets * (meas.width + x180.width),
    )


__all__ = ["active_reset"]
