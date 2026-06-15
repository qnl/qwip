"""Helpers for constructing measurement-conditioned reset operations."""

from qwip.sequencer.timeline import Timeline
from qwip.sequencer.waveform import ResetOperation, VirtualZWaveform


def _make_phase_neutral(pulse: Timeline) -> Timeline:
    """Append compensating virtual-Z markers so ``pulse`` nets zero z-phase.

    The conditional X pulse in active reset is applied on only one branch of a
    measurement-conditioned fork. Calibrated single-qubit gates carry virtual-Z
    phase updates (DRAG corrections, frame tracking on ``freq_01``/``freq_12``),
    so the branch that plays the pulse accumulates z-phase the fall-through
    branch does not. The downstream (distproc) compiler rejects conditional
    virtual-Z that isn't bound to a hardware register, raising a phase-mismatch
    error where the branches merge.

    The pulse only ever fires to drive a (residually excited) qubit back to
    |0>, after which the frame phase carries no information, so the correct fix
    is to make the pulse phase-neutral: for every virtual-Z it contains, add an
    equal-and-opposite marker at the end. Both branches then leave every frame
    at the same accumulated phase and the merge is consistent.
    """
    compensations = [
        (pulse.width, vz.evolve(phase=-vz.phase))
        for vz in pulse.operations
        if isinstance(vz, VirtualZWaveform)
    ]
    for loc, vz in compensations:
        pulse.add(vz, location=loc)
    return pulse


def active_reset(
    db,
    qubit: str,
    n_resets: int = 2,
    measure_first: bool = True,
    fproc_delay: float = 128e-9,
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
        fproc_delay: Per-round FPROC settling estimate, used *only* to budget the
            returned operation's ``width`` (the size of its slot in the parent
            timeline). The actual hardware wait is reserved at compile time by the
            FPROC Hold the backend inserts before each branch (keyed off the live
            readout end), so this value does not affect reset correctness; it only
            needs to be close enough that following operations are not laid out too
            early. Defaults to ~``fproc_meas_clks`` at the default 2 ns clock.

    Returns:
        A `ResetOperation` ready to drop into a `Timeline`.
    """
    meas = db.load_pulse(f"{qubit}_measure")
    x90 = db.load_pulse(f"{qubit}_X90")
    x180 = _make_phase_neutral(Timeline.from_layers([x90, x90]))

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
        width=n_resets * (meas.width + fproc_delay + x180.width),
    )


__all__ = ["active_reset"]
