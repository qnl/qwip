"""Helpers for constructing measurement-conditioned reset operations."""

import numpy as np
from loguru import logger

from qwip.sequencer.timeline import Timeline
from qwip.sequencer.waveform import ModulatedWaveform, ResetOperation


def _classification_means(db, qubit: str, readout_config: str) -> tuple[np.ndarray, np.ndarray]:
    """(ground, excited) blob means from the stored readout classification."""
    cls = db.config["readout"][readout_config]["registers"][qubit]["classification"]
    means = np.asarray(cls["means"], dtype=float)
    if means.shape[0] < 2:
        raise ValueError(f"{qubit} readout classification has fewer than two states")
    return means[0], means[1]


def _rotate_iq(points: np.ndarray, degrees: float) -> np.ndarray:
    """Rotates I/Q points (last axis) counter-clockwise by ``degrees``."""
    r = np.deg2rad(degrees)
    rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return np.asarray(points) @ rot.T


def hardware_demod_rotation(db, qubit: str, readout_config: str = "default") -> float:
    """Demod phase change, in degrees, that aligns the readout blobs with the
    hardware threshold.

    Qubic's measurement-conditioned branch reads one bit per readout channel:
    1 when the accumulated I value is below the channel's threshold register
    (see `hardware_state_threshold`). The I/Q plane is rotated so that the
    vector from the calibrated ground blob to the excited blob points along
    negative I; the decision line is then perpendicular to the separation and
    the threshold at the midpoint gives both states the largest margin. A demod
    phase change of +phi rotates the I/Q plane by -phi.

    Args:
        db: A `ConfigDB`.
        qubit: Qubit name, e.g. ``"Q1"``.
        readout_config: Name of the readout configuration holding the
            classification means.

    Returns:
        The change to add to the measurement's demod phase, in degrees.
    """
    ground, excited = _classification_means(db, qubit, readout_config)
    sep = excited - ground
    rotation = 180.0 - np.degrees(np.arctan2(sep[1], sep[0]))
    return -rotation


def hardware_state_threshold(
    db, qubit: str, readout_config: str = "default", demod_rotation: float | None = None
) -> int:
    """Threshold for the readout board's state-discrimination register.

    The register holds the I value, in accumulator units, that separates the
    two states as seen at the reset measurement's demod phase. This is the
    midpoint of the stored ground and excited blob means after the plane
    rotation implied by ``demod_rotation`` (the same convention as
    `hardware_demod_rotation`, which supplies the default).

    Args:
        db: A `ConfigDB`.
        qubit: Qubit name, e.g. ``"Q1"``.
        readout_config: Name of the readout configuration holding the
            classification means.
        demod_rotation: Demod phase change, in degrees, applied to the reset
            measurements. ``None`` uses `hardware_demod_rotation`.

    Returns:
        The threshold as an integer in accumulator units.
    """
    ground, excited = _classification_means(db, qubit, readout_config)
    if demod_rotation is None:
        demod_rotation = hardware_demod_rotation(db, qubit, readout_config)
    g, e = _rotate_iq(np.stack([ground, excited]), -demod_rotation)
    if e[0] >= g[0]:
        logger.warning(
            f"{qubit}: the excited blob is not at lower I than the ground blob at "
            f"demod rotation {demod_rotation:.1f} deg; the hardware state bit will "
            f"not mark |1>."
        )
    return int(round((g[0] + e[0]) / 2))


def _rotate_demod(meas: Timeline, channel: str, degrees: float) -> None:
    """Adds ``degrees`` to the modulation phase of ``channel``'s waveforms."""
    for i, (loc, op) in enumerate(meas.lw_pairs):
        if isinstance(op, ModulatedWaveform) and op.channel == channel:
            mod = op.modulation
            meas.lw_pairs[i] = (
                loc,
                op.evolve(modulation=mod.evolve(phase=mod.phase + degrees)),
            )


def active_reset(
    db,
    qubit: str,
    n_resets: int = 2,
    measure_first: bool = True,
    fproc_delay: float = 400e-9,
    measurement_variables: dict | None = None,
    readout_config: str = "default",
    demod_rotation: float | None = None,
    threshold: int | None = None,
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
        fproc_delay: Gap between each measurement and its conditional pi pulse,
            measured from the end of the measurement's ``rdlo`` pulse (which may
            extend past the measurement timeline's declared ``width``). The Qubic
            backend holds the processor and reads the state bit just before the
            pi pulse, so this is the time the measurement result has to reach the
            processor (on multi-board systems, over the inter-board link). The
            floor is ``fproc_meas_clks + jump_fproc_clks``, 72 clocks (144 ns) at
            the default 2 ns clock; compilation raises a `ValueError` below it.
            The default leaves a few hundred nanoseconds for the result to
            cross the inter-board link.
        measurement_variables: Variable overrides passed to ``db.load_pulse``
            for the reset's own measurements (the final readout of a circuit is
            not affected).
        readout_config: Readout configuration whose classification means give
            the demod rotation, see `hardware_demod_rotation`.
        demod_rotation: Degrees added to the demod phase of the reset's own
            measurements. ``None`` computes it with `hardware_demod_rotation`
            from the stored readout calibration (skipped, with a warning, if
            ``db`` has no such configuration); ``0`` keeps the stored phase.
        threshold: Value for the readout board's state-discrimination register
            (see `hardware_state_threshold`), in accumulator units. ``None``
            computes it from the stored classification at the demod rotation in
            use; the Qubic backend then writes it with every program containing
            this reset. Without a stored classification no register is written.

    Returns:
        A `ResetOperation` ready to drop into a `Timeline`.
    """
    if measurement_variables:
        meas = db.load_pulse(f"{qubit}_measure", variables=measurement_variables)
    else:
        meas = db.load_pulse(f"{qubit}_measure")
    x90 = db.load_pulse(f"{qubit}_X90")
    # A calibrated pulse's declared width can be shorter than its content (a
    # measurement's demod window usually outlasts its drive). Resolving makes
    # `width` cover the content, so the rounds and the conditional pi pulse are
    # laid out after the demod window rather than after the drive.
    meas.resolve()
    x90.resolve()
    # The X180's virtual-Z markers are folded into its pulse phases when the
    # backend compiles the branch arm, so the reset leaves every frame's phase
    # as it found it (the qubit is being returned to |0>, so the frame phase
    # carries no information) and the two arms of the branch stay consistent.
    x180 = Timeline.from_layers([x90, x90])

    rdlo_channels = [c for c in meas.channels if c.endswith(".rdlo")]
    if len(rdlo_channels) != 1:
        raise ValueError(
            f"Expected exactly one .rdlo channel in {qubit}_measure, "
            f"found {sorted(rdlo_channels)}"
        )
    rdlo = rdlo_channels[0]

    if demod_rotation is None or threshold is None:
        try:
            if demod_rotation is None:
                demod_rotation = hardware_demod_rotation(db, qubit, readout_config)
            if threshold is None:
                threshold = hardware_state_threshold(
                    db, qubit, readout_config, demod_rotation
                )
        except (AttributeError, KeyError, TypeError) as exc:
            logger.warning(
                f"No readout classification for {qubit} in readout config "
                f"{readout_config!r} ({exc!r}); the reset measurements keep the "
                f"stored demod phase and no threshold is written, so the "
                f"hardware state bit may not mark |1>."
            )
            demod_rotation = demod_rotation or 0.0
    if demod_rotation:
        _rotate_demod(meas, rdlo, demod_rotation)

    return ResetOperation(
        name=f"{qubit}_reset",
        measurement=meas,
        x_pulse=x180,
        n_resets=n_resets,
        measure_first=measure_first,
        channel=rdlo,
        fproc_delay=fproc_delay,
        threshold=threshold,
        width=n_resets * (meas.width + fproc_delay + x180.width),
    )


__all__ = ["active_reset", "hardware_demod_rotation", "hardware_state_threshold"]
