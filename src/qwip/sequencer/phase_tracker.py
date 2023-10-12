import itertools as it
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Sequence as TSequence
from typing import Protocol, runtime_checkable

import numpy as np
from attrs import evolve, field
from typing_extensions import Self

from qwip.attrs import qdefine, qfrozen
from qwip.sequencer.utils import LinearExpression


@qfrozen(kw_only=False, repr=False)
class ModulationFrequency(LinearExpression):
    ...


@qfrozen(kw_only=False, order=True)
class PhaseJump:
    """A discrete phase jump.

    A phase jump specifies a time t and a phase phi.
    """

    t: float
    phi: float = field(order=False)

    def __add__(self, other):
        if self.t != other.t:
            raise ValueError(f"Cannot add two phases at different times {t}")

        return evolve(self, phi=self.phi + other.phi)

    def __radd__(self, other):
        return evolve(self, phi=self.phi + other)


@qdefine
class PhaseTracker:
    """A data structure for maintaining a set of phase jumps.

    Each entry in the phase tracker references a relative phase between two states
    in the Hilbert space. Phases are indexed by a unique ModulationFrequency that
    also specifies the frequency at which the particular phase evolves in the
    absence of any discrete phase jumps.

    Attributes:
        phases: A dictionary mapping ModulationFrequency to a list of phase jumps.
        resets: A mapping of reference frames to a list of times at which the phase
            should get reset.
    """

    phases: dict[ModulationFrequency, list[PhaseJump]] = field(factory=dict)
    resets: dict[ModulationFrequency, list[float]] = field(factory=dict)

    @classmethod
    def from_modulations(
        cls, modulations: TSequence[ModulationFrequency | str]
    ) -> Self:
        """Creates an entry in the phase dictionary for each base modulation."""
        phases = defaultdict(list)

        for mod in modulations:
            if isinstance(mod, str):
                mod = ModulationFrequency.from_string(mod)

            if mod.variables():
                for v in mod.variables():
                    phases[v]
            else:
                phases[mod]

        return cls(phases=phases)

    def keys(self):
        return self.phases.keys()

    def __getitem__(self, val):
        """Gets the list of phase jumps associated with the phase entry."""
        if isinstance(val, str):
            val = ModulationFrequency.from_string(val)

        if not val.references:
            try:
                return self.phases[val]
            except KeyError:
                phases = self.phases[val] = []
                return phases

        phases = []
        for modkey, c in val.references:
            phases.append(tuple(evolve(pt, phi=pt.phi * c) for pt in self[modkey]))

        return sorted(it.chain.from_iterable(phases))

    def __contains__(self, val):
        """Returns True if all references are in the phase dictionary."""
        if isinstance(val, str):
            val = ModulationFrequency.from_string(val)

        deps = val.variables()
        if val.offset or not val.variables():
            deps.add(ModulationFrequency(val.offset))

        return all(v in self.phases for v in deps)

    def append(self, modkey: ModulationFrequency | str, phase: PhaseJump):
        """Adds a phase jump."""
        if isinstance(modkey, str):
            modkey = ModulationFrequency.from_string(modkey)

        if modkey.references:
            raise ValueError(
                f"Cannot add a virtual phase on a non-independent phase {modkey}"
            )

        self[modkey].append(phase)

    def reset(self, modkey: ModulationFrequency, time: float):
        """Adds a phase reset."""

        try:
            self.resets[modkey].append(time)
        except KeyError:
            self.resets[modkey] = [time]

    @staticmethod
    def compress(phases: Iterable[PhaseJump]) -> list[PhaseJump]:
        """Returns a list of phases with a single entry per timepoint."""
        return [sum(tphis) for _, tphis in it.groupby(phases, key=lambda pt: pt.t)]

    def compressed(self, modkey: ModulationFrequency) -> list[PhaseJump]:
        return type(self).compress(self[modkey])

    def accumulated(self, modkey: ModulationFrequency) -> list[PhaseJump]:
        phis = type(self).compress(self[modkey])

        return list(
            it.accumulate(
                phis, func=lambda pj1, pj2: evolve(pj2, phi=pj1.phi + pj2.phi)
            )
        )

    def integrate_phase(
        self,
        modkey: ModulationFrequency,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the total accumulated phase from phase jumps.

        Args:
            modkey: The reference frame on which to compute the phase accumulation.

        Returns:
            The timestep of each phase jump, and the resulting cumulative phase after
            each phase jump.
        """
        phase_jumps = self.compressed(modkey)
        if phase_jumps:
            t_jump, phase_jumps = np.array([(pj.t, pj.phi) for pj in phase_jumps]).T
        else:
            t_jump = phase_jumps = np.zeros(1)

        # Get accumulated phase accounting for phase resets
        t_resets = np.sort(self.resets.get(modkey, []))
        acc_idx = np.r_[0, np.searchsorted(t_jump, t_resets), len(t_jump)]
        acc_idx = np.unique(acc_idx)
        accumulated_phase = np.zeros_like(phase_jumps)
        for (
            s,
            e,
        ) in zip(acc_idx, acc_idx[1:]):
            accumulated_phase[s:e] = np.cumsum(phase_jumps[s:e])

        return t_jump, accumulated_phase

    def compute_integrated_phase(
        self,
        modkey: ModulationFrequency,
        ts: np.ndarray,
    ) -> np.ndarray:
        """Computes the jump phases for a set of timepoints.

        Args:
            modkey: The reference frame for the phase jumps.
            ts: The timepoints at which to evaluate the phases.

        Returns:
            The jump phase at each time point.
        """
        if modkey not in self:
            return np.zeros_like(ts)

        t_jump, accumulated_phase = self.integrate_phase(modkey)

        # Find phase_jumps that are relevant for the time slice
        s = np.searchsorted(t_jump, ts[0])
        e = np.searchsorted(t_jump, ts[-1], side="right")

        idx = np.searchsorted(ts, t_jump[s:e])

        # Set phis equal to last phase before or equal to ts[0]
        phis = accumulated_phase[max(s - 1, 0)] * np.ones_like(ts)

        N = phis.shape[0]

        for i, (left, right) in enumerate(zip(idx, idx[1:])):
            phis[left:right] = accumulated_phase[s + i]

            if right >= N:
                break
        else:
            # Handle any remaining bit
            if len(idx):
                phis[idx[-1] :] = accumulated_phase[s + len(idx) - 1]

        return phis

    def compute_oscillator_phase(
        self,
        modkey: ModulationFrequency,
        ts: np.ndarray,
        modulations: dict[str, ModulationFrequency] = {},
    ) -> np.ndarray:
        """Computes the phase on a reference frame due to time evolution.

        When there are no specified phase resets before `t=0`, it is assumed that the
        time evolution begins at `t=0`. However, if there is a phase reset for some
        `t < 0`, the phase at `t=0` will no longer be zero unless another phase reset
        at `t=0` is explicitly added. Note that a phase reset at time `t` will only
        affect timepoints greater than or equal to `t`.

        Args:
            modkey: The reference frame to evaluate.
            ts: The timepoints at which to compute the time-evolved phase.
            modulations: A map from reference frame names to concrete frequencies.

        Returns:
            The time evolved phase at the specified timepoints. This phase is assumed
        """
        freq = modkey.resolve(**modulations).offset

        t_resets = np.sort(np.unique(self.resets.get(modkey, [])))
        diffs = np.diff(np.r_[0, t_resets])
        adjusted_ts = ts - sum(d * (ts >= t_r) for t_r, d in zip(t_resets, diffs))

        return 2 * np.pi * freq * adjusted_ts


@runtime_checkable
class PhaseUpdater(Protocol):
    def update_phase_tracker(
        self,
        time: float,
        phase_tracker: dict[ModulationFrequency, list[tuple[float, float]]],
    ) -> None:
        ...


__all__ = [
    "ModulationFrequency",
    "PhaseJump",
    "PhaseTracker",
    "PhaseUpdater",
]
