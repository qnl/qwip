import itertools as it

from collections.abc import Iterable
from collections.abc import Sequence as TSequence
from collections import defaultdict
from typing import Protocol, runtime_checkable
from typing_extensions import Self

from attrs import field, evolve

from qwip.settings.settings import qdefine, qfrozen
from qwip.sequencer.utils import LinearExpression

@qfrozen(kw_only=False, repr=False)
class ModulationFrequency(LinearExpression):
    ...


@qfrozen(kw_only=False, order=True)
class PhaseJump:
    """"""
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
    """
    phases: dict[ModulationFrequency, list[PhaseJump]] = field(factory=dict)

    @classmethod
    def from_modulations(
        cls,
        modulations: TSequence[ModulationFrequency | str]
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
            return self.phases[val]

        phases = []
        for modkey, c in val.references:
            phases.append(
                tuple(evolve(pt, phi=pt.phi*c) for pt in self[modkey])
            )

        return sorted(it.chain.from_iterable(phases))

    def __contains__(self, val):
        """Returns True if all references are in the phase dictionary."""
        if isinstance(val, str):
            val = ModulationFrequency.from_string(val)

        return all(v in self.phases for v in val.variables())

    def append(self, modkey: ModulationFrequency | str, phase: PhaseJump):
        """Adds a phase jump."""
        if isinstance(modkey, str):
            modkey = ModulationFrequency.from_string(modkey)

        if modkey.references:
            raise ValueError(
                f"Cannot add a virtual phase on a non-independent phase {modkey}"
            )

        self.phases[modkey].append(phase)

    @staticmethod
    def compress(phases: Iterable[PhaseJump]) -> list[PhaseJump]:
        """Returns a list of phases with a single entry per timepoint."""
        return [sum(tphis) for _, tphis in it.groupby(phases, key=lambda pt: pt.t)]


@runtime_checkable
class PhaseUpdater(Protocol):
    def update_phase_tracker(
        self,
        time: float,
        phase_tracker: dict[ModulationFrequency, list[tuple[float, float]]]
    ) -> None:
        ...