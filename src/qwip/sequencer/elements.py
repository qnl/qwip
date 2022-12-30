from typing import Callable, Union, ForwardRef
from numbers import Real
from attrs import field
from collections.abc import Collection
from typing_extensions import Self
from copy import copy, deepcopy

import numpy as np

from qwip.settings.settings import qdefine
from qwip.sequencer.utils import Location
from qwip.sequencer.waveform import (
    Waveform,
    Channel,
    ModulatedWaveform,
    Marker,
    VirtualZWaveform,
    ModulationFrequency
)

SequenceNode = Union[Waveform, ForwardRef('SequenceElement')]
LocationLike = Location | str | Real

class UnderconstrainedSolveError(Exception):
    ...

@qdefine
class SequenceElement:
    locations: dict[Location, list[SequenceNode]] = field(factory=dict)
    constraints: dict[str, Location | str] = field(factory=dict)
    channels: set[Channel] = field(factory=set)

    @classmethod
    def fromtuples(
        cls,
        pulse_locations: list[tuple[LocationLike, Waveform | None]],
        **constraints
    ) -> 'SequenceElement':
        """Constructs a sequence from a tuple of locations and waveforms.
        
        The constructor will add `Location('start')` to every location if it does not
        already contain a 'start' location.

        Args:
            locations: A list of (location, pulse) pairs to add to the sequence
            **constraints: remaining keyword arguments will be added to the mapping
                of constraints.

        Returns:
            The resulting sequence element instance
        """
        locations = {}
        channels = set()

        for loc, wave in pulse_locations:
            if isinstance(loc, (str, Real)):
                loc = Location(loc)

            locations[loc] = locations.get(loc, list())

            if wave is not None:
                locations[loc].append(wave)
                channels.update(wave.channels)

        constraints = {
            k: Location(l) if isinstance(l, str) else l for k, l in constraints.items()
        }

        return cls(locations=locations, constraints=constraints, channels=channels)

    def add_waveform(
        self,
        waveform: Waveform | Collection[Waveform],
        location: LocationLike = Location(),
    ) -> Self:
        """Adds a waveform to the sequence element at the specified location.

        Args:
            location: The location at which to place the waveform
            waveform: The waveform to add
        """
        if not isinstance(location, Location):
            location = Location(location)

        if isinstance(waveform, Waveform):
            waveform = [waveform]
        
        self.locations[location] = self.locations.get(location, list()) + waveform
        self.channels.update(*(w.channels for w in waveform))

        return self

    def remove_waveform(
        self,
        waveform: Waveform | Collection[Waveform],
        location: LocationLike | None = None
    ) -> Self:
        """Removes a waveform to the sequence element.
        
        By default will remove all instances of the waveform(s). If a
        location is specified, only instances of the waveform at the 
        specified location are removed.
        """

        if isinstance(waveform, Waveform):
            waveform = (waveform,)

        if location is not None:
            if not isinstance(location, Location):
                location = Location(location)

            self.locations[location][:] = [
                w for w in self.locations[location] if w not in waveform
            ]

            return self

        for loc in self.locations:
            self.locations[loc][:] = [
                w for w in self.locations[loc] if w not in waveform
            ]

        return self

    def add_constraints(
        self,
        *,
        overwrite: bool = True,
        **kwargs
    ) -> None:
        """Adds constraints to the set of existing constraints.
        
        All constraints are of the form `'variable_name' = Location(...)`.

        Args:
            overwrite: Whether to overwrite existing constraints for the specified
                variables. Defaults to True.
            **kwargs: constraints are specified as name=location arguments
        """

        for name, location in kwargs.items():
            if not isinstance(location, Location):
                location = Location(location)

            if not overwrite and name in self.constraints:
                raise ValueError(
                    f'Constraint {self.constraints[name]} for {name} already exists. '
                    f'Set `overwrite=True` to overwrite this constraint.'
                )

            self.constraints[name] = location

    def remove_constraint(
        self,
        name: str
    ) -> Location | None:
        """Removes a constraint from the constraint mapping.
        
        Args:
            name: The variable to remove the constraint for.

        Returns:
            The Location specified in the constraint or None if `name` was
            not in the constraint mapping.
        """
        return self.constraints.pop(name, None)

    def append(self,
        other: 'SequenceElement',
        self_loc: LocationLike = Location(),
        other_loc: LocationLike = Location(),
        shared: set[str] = set(),
        name: str | None = None,
    ) -> 'SequenceElement':
        """Appends a sequence element.
        
        Args:
            other: The sequence element to append.
            name: A (optional) variable name to set the new location of the origin
                for the `other` sequence. This makes it simple to shift the origin
                later.
            self_loc: The location in the current sequence element to line up with
                the location in the `other` sequence element.
            other_loc: The location in the `other` sequence element to line up with
                the location in the current sequence element.
            shared: The set of variables that are shared between the two sequence
                elements.

        Raises:
            ValueError: If the two sequence elements share any variables that are not
                explicitly declared in `shared`, or if `name` conflicts with any
                existing variables.
        """
        if not isinstance(self_loc, Location):
            self_loc = Location(self_loc)
        
        if not isinstance(other_loc, Location):
            other_loc = Location(other_loc)
        
        conflict = (self.variables() & other.variables()) - shared
        if conflict:
            errorstring = '\n\t' + '\n\t'.join(f'- {v}' for v in conflict)
            raise ValueError(
                f'The following variables exist in both sequence elements. Rename '
                f'the variables in one sequence to avoid conflicts or declare them '
                f'as shared variables.{errorstring}'
            )

        if name in (self.variables() | other.variables()):
            raise ValueError(
                f'Variable name \'{name}\' is already in use.'
            )
        
        dt = self_loc - other_loc
        if name is not None:
            self.constraints[name] = dt
            dt = Location(name)

        for loc, waves in other.locations.items():
            loc += dt
            self.locations[loc] = self.locations.get(loc, [])
            self.locations[loc].extend(waves)

        for var, loc in other.constraints:
            if var in self.constraints and self.constraints[var] != loc:
                raise ValueError(
                    f'Conflicting constraints:\n'
                    f'\t{var} = {self.constraints[var]}\n'
                    f'\t{var} = {loc}'
                )

            self.constraints[var] = loc

        self.channels.update(other.channels)

        return self

    def variables(self) -> set[str]:
        """Returns the set of variables referenced in the sequence element.

        Returns:
            A set that contains the names of all variables referenced within the
            location mapping or the constraint mapping.
        """
        lvars = (
            loc.variables(return_string=True) for loc in self.locations
        )
        cvars = (
            {k, *loc.variables(return_string=True)} 
                for k, loc in self.constraints.items()
        )
        
        return set().union(*cvars, *lvars)

    def rename_variables(
        self,
        rename_func: Callable[[str], str]
    ) -> set[str]:
        """Renames all variables.

        This function renames variables according to `rename_func`.

        Args:
            rename_func: a function that, given a string, returns a new string.

        Returns:
            A set containing the new names of all variables referenced by the
            sequence element.
        """
        varmap = {n: rename_func(n) for n in self.variables()}
        
        self.locations = {loc.resolve(**varmap): waves for loc, waves in self.locations.items()}
        self.constraints = {
            varmap.get(name, name): loc.resolve(**varmap) for name, loc in self.constraints
        }

        return set(varmap.items())

    @staticmethod
    def _solve_constraint_matrix(
        basis_set: dict[Location, int],
        constraints: dict[str, Location]
    ) -> dict[str, Location]:
        """Solves a constraint matrix.

        Args:
            basis_set: A dictionary mapping variables to a basis index. This dictionary
                should assign a unique integer in [0, N) to each variable, where
                N is the number of variables in the basis set.
            constraints: The set of constraints to solve. This specifies a linear system
                of equations.

        Returns:
            A dict mapping variable names to concrete locations.

        Raises:
            numpy.linalg.LinalgError: If the constraint matrix is singular.
        """
        N = len(basis_set)

        A = np.zeros((N, N))
        b = np.zeros(N)

        for y, xs in constraints.items():
            y = Location(y)
            y_idx = basis_set[y]
            
            b[y_idx] = xs.offset

            A[y_idx, y_idx] = 1

            for x, coefficient in xs.references:
                x_idx = basis_set[x]
                A[y_idx, x_idx] -= coefficient

        result = np.linalg.solve(A, b)

        return {loc.offset: result[i] for loc, i in basis_set.items()}
            

    def solve_constraints(self, **kwargs) -> dict[str, Location]:
        """Solves all timing constraints for the sequence element.

        **kwargs: Keyword arguments can be used to add constraints and are passed
            directly to `add_constraints`.

        Returns:
            A dictionary mapping all location names to concrete locations.
        """
        self.add_constraints(**kwargs)

        all_vars = self.variables()

        if (num_vars := len(all_vars)) > (num_cons := len(self.constraints)):
            raise UnderconstrainedSolveError(
                f'Found {num_vars} variables but only {num_cons} constraints. '
                f'(variables = {all_vars})'
            )

        basis_set = {
            Location(v): i for i, v in enumerate(all_vars)
        }

        result = type(self)._solve_constraint_matrix(basis_set, self.constraints)

        return result

    def resolve_locations(
        self,
        sort: bool = True,
        reset_zero: str | None = 'neg',
        end_marker: str = 'end',
        **kwargs
    ) -> dict[Location, list[Waveform]]:
        """Resolves all locations into concrete times.

        Optionally time orders the location mapping and sets the earliest location
        to t = 0.
        
        Args:
            sort: Whether or not to time order the location mapping.
            reset_zero: Whether or not to translate the location mapping such that
                the earliest location is t = 0. Can be 'neg', 'pos', 'both', or None.
            **kwargs: Additional constraints to add to the sequence elements
                before solving for the locations.

        Returns:
            A dictionary mapping concrete locations to lists of Waveforms
        """
        constraints = self.solve_constraints(**kwargs)

        locations = dict()

        t_max = Location()

        for loc, waves in self.locations.items():
            loc = loc.resolve(**constraints)
            locations[loc] = locations.get(loc, list())

            locations[loc].extend(waves)

            t = loc + max(w.width for w in waves)

            t_max = t if t > t_max else t_max

        locations[t_max] = locations.get(t_max, [])
        locations[t_max] += [Marker(name=end_marker)]

        if sort:
            locations = dict(sorted(locations.items(), key=lambda l: l[0]))

        t0 = next(iter(locations)) if sort else min(locations)
        should_reset = (
            (reset_zero == 'neg' and t0 < Location()) or
            (reset_zero == 'pos' and t0 > Location()) or
            (reset_zero == 'both')
        )
        if should_reset:
            locations = {l - t0: w for l, w in locations.items()}

        return locations

    def translate(
        self,
        dt: LocationLike,
    ) -> Self:
        """Shifts a sequence element in time.
        
        This function translates all the waveforms in the sequence element by dt, which
        is equivalent to taking s(t) -> s(t - dt).

        Args:
            dt: Amount of time to translate locations by.

        Returns:
            The sequence element.
        """
        
        self.locations = {
            loc + dt: waves for loc, waves in self.locations
        }

        return self

    def copy(
        self,
        deep: bool = True
    ) -> Self:
        """Copies a sequence element.
        
        This function makes a copy of the sequence element. Defaults to making a deep copy
        but can also make a shallow copy where the constraint dict and waveform mappings are
        shared.

        Args:
            deep: Whether to make a deep copy or shallow copy. Defaults to True.
        
        Returns:
            The new sequence element.
        """

        return deepcopy(self) if deep else copy(self)

    def get_channel_map(
        self,
        *channels: str | Channel
    ) -> dict[Channel, list[tuple[Location, Waveform]]]:
        """Splits
        
        Makes a single pass through the location dict.

        Args:
            channels: The channels to include in the channel map.

        Returns:
            A dictionary mapping channels to (location, waveform) pairs.
        """

        if channels:
            channels = (
                Channel(c) if isinstance(c, str) else c for c in channels
            )
        else:
            channels = self.channels

        channel_map = {c: [] for c in channels}

        for loc, waves in self.locations.items():
            for w in waves:
                for wchan in w.channels:
                    if wchan in channel_map:
                        channel_map[wchan].append((loc, w))

        return channel_map

    def __getitem__(self, key: LocationLike):
        try:
            if not isinstance(key, Location):
                key = Location(key)
        except:
            pass

        return self.locations[key]

    def __contains__(self, waveform: Waveform) -> bool:
        """Checks if the waveform exists in the sequence element."""

        for waves in self.locations.values():
            if waveform in waves:
                return True

        return False

    def __add__(self, other: Self) -> Self:
        """Adds two sequence elements.

        The sum of two sequence elements s(t) and r(t) is equivalent to the
        pointwise addition at every point in time.

        Args:
            other: The other sequence element to add.
        
        Returns:
            A new sequence element equal to s(t) + r(t).
        """

        locations = dict()
        channels = set()
        constraints = dict()

        for loc, waves in self.locations.items():
            locations[loc] = []
            for w in waves:
                locations[loc].append(w)

        for loc, waves in other.locations.items():
            locations[loc] = locations.get(loc, list())
            for w in waves:
                locations[loc].append(w)
        
        channels.update(self.channels)
        channels.update(other.channels)

        for key, loc in self.constraints.items():
            constraints[key] = loc

        for key, loc in other.constraints.items():
            if constraints.get(key, loc) != loc:
                raise ValueError(
                    f'Cannot add two sequences with conflicting constraints. '
                    f'{key} = {loc} is incompatible with {key} = {constraints[key]}.'
                )

            constraints[key] = loc

        return SequenceElement(
            locations=locations,
            constraints=constraints,
            channels=channels
        )
