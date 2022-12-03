from numbers import Real

import attrs
from attrs import field, resolve_types

from qwip.settings.settings import qfrozen
from qwip.flatdict import FlatDict
from qwip.settings.validation import resolve_types_with_validation


@qfrozen(kw_only=False)
class Location:
    offset: float | None = 0
    references: 'frozenset[tuple[Location, float]]' = field(factory=frozenset)

    def __neg__(self) -> 'Location':
        """Negates a location."""

        return Location(
            offset=-self.offset,
            references=frozenset((loc, -c) for loc, c in self.references)
        )

    def __add__(self, other) -> 'Location':
        """Adds two locations."""
        if not isinstance(other, (Location, Real)):
            raise TypeError(
                f'unsupported operand type(s) for +: {type(self)!r} and {type(other)!r}'
            )

        if isinstance(other, Real):
            return Location(
                offset=self.offset + other, references=self.references
            )

        coeff_map = {loc: c for loc, c in self.references}
        
        for c, loc in other.references:
            coeff_map[loc] = coeff_map.get(loc, 0) + c

        return Location(
            offset=self.offset + other.offset,
            references=frozenset(
                (loc, c) for loc, c in coeff_map.items() if c != 0
            )
        )

    def __radd__(self, other) -> 'Location':
        """Adds two locations."""
        if not isinstance(other, (Location, Real)):
            raise TypeError(
                f'unsupported operand type(s) for +=: {type(self)!r} and {type(other)!r}'
            )
        return self.__add__(other)

    def __sub__(self, other_location) -> 'Location':
        """Subtracts two locations."""
        coeff_map = {loc: c for loc, c in self.references}

        for c, loc in other_location.references:
            coeff_map[loc] = coeff_map.get(loc, 0) - c

        return Location(
            offset=self.offset - other_location.offset,
            references=frozenset(
                (loc, c) for loc, c in coeff_map.items() if c != 0
            )
        )
    
    def __rsub__(self, other_location) -> 'Location':
        """Subtracts two locations."""
        return self.__sub__(other_location)

resolve_types_with_validation(Location, globals(), locals())