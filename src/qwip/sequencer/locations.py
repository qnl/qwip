from numbers import Real
from typing import ClassVar

import attrs
from attrs import field, resolve_types

from qwip.settings.settings import qfrozen
from qwip.flatdict import FlatDict
from qwip.settings.validation import resolve_types_with_validation


def _type_error_text(op1, op2, operand: str) -> str:
    return f'unsupported operand type(s) for {operand}: {type(op1)!r} and {type(op2)!r}'

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
            raise TypeError(_type_error_text(self, other, '+'))

        if isinstance(other, Real):
            return Location(
                offset=self.offset + other, references=self.references
            )

        coeff_map = {loc: c for loc, c in self.references}
        
        for loc, c in other.references:
            coeff_map[loc] = coeff_map.get(loc, 0) + c

        return Location(
            offset=self.offset + other.offset,
            references=frozenset(
                (loc, c) for loc, c in coeff_map.items() if c != 0
            )
        )

    def __radd__(self, other) -> 'Location':
        """Adds two locations."""
        return self.__add__(other)

    def __sub__(self, other) -> 'Location':
        """Subtracts two locations."""
        if not isinstance(other, (Location, Real)):
            raise TypeError(_type_error_text(self, other, '-'))

        if isinstance(other, Real):
            return Location(
                offset=self.offset - other, references=self.references
            )

        coeff_map = {loc: c for loc, c in self.references}

        for loc, c in other.references:
            coeff_map[loc] = coeff_map.get(loc, 0) - c

        return Location(
            offset=self.offset - other.offset,
            references=frozenset(
                (loc, c) for loc, c in coeff_map.items() if c != 0
            )
        )
    
    def __rsub__(self, other) -> 'Location':
        """Subtracts two locations."""
        return -self.__sub__(other)

    def __mul__(self, other) -> 'Location':
        """Scalar multiplication of a location."""
        if not isinstance(other, Real):
            raise TypeError(_type_error_text(self, other, '*'))

        if other == 0:
            return Location()
        else:
            return Location(
                self.offset * other,
                frozenset((loc, c * other) for loc, c in self.references)
            )

    def __rmul__(self, other) -> 'Location':
        """Scalar multiplication of a location."""
        return self.__mul__(other)

resolve_types_with_validation(Location, globals(), locals())
