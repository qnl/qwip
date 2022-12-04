from numbers import Real
from typing import Union, ForwardRef

import attrs
from attrs import field, resolve_types

from qwip.settings.settings import qfrozen
from qwip.flatdict import FlatDict
from qwip.settings.validation import resolve_types_with_validation


def _type_error_text(op1, op2, operand: str) -> str:
    return f'unsupported operand type(s) for {operand}: {type(op1)!r} and {type(op2)!r}'

def _convert_offset(val) -> float | str:
    try:
        return float(val)
    except ValueError:
        return val

@qfrozen(kw_only=False)
class Location:
    offset: float | str = field(converter=_convert_offset, default=0)
    references: frozenset[tuple[ForwardRef('Location'), float]] = field(factory=frozenset)

    def substitute(self, new: 'Location | Real', old: 'Location') -> 'Location':
        if old == new:
            return new if isinstance(new, type(self)) else type(self)(new)
        else:
            found_loc = False
            refs = {}

            for loc, c in self.references:
                sub = loc == old
                found_loc = found_loc or sub

                refs.add((new if sub else loc, c))

            if not found_loc:
                raise ValueError(f'{old} is not a dependency of {self}.')

            return Location(
                offset=self.offset,
                reference=frozenset(refs)
            )

    def __neg__(self) -> 'Location':
        """Negates a location."""
        if isinstance(self.offset, str):
            offset = 0
            return Location(0, frozenset({(self, -1)}))

        offset = -self.offset
    
        return Location(
            offset=offset,
            references=frozenset((loc, -c) for loc, c in self.references)
        )

    def __add__(self, other) -> 'Location':
        """Adds two locations."""
        zero = Location()
        if other == 0 or other == zero:
            return self
        elif self == zero:
            return other if isinstance(other, Location) else Location(other)

        # Check for string or number
        if isinstance(other, (str, Real)):
            other = Location(other)
        elif not isinstance(other, Location):
            raise TypeError(_type_error_text(self, other, '+'))

        offset = 0
        coeff_map = {}
        if isinstance(self.offset, str):
            coeff_map[self] = 1
        else:
            offset += self.offset
            coeff_map = {loc: c for loc, c in self.references}
        
        if isinstance(other.offset, str):
            coeff_map[other] = coeff_map.get(other, 0) + 1
        else:
            offset += other.offset
            for loc, c in other.references:
                coeff_map[loc] = coeff_map.get(loc, 0) + c

        return Location(
            offset=offset,
            references=frozenset(
                (loc, c) for loc, c in coeff_map.items() if c != 0
            )
        )

    def __radd__(self, other) -> 'Location':
        """Adds two locations."""
        return self.__add__(other)

    def __sub__(self, other) -> 'Location':
        """Subtracts two locations."""
        zero = Location()
        if other == 0 or other == zero:
            return self
        elif self == zero:
            return -other if isinstance(other, Location) else Location(-other)

        # Check for string or number
        if isinstance(other, (str, Real)):
            other = Location(other)
        elif not isinstance(other, Location):
            raise TypeError(_type_error_text(self, other, '-'))

        offset = 0
        coeff_map = {}
        if isinstance(self.offset, str):
            coeff_map[self] = 1
        else:
            offset += self.offset
            coeff_map = {loc: c for loc, c in self.references}
        
        if isinstance(other.offset, str):
            coeff_map[other] = coeff_map.get(other, 0) - 1
        else:
            offset -= other.offset
            for loc, c in other.references:
                coeff_map[loc] = coeff_map.get(loc, 0) - c

        return Location(
            offset=offset,
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

        zero = Location()
        if other == 0 or self == zero:
            return zero
        elif other == 1:
            return self

        if isinstance(self.offset, str):
            return Location(
                0,
                frozenset((self, other))
            )
        else:
            return Location(
                self.offset * other,
                frozenset((loc, c * other) for loc, c in self.references)
            )

    def __rmul__(self, other) -> 'Location':
        """Scalar multiplication of a location."""
        return self.__mul__(other)

resolve_types_with_validation(Location, globals(), locals())
