from numbers import Real
from typing import Union, ForwardRef, TypeVar
from functools import lru_cache
from typing_extensions import Self

import attrs
from attrs import field, resolve_types

import qwip
from qwip._cattr import make_attrs_structure_fn
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

def _no_refs_in_string_location(inst, attr, value):
    if isinstance(inst.offset, str) and len(value) != 0:
        raise ValueError(
            f'Variables cannot hold references. Got {value} that'
            f'is non-empty.'
        )

@qfrozen(kw_only=False)
class LinearExpression:
    offset: float | str = field(converter=_convert_offset, default=0)
    references: frozenset[tuple[Self, float]] = field(
        validator=_no_refs_in_string_location,
        factory=frozenset
    )

    def substitute(
        self,
        new: Self | Real,
        old: Self
    ) -> Self:
        cls = type(self)
        if old == new:
            return new if isinstance(new, cls) else cls(new)
        else:
            found_loc = False
            refs = {}

            for loc, c in self.references:
                sub = loc == old
                found_loc = found_loc or sub

                refs.add((new if sub else loc, c))

            if not found_loc:
                raise ValueError(f'{old} is not a dependency of {self}.')

            return cls(
                offset=self.offset,
                reference=frozenset(refs)
            )
    
    def resolve(self, **variable_map):
        """Resolves string variables referenced in a location."""
        cls = type(self)
        if isinstance(self.offset, str) and (loc := variable_map.get(self.offset)) is not None:
            return loc if isinstance(loc, cls) else cls(loc)
        elif len(self.references) == 0:
            return self

        offset = self.offset
        unresolved_refs = {}
        for loc, c in self.references:
            loc = loc.resolve(**variable_map)

            if not isinstance(loc.offset, str):
                offset += loc.offset * c
                
                for subloc, subc in loc.references:
                    unresolved_refs[subloc] = unresolved_refs.get(subloc, 0) + c * subc
            else:
                unresolved_refs[loc] = unresolved_refs.get(loc, 0) + c

        return cls(offset, frozenset(unresolved_refs.items()))

    @lru_cache
    def variables(self, return_string=False) -> set[Self]:
        """Returns the set of variables that the location depends on."""

        if isinstance(self.offset, str):
            return {self.offset if return_string else self}
        
        subsets = (
            loc.variables(return_string=return_string)
                for loc, c in self.references
        )
        return set().union(*subsets)

    def __contains__(self, variable: str) -> bool:
        return variable in self.variables(return_string=True)

    def __neg__(self) -> Self:
        """Negates a location."""
        cls = type(self)

        if isinstance(self.offset, str):
            offset = 0
            return cls(0, frozenset({(self, -1)}))

        offset = -self.offset

        if len(self.references) == 1 and offset == 0:
            loc, c = next(iter(self.references))
            if -c == 1:
                return loc
    
        return cls(
            offset=offset,
            references=frozenset((loc, -c) for loc, c in self.references if c != 0)
        )

    def __add__(self, other) -> Self:
        """Adds two locations."""
        cls = type(self)

        zero = cls()
        if other == 0 or other == zero:
            return self
        elif self == zero:
            return other if isinstance(other, cls) else cls(other)

        # Check for string or number
        if isinstance(other, (str, Real)):
            other = cls(other)
        elif not isinstance(other, cls):
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

        if len(coeff_map) == 1 and offset == 0:
            (loc, c), = coeff_map.items()
            if c == 1:
                return loc

        return cls(
            offset=offset,
            references=frozenset(
                (loc, c) for loc, c in coeff_map.items() if c != 0
            )
        )

    def __radd__(self, other) -> Self:
        """Adds two locations."""
        return self.__add__(other)

    def __sub__(self, other) -> Self:
        """Subtracts two locations."""
        cls = type(self)
    
        zero = cls()
        if other == 0 or other == zero:
            return self
        elif self == zero:
            return -other if isinstance(other, cls) else -cls(other)

        # Check for string or number
        if isinstance(other, (str, Real)):
            other = cls(other)
        elif not isinstance(other, cls):
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

        if len(coeff_map) == 1 and offset == 0:
            (loc, c), = coeff_map.items()
            if c == 1:
                return loc

        return cls(
            offset=offset,
            references=frozenset(
                (loc, c) for loc, c in coeff_map.items() if c != 0
            )
        )
    
    def __rsub__(self, other) -> Self:
        """Subtracts two locations."""
        return -self.__sub__(other)

    def __mul__(self, other) -> Self:
        """Scalar multiplication of a location."""
        if not isinstance(other, Real):
            raise TypeError(_type_error_text(self, other, '*'))

        cls = type(self)

        zero = cls()
        if other == 0 or self == zero:
            return zero
        elif other == 1:
            return self

        if isinstance(self.offset, str):
            return cls(
                0,
                {(self, other)}
            )

        if len(self.references) == 1 and self.offset == 0:
            loc, c = next(iter(self.references))
            if c*other == 1:
                return loc
        else:
            return cls(
                self.offset * other,
                frozenset((loc, c * other) for loc, c in self.references)
            )

    def __rmul__(self, other) -> Self:
        """Scalar multiplication of a location."""
        return self.__mul__(other)

    def __div__(self, other) -> Self:
        """Scalar division of a location."""
        return self.__mul__(1/other)

    def __lt__(self, other) -> bool:
        """Compares two expressions.
        
        Returns:
            True if self < other.
        """
        cls = type(self)

        if not isinstance(other, cls):
            other = cls(other)

        if self.variables() or other.variables():
            raise ValueError(
                'Variable expressions must be resolved before comparison. '
                f'({self} < {other})'
            )
        
        if self.references:
            self = self.resolve()

        if other.references:
            other = other.resolve()

        return self.offset < other.offset

    def __gt__(self, other) -> bool:
        """Compares two expressions.
        
        Returns:
            True if self > other.
        """
        cls = type(self)

        if not isinstance(other, cls):
            other = cls(other)

        if self.variables() or other.variables():
            raise ValueError(
                'Variable expressions must be resolved before comparison. '
                f'({self} > {other})'
            )
        
        if self.references:
            self = self.resolve()

        if other.references:
            other = other.resolve()

        return self.offset > other.offset

    def __le__(self, other) -> bool:
        """Compares two expressions.
        
        Returns:
            True if self <= other.
        """
        cls = type(self)

        if not isinstance(other, cls):
            other = cls(other)

        if self.variables() or other.variables():
            raise ValueError(
                'Variable expressions must be resolved before comparison. '
                f'({self} <= {other})'
            )
        
        if self.references:
            self = self.resolve()

        if other.references:
            other = other.resolve()

        return self.offset <= other.offset

    def __ge__(self, other) -> bool:
        """Compares two expressions.
        
        Returns:
            True if self >= other.
        """
        cls = type(self)

        if not isinstance(other, cls):
            other = cls(other)

        if self.variables() or other.variables():
            raise ValueError(
                'Variable expressions must be resolved before comparison. '
                f'({self} >= {other})'
            )
        
        if self.references:
            self = self.resolve()

        if other.references:
            other = other.resolve()

        return self.offset >= other.offset

    def __copy__(self) -> Self:
        """Overrides copy for LinearExpression objects.
        
        Since LinearExpressions are immutable and only contain references
        to other immutable objects we just return self instead of 
        unnecessarily creating new objects.

        Returns:
            The LinearExpression object.
        """
        return self

    def __deepcopy__(self, memo) -> Self:
        """Overrides deepcopy for LinearExpression objects.
        
        Since LinearExpressions are immutable and only contain references
        to other immutable objects we just return self instead of 
        unnecessarily creating new objects.

        Returns:
            The LinearExpression object.
        """
        return self


resolve_types_with_validation(LinearExpression, globals(), locals())


def make_linear_expression_structure_fn(cls):
    structure_attrs = make_attrs_structure_fn(cls)

    def structure_fn(obj, cls):
        if isinstance(obj, str):
            return cls(obj)

        return structure_attrs(obj, cls)

    return structure_fn

qwip.converter.register_structure_hook_factory(
    lambda cls: issubclass(cls, LinearExpression),
    make_linear_expression_structure_fn
)

@qfrozen(kw_only=False)
class Location(LinearExpression):
    ...