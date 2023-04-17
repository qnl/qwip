import ast
import itertools as it
import operator
import re
from functools import lru_cache
from numbers import Real
from typing import ForwardRef, TypeVar, Union

import attrs
from attrs import field, resolve_types
from typing_extensions import Self

import qwip
from qwip._cattr import make_attrs_structure_fn, make_attrs_unstructure_fn
from qwip.attrs import qfrozen
from qwip.flatdict import FlatDict


def _type_error_text(op1, op2, operand: str) -> str:
    return f"unsupported operand type(s) for {operand}: {type(op1)!r} and {type(op2)!r}"


def _convert_offset(val) -> float | str:
    try:
        return float(val)
    except ValueError:
        return val


def _no_refs_in_string_location(inst, attr, value):
    if isinstance(inst.offset, str) and len(value) != 0:
        raise ValueError(
            f"Variables cannot hold references. Got {value} that" f"is non-empty."
        )


@attrs.define(kw_only=False)
class _ExprParser(ast.NodeVisitor):
    _cls: type
    error: str
    operator_func: dict = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def generic_visit(self, node):
        raise ValueError(self.error)

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Module(self, node):
        return self.visit(node.body[0])

    def visit_BinOp(self, node):
        return self.operator_func[type(node.op)](
            self.visit(node.left), self.visit(node.right)
        )

    def visit_UnaryOp(self, node: ast.UnaryOp):
        return self.operator_func[type(node.op)](self.visit(node.operand))

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_Name(self, node: ast.Name):
        return self._cls(node.id)

    def visit_Load(self, node: ast.Load):
        return self.visit(node)


@qfrozen(kw_only=False, repr=False)
class LinearExpression:
    offset: float | str = field(converter=_convert_offset, default=0)
    references: frozenset[tuple[Self, float]] = field(
        validator=_no_refs_in_string_location, factory=frozenset
    )

    @property
    def resolved(self) -> bool:
        """True if a LinearExpression contains no variables.

        Returns:
            A boolean that specifies if a linear expression has any variables.
        """
        return not bool(self.variables())

    def substitute(self, new: Self | Real, old: Self) -> Self:
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
                raise ValueError(f"{old} is not a dependency of {self}.")

            return cls(offset=self.offset, reference=frozenset(refs))

    def resolve(self, **variable_map):
        """Resolves string variables referenced in a location."""
        cls = type(self)
        if (
            isinstance(self.offset, str)
            and (loc := variable_map.get(self.offset)) is not None
        ):
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
            loc.variables(return_string=return_string) for loc, c in self.references
        )
        return set().union(*subsets)

    @classmethod
    def from_string(cls, s: str, /):
        """Class constructor for a string expression.

        This constructor takes a string and converts it to a LinearExpression
        by parsing the string into a Python ast.

        Args:
            s: The string to convert to a LinearExpression.

        Returns:
            The resulting LinearExpression.

        Raises:
            TypeError: If `s` is not a string.
            ValueError: If the string `s` is not a valid expression.
        """
        error_str = f'Unable to convert invalid linear expression: "{s}".'

        try:
            tree = ast.parse(s)
        except SyntaxError as e:
            raise ValueError(error_str) from e
        except TypeError as e:
            raise TypeError(f"s must be a str, got {s} that is {type(s)}.") from e

        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            raise ValueError(error_str)

        calc = _ExprParser(cls, error_str)
        try:
            expr = calc.visit(tree)
        except TypeError as e:
            raise ValueError(error_str) from e

        if not isinstance(expr, cls):
            expr = cls(expr)

        return expr

    def __repr__(self) -> str:
        """repr for LinearExpression.

        Returns:
            A string representation of the expression that specifies the type.
        """
        return f"{type(self).__name__}({str(self)})"

    def __str__(self) -> str:
        """str for LinearExpression.

        Returns:
            A string representation of the expression.
        """

        def monomial_to_str_tuple(c, l):
            if c == 1:
                return (" + ", l)

            if not (l.startswith("(") and l.endswith(")")) and re.search(r"[\+-]", l):
                l = f"({l})"

            if c < 0:
                return (" - ", monomial_to_str_tuple(-c, l)[1])

            return (" + ", f"{c:g} * {l}")

        terms = [monomial_to_str_tuple(c, str(loc)) for loc, c in self.references]
        if self.offset or not terms:
            constant = self.offset
            if not isinstance(constant, str):
                constant = f"{constant:g}"
            terms = [constant] + terms

        eq_str = "".join(it.chain(*terms))

        # Necessary to remove white space before unary operators
        return eq_str.strip(" +")

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
            references=frozenset((loc, -c) for loc, c in self.references if c != 0),
        )

    def __pos__(self) -> Self:
        """Identity."""
        return self

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
            raise TypeError(_type_error_text(self, other, "+"))

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
            ((loc, c),) = coeff_map.items()
            if c == 1:
                return loc

        return cls(
            offset=offset,
            references=frozenset((loc, c) for loc, c in coeff_map.items() if c != 0),
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
            raise TypeError(_type_error_text(self, other, "-"))

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
            ((loc, c),) = coeff_map.items()
            if c == 1:
                return loc

        return cls(
            offset=offset,
            references=frozenset((loc, c) for loc, c in coeff_map.items() if c != 0),
        )

    def __rsub__(self, other) -> Self:
        """Subtracts two locations."""
        return -self.__sub__(other)

    def __mul__(self, other) -> Self:
        """Scalar multiplication of a location."""
        if not isinstance(other, Real):
            raise TypeError(_type_error_text(self, other, "*"))

        cls = type(self)

        zero = cls()
        if other == 0 or self == zero:
            return zero
        elif other == 1:
            return self

        if isinstance(self.offset, str):
            return cls(0, {(self, other)})

        if len(self.references) == 1 and self.offset == 0:
            loc, c = next(iter(self.references))
            if c * other == 1:
                return loc
        else:
            return cls(
                self.offset * other,
                frozenset((loc, c * other) for loc, c in self.references),
            )

    def __rmul__(self, other) -> Self:
        """Scalar multiplication of a location."""
        return self.__mul__(other)

    def __div__(self, other) -> Self:
        """Scalar division of a location."""
        return self.__mul__(1 / other)

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
                "Variable expressions must be resolved before comparison. "
                f"({self} < {other})"
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
                "Variable expressions must be resolved before comparison. "
                f"({self} > {other})"
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
                "Variable expressions must be resolved before comparison. "
                f"({self} <= {other})"
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
                "Variable expressions must be resolved before comparison. "
                f"({self} >= {other})"
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


def make_linear_expression_structure_fn(cls):
    structure_attrs = make_attrs_structure_fn(cls)

    def structure_fn(obj, cls):
        if isinstance(obj, (str, Real)):
            return cls.from_string(str(obj))

        return structure_attrs(obj, cls)

    return structure_fn


def make_linear_expression_unstructure_fn(cls):
    def unstructure_fn(obj):
        if len(obj.references):
            return str(obj)

        return obj.offset

    return unstructure_fn


qwip.converter.register_structure_hook_factory(
    lambda cls: issubclass(cls, LinearExpression), make_linear_expression_structure_fn
)

qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, LinearExpression), make_linear_expression_unstructure_fn
)


@qfrozen(kw_only=False, repr=False)
class Location(LinearExpression):
    ...


__all__ = ["Location"]
