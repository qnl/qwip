import itertools as it
from contextlib import nullcontext as noerror
from copy import copy, deepcopy

import attrs
import pytest

import qwip
from qwip.sequencer.utils import LinearExpression

LE = LinearExpression


class TestLinearExpression:
    def test_create(self):
        loc = LE()

        assert loc.offset == 0 and loc.references == frozenset()

        l1 = LE(-1)
        l2 = LE(0)
        l3 = LE(1.5)

        assert l1.offset == -1 and l1.references == frozenset()
        assert l2.offset == 0 and l2.references == frozenset()
        assert l3.offset == 1.5 and l3.references == frozenset()

        l4 = LE(references={(l1, 0.5), (l2, 0.5), (l3, 0.5)})
        assert l4.references == frozenset({(l1, 0.5), (l2, 0.5), (l3, 0.5)})

    def test_create_str(self):
        loc = LE("a")
        assert loc.offset == "a" and loc.references == frozenset()

        loc = LE(0, {("a", -1)})
        assert loc.offset == 0 and loc.references == frozenset({(LE("a"), -1)})

        with pytest.raises(ValueError):
            loc = LE("a", {(LE(), 1)})

    @pytest.mark.parametrize(
        "op1,op2,equal",
        [
            (LE(), LE(), True),
            (LE(1), LE(1), True),
            (LE(1), LE(-1), False),
            (LE(0, {(LE(1), 1)}), LE(0, {(LE(1), 1)}), True),
            (LE(0, {(LE(1), 1)}), LE(0, {(LE(-1), -1)}), False),
            (LE("a"), LE("b"), False),
            (LE("a"), LE("a"), True),
        ],
    )
    def test_equal(self, op1, op2, equal):
        assert (op1 == op2) is equal

    @pytest.mark.parametrize(
        "LE,inverse",
        [
            (LE(), LE()),
            (LE("str"), LE(0, {(LE("str"), -1)})),
            (-LE("str"), LE("str")),
            (LE(1), LE(-1)),
            (LE(-1), LE(1)),
            (LE(-1, {(LE(1), 1)}), LE(1, {(LE(1), -1)})),
        ],
    )
    def test_negation(self, LE, inverse):
        assert -LE == inverse

    @pytest.mark.parametrize(
        "op1,op2,result",
        [
            (LE(), LE(), LE()),
            (LE(1), LE(-1), LE()),
            (LE(), LE(5), LE(5)),
            (LE(), 5, LE(5)),
            (5, LE(), LE(5)),
            ([], LE(), pytest.raises(TypeError)),
            (LE(1, {(LE(5), 1)}), LE(-1, {(LE(5), -1)}), LE()),
        ],
    )
    def test_add(self, op1, op2, result):
        if hasattr(result, "__enter__"):
            with result:
                op1 + op2
        else:
            assert op1 + op2 == result

    @pytest.mark.parametrize(
        "op1,op2,result",
        [
            ("a", LE(), LE("a")),
            (LE(), LE("a"), LE("a")),
            (LE("a"), LE("b"), LE(0, {("a", 1), ("b", 1)})),
            (LE("a"), LE(2), LE(2, {("a", 1)})),
            (LE(-1, {("a", 1)}), LE(1), LE("a")),
        ],
    )
    def test_add_str(self, op1, op2, result):
        if hasattr(result, "__enter__"):
            with result:
                op1 + op2
        else:
            assert op1 + op2 == result

    @pytest.mark.parametrize(
        "op1,op2,result",
        [
            (LE(), LE(), LE()),
            (LE(1), LE(1), LE()),
            (LE(), LE(5), LE(-5)),
            (LE(), 5, LE(-5)),
            (5, LE(), LE(5)),
            ([], LE(), pytest.raises(TypeError)),
            (LE(1, {(LE(5), 1)}), LE(1, {(LE(5), 1)}), LE()),
        ],
    )
    def test_subtract(self, op1, op2, result):
        if hasattr(result, "__enter__"):
            with result:
                op1 - op2
        else:
            assert op1 - op2 == result

    @pytest.mark.parametrize(
        "op1,op2,result",
        [
            ("a", LE(), LE("a")),
            (LE(), LE("a"), LE(0, {(LE("a"), -1)})),
            (LE("a"), LE("b"), LE(0, {(LE("a"), 1), (LE("b"), -1)})),
            (LE("a"), LE(2), LE(-2, {(LE("a"), 1)})),
            (LE(1, {(LE("a"), 1)}), LE(1), LE("a")),
        ],
    )
    def test_subtract_str(self, op1, op2, result):
        if hasattr(result, "__enter__"):
            with result:
                op1 - op2
        else:
            assert op1 - op2 == result

    @pytest.mark.parametrize(
        "op1,op2,result",
        [
            (LE(1), 0, LE()),
            (0, LE(1, {(LE(5), 1)}), LE()),
            (LE("a"), 0, LE()),
            (2, LE(0, {(LE("a"), 0.5)}), LE("a")),
            (LE("a"), 3, LE(0, {(LE("a"), 3)})),
            (LE(1, {(LE(2), 1)}), 0.5, LE(0.5, {(LE(2), 0.5)})),
            ([], LE(), pytest.raises(TypeError)),
        ],
    )
    def test_scalar_multiply(self, op1, op2, result):
        if hasattr(result, "__enter__"):
            with result:
                op1 * op2
        else:
            assert op1 * op2 == result

        if op1 == 1:
            assert op1 * op2 is op2
        elif op2 == 1:
            assert op1 * op2 is op1

    @pytest.mark.parametrize(
        "loc,variable_map,result",
        [
            (LE("a"), dict(a=0), LE()),
            (LE("a"), dict(a=1), LE(1)),
            (LE(1), dict(a=1), LE(1)),
            (2 + LE("a"), dict(a=1), LE(3)),
            (2 + LE("a"), dict(b=1), LE(2, {(LE("a"), 1)})),
            (LE(1, {(LE(1, {(LE("a"), 2), (LE("b"), 1)}), 2)}), dict(a=1, b=1), LE(9)),
            (
                LE(1, {(LE(1, {(LE("a"), 2), (LE("b"), 1)}), 2)}),
                dict(b=1),
                LE(5, {(LE("a"), 4)}),
            ),
            (1 + LE("a"), dict(a="b"), 1 + LE("b")),
            (1 + LE("a"), dict(a=LE(5, {(LE("b"), 2)})), 6 + 2 * LE("b")),
        ],
    )
    def test_resolve(self, loc, variable_map, result):
        assert loc.resolve(**variable_map) == result

    @pytest.mark.parametrize(
        "loc,expect",
        [
            (LE(), True),
            (LE("a"), False),
        ],
    )
    def test_resolved(self, loc, expect):
        assert loc.resolved == expect

    @pytest.mark.parametrize(
        "loc,result,return_string",
        [
            (LE(), set(), False),
            (LE(), set(), True),
            (LE("a"), {LE("a")}, False),
            (LE("a"), {"a"}, True),
            (LE(1) + "a" + "b", {"a", "b"}, True),
            (LE(1, {("a", 2), (LE(2, {("b", 1)}), 1)}), {"a", "b"}, True),
        ],
    )
    def test_variables(self, loc, result, return_string):
        assert loc.variables(return_string) == result

    @pytest.mark.parametrize(
        "loc,variable,expect",
        [
            (LE(), "start", False),
            (LE("start"), "start", True),
            (LE(2, {(LE(3, {(LE("b"), 1)}), 2), (LE("a"), 0.5)}), "a", True),
            (LE(2, {(LE(3, {(LE("b"), 1)}), 2), (LE("a"), 0.5)}), "b", True),
            (LE(2, {(LE(3, {(LE("b"), 1)}), 2), (LE("a"), 0.5)}), "c", False),
        ],
    )
    def test_contains(self, loc, variable, expect):
        assert (variable in loc) is expect

    @pytest.mark.parametrize(
        "exp1,exp2,op,expect",
        [
            (LE(), 1, "lt", True),
            (LE(), 0, "ge", True),
            (LE(), 0, "le", True),
            (LE(), 0, "gt", False),
            (LE(), LE(3, {(LE(2), -3)}), "lt", False),
            (LE(), "var", "gt", pytest.raises(ValueError)),
        ],
    )
    def test_compariosn(self, exp1, exp2, op, expect):
        import operator

        op = getattr(operator, op)

        if hasattr(expect, "__enter__"):
            with expect:
                op(exp1, exp2)

        else:
            assert op(exp1, exp2) == expect

    @pytest.mark.parametrize("loc", [LE(), LE("a"), LE("a") + LE("b")])
    def test_copy(self, loc):
        assert copy(loc) is loc
        assert deepcopy(loc) is loc

    STR_EXPR_PAIRS = [
        ("0", LE()),
        ("1.2", LE(1.2)),
        ("a", LE("a")),
        ("+1", LE(1)),
        ("-a", -LE("a")),
        ("a + 1", 1 + LE("a")),
        ("a - 5*b", LE("a") - 5 * LE("b")),
        ("1 + 5 * (a + b + 2 * c)", 1 + 5 * (LE("a") + LE("b") + 2 * LE("c"))),
        ("a + ", pytest.raises(ValueError)),
        ("a * b", pytest.raises(ValueError)),
        (1, pytest.raises(TypeError)),
    ]

    @pytest.mark.parametrize("s,loc_or_error", STR_EXPR_PAIRS)
    def test_from_string(self, s, loc_or_error):
        context = noerror()
        if hasattr(loc_or_error, "__enter__"):
            context = loc_or_error

        with context:
            assert LE.from_string(s) == loc_or_error

    EXPR_STR_PAIRS = [
        (LE(), "0"),
        (LE("a"), "a"),
        (LE(1.2), "1.2"),
        (LE("a") + 1.2, "1.2 + a"),
        (LE("a") - 1.2, "-1.2 + a"),
        (1 - 1 * (LE("a") + (LE("b"))), ("1 - a - b", "1 - b - a")),
        (LE(1, references={(2 + LE("a"), -1)}), "1 - (2 + a)"),
        (LE(1, references={(3 + LE("a"), -2)}), "1 - 2 * (3 + a)"),
    ]

    @pytest.mark.parametrize("loc,expect", EXPR_STR_PAIRS)
    def test_str_repr(self, loc, expect):
        assert str(loc) in expect
        assert repr(loc) == f"LinearExpression({loc})"

    @pytest.mark.parametrize(
        "loc",
        it.chain(
            (exp for exp, _ in EXPR_STR_PAIRS),
            (exp for _, exp in STR_EXPR_PAIRS if isinstance(exp, LinearExpression)),
        ),
    )
    def test_serialization(self, loc):
        unstructured = qwip.converter.unstructure(loc)
        structured = qwip.converter.structure(unstructured, LinearExpression)

        assert loc.resolve() == structured

    @pytest.mark.parametrize(
        "to_structure,linexp",
        [
            ("a", LE("a")),
            (dict(offset=0), LE()),
            (dict(offset=1, references={("a", 1)}), 1 + LE("a")),
        ],
    )
    def test_structure_dict(self, to_structure, linexp):
        assert qwip.converter.structure(to_structure, LinearExpression) == linexp
