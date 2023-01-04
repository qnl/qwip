import pytest
import attrs

import itertools as it
from contextlib import nullcontext as noerror

from copy import copy, deepcopy

import qwip
from qwip.sequencer.utils import LinearExpression, Location

class TestLocation:
    def test_create(self):
        loc = Location()

        assert loc.offset == 0 and loc.references == frozenset()

        l1 = Location(-1)
        l2 = Location(0)
        l3 = Location(1.5)

        assert l1.offset == -1 and l1.references == frozenset()
        assert l2.offset == 0 and l2.references == frozenset()
        assert l3.offset == 1.5 and l3.references == frozenset()

        l4 = Location(references={(l1, 0.5), (l2, 0.5), (l3, 0.5)})
        assert l4.references == frozenset({(l1, 0.5), (l2, 0.5), (l3, 0.5)})

    def test_create_str(self):
        loc = Location('a')
        assert loc.offset == 'a' and loc.references == frozenset()

        loc = Location(0, {('a', -1)})
        assert loc.offset == 0 and loc.references == frozenset({(Location('a'), -1)})

        with pytest.raises(ValueError):
            loc = Location('a', {(Location(), 1)})

    @pytest.mark.parametrize(
        'to_structure,location',
        [
            ('a', Location('a')),
            (dict(offset=0), Location()),
            (dict(offset=1, references={('a', 1)}), 1 + Location('a'))
        ]
    )
    def test_structure_(self, to_structure, location):
        assert qwip.converter.structure(to_structure, Location) == location

    @pytest.mark.parametrize(
        'op1,op2,equal',
        [
            (Location(), Location(), True),
            (Location(1), Location(1), True),
            (Location(1), Location(-1), False),
            (Location(0, {(Location(1), 1)}), Location(0, {(Location(1), 1)}), True),
            (Location(0, {(Location(1), 1)}), Location(0, {(Location(-1), -1)}), False),
            (Location('a'), Location('b'), False),
            (Location('a'), Location('a'), True)
        ]
    )
    def test_equal(self, op1, op2, equal):
        assert (op1 == op2) is equal

    @pytest.mark.parametrize(
        'location,inverse',
        [
            (Location(), Location()),
            (Location('str'), Location(0, {(Location('str'), -1)})),
            (-Location('str'), Location('str')),
            (Location(1), Location(-1)),
            (Location(-1), Location(1)),
            (Location(-1, {(Location(1), 1)}), Location(1, {(Location(1), -1)}))
        ]
    )
    def test_negation(self, location, inverse):
        assert -location == inverse

    @pytest.mark.parametrize(
        'op1,op2,result',
        [
            (Location(), Location(), Location()),
            (Location(1), Location(-1), Location()),
            (Location(), Location(5), Location(5)),
            (Location(), 5, Location(5)),
            (5, Location(), Location(5)),
            ([], Location(), pytest.raises(TypeError)),
            (Location(1, {(Location(5), 1)}), Location(-1, {(Location(5), -1)}), Location())
        ]
    )
    def test_add(self, op1, op2, result):
        if hasattr(result, '__enter__'):
            with result:
                op1 + op2
        else:
            assert op1 + op2 == result

    @pytest.mark.parametrize(
        'op1,op2,result',
        [
            ('a', Location(), Location('a')),
            (Location(), Location('a'), Location('a')),
            (
                Location('a'),
                Location('b'),
                Location(0, {('a', 1), ('b', 1)})
            ),
            (Location('a'), Location(2), Location(2, {('a', 1)})),
            (Location(-1, {('a', 1)}), Location(1), Location('a'))
        ]
    )
    def test_add_str(self, op1, op2, result):
        if hasattr(result, '__enter__'):
            with result:
                op1 + op2
        else:
            assert op1 + op2 == result

    @pytest.mark.parametrize(
        'op1,op2,result',
        [
            (Location(), Location(), Location()),
            (Location(1), Location(1), Location()),
            (Location(), Location(5), Location(-5)),
            (Location(), 5, Location(-5)),
            (5, Location(), Location(5)),
            ([], Location(), pytest.raises(TypeError)),
            (Location(1, {(Location(5), 1)}), Location(1, {(Location(5), 1)}), Location())
        ]
    )
    def test_subtract(self, op1, op2, result):
        if hasattr(result, '__enter__'):
            with result:
                op1 - op2
        else:
            assert op1 - op2 == result

    @pytest.mark.parametrize(
        'op1,op2,result',
        [
            ('a', Location(), Location('a')),
            (Location(), Location('a'), Location(0, {(Location('a'), -1)})),
            (
                Location('a'),
                Location('b'),
                Location(0, {(Location('a'), 1), (Location('b'), -1)})
            ),
            (Location('a'), Location(2), Location(-2, {(Location('a'), 1)})),
            (Location(1, {(Location('a'), 1)}), Location(1), Location('a'))
        ]
    )
    def test_subtract_str(self, op1, op2, result):
        if hasattr(result, '__enter__'):
            with result:
                op1 - op2
        else:
            assert op1 - op2 == result

    @pytest.mark.parametrize(
        'op1,op2,result',
        [
            (Location(1), 0, Location()),
            (0, Location(1, {(Location(5), 1)}), Location()),
            (Location('a'), 0, Location()),
            (2, Location(0, {(Location('a'), 0.5)}), Location('a')),
            (Location('a'), 3, Location(0, {(Location('a'), 3)})),
            (Location(1, {(Location(2), 1)}), 0.5, Location(0.5, {(Location(2), 0.5)})),
            ([], Location(), pytest.raises(TypeError)),
        ]
    )
    def test_scalar_multiply(self, op1, op2, result):
        if hasattr(result, '__enter__'):
            with result:
                op1 * op2
        else:
            assert op1 * op2 == result
        
        if op1 == 1:
            assert op1 * op2 is op2
        elif op2 == 1:
            assert op1 * op2 is op1

    @pytest.mark.parametrize(
        'loc,variable_map,result',
        [
            (Location('a'), dict(a=0), Location()),
            (Location('a'), dict(a=1), Location(1)),
            (Location(1), dict(a=1), Location(1)),
            (2 + Location('a'), dict(a=1), Location(3)),
            (2 + Location('a'), dict(b=1), Location(2, {(Location('a'), 1)})),
            (
                Location(1, {(Location(1, {(Location('a'), 2), (Location('b'), 1)}), 2)}),
                dict(a=1, b=1),
                Location(9)
            ),
            (
                Location(1, {(Location(1, {(Location('a'), 2), (Location('b'), 1)}), 2)}),
                dict(b=1),
                Location(5, {(Location('a'), 4)})
            ),
            (1 + Location('a'), dict(a='b'), 1 + Location('b')),
            (
                1 + Location('a'),
                dict(a=Location(5, {(Location('b'), 2)})),
                6 + 2*Location('b')
            )
        ]
    )
    def test_resolve(self, loc, variable_map, result):
        assert loc.resolve(**variable_map) == result

    @pytest.mark.parametrize(
        'loc,expect',
        [
            (Location(), True),
            (Location('a'), False),
        ]
    )
    def test_resolved(self, loc, expect):
        assert loc.resolved == expect

    @pytest.mark.parametrize(
        'loc,result,return_string',
        [
            (Location(), set(), False),
            (Location(), set(), True),
            (Location('a'), {Location('a')}, False),
            (Location('a'), {'a'}, True),
            (Location(1) + 'a' + 'b', {'a', 'b'}, True),
            (
                Location(1, {('a', 2), (Location(2, {('b', 1)}), 1)}),
                {'a', 'b'},
                True
            )
        ]
    )
    def test_variables(self, loc, result, return_string):
        assert loc.variables(return_string) == result

    @pytest.mark.parametrize(
        'loc,variable,expect',
        [
            (Location(), 'start', False),
            (Location('start'), 'start', True),
            (
                Location(2, {(Location(3, {(Location('b'), 1)}), 2), (Location('a'), 0.5)}),
                'a',
                True
            ),
            (
                Location(2, {(Location(3, {(Location('b'), 1)}), 2), (Location('a'), 0.5)}),
                'b',
                True
            ),
            (
                Location(2, {(Location(3, {(Location('b'), 1)}), 2), (Location('a'), 0.5)}),
                'c',
                False
            ),
        ]
    )
    def test_contains(self, loc, variable, expect):
        assert (variable in loc) is expect

    @pytest.mark.parametrize(
        'exp1,exp2,op,expect',
        [
            (Location(), 1, 'lt', True),
            (Location(), 0, 'ge', True),
            (Location(), 0, 'le', True),
            (Location(), 0, 'gt', False),
            (Location(), Location(3, {(Location(2), -3)}), 'lt', False),
            (Location(), 'var', 'gt', pytest.raises(ValueError))
        ]
    )
    def test_compariosn(self, exp1, exp2, op, expect):
        import operator
        op = getattr(operator, op)

        if hasattr(expect, '__enter__'):
            with expect:
                op(exp1, exp2)

        else:
            assert op(exp1, exp2) == expect

    @pytest.mark.parametrize(
        'loc',
        [Location(), Location('a'), Location('a') + Location('b')]
    )
    def test_copy(self, loc):
        assert copy(loc) is loc
        assert deepcopy(loc) is loc

    STR_EXPR_PAIRS = [
        ('0', Location()),
        ('1.2', Location(1.2)),
        ('a', Location('a')),
        ('+1', Location(1)),
        ('-a', -Location('a')),
        ('a + 1', 1 + Location('a')),
        ('a - 5*b', Location('a') - 5 * Location('b')),
        ('1 + 5 * (a + b + 2 * c)', 1 + 5 * (Location('a') + Location('b') + 2 * Location('c'))),
        ('a + ', pytest.raises(ValueError)),
        ('a * b', pytest.raises(ValueError)),
        (1, pytest.raises(TypeError))
    ]

    @pytest.mark.parametrize('s,loc_or_error',STR_EXPR_PAIRS)
    def test_from_string(self, s, loc_or_error):
        context = noerror()
        if hasattr(loc_or_error, '__enter__'):
            context = loc_or_error

        with context:
            assert Location.from_string(s) == loc_or_error

    EXPR_STR_PAIRS = [
        (Location(), '0'),
        (Location('a'), 'a'),
        (Location(1.2), '1.2'),
        (Location('a') + 1.2, '1.2 + a'),
        (Location('a') - 1.2, '-1.2 + a'),
        (1 - 1 * (Location('a') + (Location('b'))), ('1 - a - b', '1 - b - a')),
        (Location(1, references={(2 + Location('a'), -1)}), '1 - (2 + a)'),
        (Location(1, references={(3 + Location('a'), -2)}), '1 - 2 * (3 + a)')
    ]

    @pytest.mark.parametrize('loc,expect', EXPR_STR_PAIRS)
    def test_str_repr(self, loc, expect):
        assert str(loc) in expect
        assert repr(loc) == f'Location({loc})'

    @pytest.mark.parametrize(
        'loc',
        it.chain(
            (exp for exp, _ in EXPR_STR_PAIRS),
            (exp for _, exp in STR_EXPR_PAIRS if isinstance(exp, LinearExpression))
        )
    )
    def test_serialization(self, loc):
        unstructured = qwip.converter.unstructure(loc)
        structured = qwip.converter.structure(unstructured, Location)

        assert loc.resolve() == structured
