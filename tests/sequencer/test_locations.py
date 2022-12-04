import pytest
import attrs

import qwip
from qwip.sequencer.locations import Location

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

    def test_repr(self):
        l0 = Location()
        l1 = Location(5)
        l2 = Location(10)

        l3 = Location(1, references={(l1, 0.5), (l2, 1)})
