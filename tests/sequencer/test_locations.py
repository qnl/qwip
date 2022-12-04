import pytest
import attrs

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

    @pytest.mark.parametrize(
        'location,inverse',
        [
            (Location(), Location()),
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
            ('5', Location(), pytest.raises(TypeError)),
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
            (Location(), Location(), Location()),
            (Location(1), Location(1), Location()),
            (Location(), Location(5), Location(-5)),
            (Location(), 5, Location(-5)),
            (5, Location(), Location(5)),
            ('5', Location(), pytest.raises(TypeError)),
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
            (Location(1), 0, Location()),
            (0, Location(1, {(Location(5), 1)}), Location()),
            (Location(1, {(Location(2), 1)}), 0.5, Location(0.5, {(Location(2), 0.5)})),
            ('5', Location(), pytest.raises(TypeError)),
        ]
    )
    def test_scalar_multiply(self, op1, op2, result):
        if hasattr(result, '__enter__'):
            with result:
                op1 * op2
        else:
            assert op1 * op2 == result

    def test_repr(self):
        l0 = Location()
        l1 = Location(5)
        l2 = Location(10)

        l3 = Location(1, references={(l1, 0.5), (l2, 1)})
