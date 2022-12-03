import pytest

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

    def test_negation(self):
        assert Location() == -Location()
        assert Location(5) == -Location(-5)

        l1 = Location(
            references={(Location(1), 1), (Location(2), 0.5)}
        )

        l1neg = Location(
            references={(Location(1), -1), (Location(2), -0.5)}
        )
        assert l1 == -l1neg
        assert -l1 == l1neg


    def test_add(self):
        assert Location() + Location() == Location()

        l1 = Location(

        )

    def test_repr(self):
        l0 = Location()
        l1 = Location(5)
        l2 = Location(10)

        l3 = Location(1, references={(l1, 0.5), (l2, 1)})

        print(l0)
        print(l1)
        print(l2)
        print(l3)
