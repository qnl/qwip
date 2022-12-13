import pytest
import numpy as np

from contextlib import nullcontext

from qwip.sequencer.utils import Location
from qwip.sequencer.elements import (
    SequenceElement,
    WaveformCompiler
)
from qwip.sequencer.waveform import (
    VirtualZWaveform,
    ModulationFrequency,
    ModulatedWaveform,
    SquareWaveform
)

class TestSequenceElement:
    def test_create(self):
        se = SequenceElement()
        assert se.locations == dict()
        assert se.constraints == dict()

    @pytest.mark.parametrize(
        'locations,start',
        [
            ([0, 5, 10], None),
            (['start', 'end'], None),
            ([10, 12], Location(-10))
        ]
    )
    def test_fromtuples(self, locations, start):
        s = SquareWaveform()
        constraints = dict(start=start) if start else dict()

        se = SequenceElement.fromtuples(
            [(l, s) for l in locations],
            **constraints
        )

        variables= {v for v in locations if isinstance(v, str)} | set(constraints)
        assert se.variables() == variables
        
        if start:
            assert se.constraints['start'] == start

    def test_add_waveform(self):
        s = SquareWaveform()

        se = SequenceElement()



    @pytest.mark.parametrize(
        'locations,constraints,expect',
        [
            ([], {}, set()),
            ([0, 'a', 'b'], {}, {'a', 'b'}),
            ([0, 1, 2], {}, set()),
            ([], dict(x='y'), {'x', 'y'})
        ]
    )
    def test_variables(self, locations, constraints, expect):
        s = SquareWaveform()
        
        se = SequenceElement.fromtuples(
            [(l, s) for l in locations],
            **constraints
        )

        assert se.variables() == expect

    @pytest.mark.parametrize(
        'locations,constraints,expect',
        [
            (['start'], dict(start=0), dict(start=0)),
            (
                ['start', 'a', 'b', 'c'],
                dict(
                    start=0,
                    a=Location('start') + 10,
                    b=Location('start') + 20,
                    c=0.5*(Location('a') + Location('b'))
                ),
                dict(start=0, a=10, b=20, c=15)
            ),
            (
                ['a', 'b', 'c'],
                dict(
                    a=0.5*Location('b') - 'c',
                    b='a' + 2*Location('c') + 1,
                    c=3*Location('a') + 2*Location('b') - 1
                ),
                dict(a=1, b=-2, c=-2)
            ),
            (
                ['start', 'underconstrained'],
                dict(start=0),
                pytest.raises(np.linalg.LinAlgError)
            ),
            (
                ['start', 'end'],
                dict(
                    start='end' + Location(1),
                    end='start' - Location(1)
                ),
                pytest.raises(np.linalg.LinAlgError)
            )
        ]
    )
    def test_solve_constraints(self, locations, constraints, expect):
        basis_set = {
            loc if isinstance(loc, Location) else Location(loc): i
                for i, loc in enumerate(locations)
        }

        constraints = {
            k: loc if isinstance(loc, Location) else Location(loc)
                for k, loc in constraints.items()
        }

        context = expect if hasattr(expect, '__enter__') else nullcontext()
        with context:
            result = SequenceElement._solve_constraint_matrix(
                basis_set, constraints
            )
        
            assert result.keys() == expect.keys()
            assert all(
                np.allclose(result[k], expect[k]) 
                    for k in result.keys()
            )

    def test_solve_timings(self):
        ...

    def test_append_sequence(self):
        se1 = SequenceElement.fromtuples([('a', None), ('b', None)])
        se2 = SequenceElement.fromtuples([('c', None), ('d', None)])

        se1.append(se2)

        assert se1.locations == {Location(l): [] for l in 'abcd'}

    @pytest.mark.parametrize(
        'vars1,vars2,name,shared,expect',
        [
            (['a', 'b', 'c'], ['b', 'c', 'd'], None, set(), pytest.raises(ValueError)),
            (['a', 'b', 'c'], ['b', 'c', 'd'], None, set('bc'), set('abcd')),

        ]
    )
    def test_append_shared_variables(self, vars1, vars2, name, shared, expect):
        se1 = SequenceElement.fromtuples([(v, None) for v in vars1])
        se2 = SequenceElement.fromtuples([(v, None) for v in vars2])

        if hasattr(expect, '__enter__'):
            with expect:
                se1.append(se2, name=name, shared=shared)
        else:
            se1.append(se2, name=name, shared=shared)
            se1.variables() == expect

    @pytest.mark.parametrize(
        'self_loc,other_loc,name',
        [
            (Location(), Location(), 'seq2/start'),
            (Location(), Location(5), 'seq2/start'),
            (Location(5), Location(), 'seq2/start'),
            (Location(), Location(), None)
        ]
    )
    def test_append_location_name(self, self_loc, other_loc, name):
        se1 = SequenceElement()
        se2 = SequenceElement()

        se1.append(se2, self_loc, other_loc, name=name)

        if name:
            assert se1.constraints[name] == self_loc - other_loc
        else:
            assert se1.constraints == dict()
