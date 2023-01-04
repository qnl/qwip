import pytest
import numpy as np

from contextlib import nullcontext as noerror

import qwip
from qwip.testing import ignore_order
from qwip.sequencer.utils import Location
from qwip.sequencer.elements import (
    SequenceElement,
)
from qwip.sequencer.waveform import (
    VirtualZWaveform,
    ModulationFrequency,
    ModulatedWaveform,
    GaussianWaveform,
    SquareWaveform,
    CosineRampWaveform,
    Channel
)

WAVEFORMS = dict(
    g1=GaussianWaveform(width=32e-9, amplitude=1, channels={'I'}),
    g2=GaussianWaveform(width=32e-9, amplitude=0.5, channels={'Q'}),
    s1=SquareWaveform(width=32e-9, amplitude=1, channels={'I', 'Q'}),
    c1=CosineRampWaveform(width=32e-9, amplitude=1, channels={'F1'})
)

class TestSequenceElement:
    def test_create(self):
        se = SequenceElement()
        assert se.locations == dict()
        assert se.constraints == dict()

        se = SequenceElement(
            locations=dict(start=[WAVEFORMS['g1']]),
            constraints=dict(start=Location()),
        )

        assert se.locations == {Location('start'): [WAVEFORMS['g1']]}
        assert se.constraints == dict(start=Location())
        assert se.channels == set({Channel('I')})

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

    @pytest.mark.parametrize(
        'all_locs,get_loc,expect',
        [
            (('a', 'b', 'c'), 'b', 'g2'),
            (('a', 'b', 'c'), 1, pytest.raises(KeyError)),
            (tuple(), 'a', pytest.raises(KeyError))
        ]
    )
    def test_getitem(self, all_locs, get_loc, expect):
        se = SequenceElement.fromtuples(
            [(l, w) for l, w in zip(all_locs, WAVEFORMS.values())]
        )

        context = noerror() if isinstance(expect, str) else expect
        with context:
            assert se[get_loc] == [WAVEFORMS[expect]]

    @pytest.mark.parametrize(
        'pairs,wave,expect',
        [
            (
                [(0, WAVEFORMS['s1']), (0, (WAVEFORMS['g2']))],
                WAVEFORMS['s1'],
                True
            ),
            (
                [(0, WAVEFORMS['s1']), (0, (WAVEFORMS['g2']))],
                WAVEFORMS['g1'],
                False
            )
        ]
    )
    def test_contains(self, pairs, wave, expect):
        se = SequenceElement.fromtuples(pairs)
        assert (wave in se) == expect

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
            ),
            (['a'], dict(a='b', b='c', c=1), dict(a=1, b=1, c=1))
        ]
    )
    def test_solve_constraints(self, locations, constraints, expect):
        se = SequenceElement()
        for loc in locations:
            se.add_waveform([], loc) 

        se.add_constraints(**constraints)

        context = expect if hasattr(expect, '__enter__') else noerror()
        with context:
            result = se.solve_constraints()
        
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

    @pytest.mark.parametrize(
        'se1,se2,result',
        [
            (SequenceElement(), SequenceElement(), SequenceElement()),
            (
                SequenceElement.fromtuples(
                    [
                        (Location('start'), SquareWaveform(channels=['a']))
                    ],
                    start=Location()
                ),
                SequenceElement.fromtuples(
                    [
                        (Location('start'), SquareWaveform(channels=['b']))
                    ],
                    start=Location(),
                    width=Location(10)
                ),
                SequenceElement.fromtuples(
                    [
                        (Location('start'), SquareWaveform(channels=['a'])),
                        (Location('start'), SquareWaveform(channels=['b'])),
                    ],
                    start=Location(),
                    width=Location(10)
                )
            ),
            (
                SequenceElement.fromtuples([], start=Location()),
                SequenceElement.fromtuples([], start=Location(1)),
                pytest.raises(ValueError)
            )
        ]
    )
    def test_add(self, se1, se2, result):
        context = result if hasattr(result, '__enter__') else noerror()
        with context:
            assert se1 + se2 == result

    @pytest.mark.parametrize(
        'se',
        [
            SequenceElement(),
            SequenceElement.fromtuples(
                [('a', WAVEFORMS['c1']), ('b', WAVEFORMS['g1'])]
            ),
            SequenceElement.fromtuples(
                [('a', WAVEFORMS['c1']), ('a', WAVEFORMS['g1'])],
                a=Location()
            )
        ]
    )
    def test_deep_copy(self, se):
        secopy = se.copy()

        assert se == secopy
        assert se.locations is not secopy.locations
        assert se.constraints is not secopy.constraints
        assert se.channels is not secopy.channels

        for loc in se.locations:
            assert se[loc] is not secopy[loc]

    @pytest.mark.parametrize(
        'waveforms,channels,channel_map',
        [
            (
                ['c1', 's1'],
                [],
                dict(
                    I=[(Location(1), WAVEFORMS['s1'])],
                    Q=[(Location(1), WAVEFORMS['s1'])],
                    F1=[(Location(0), WAVEFORMS['c1'])],
                )
            ),
            (
                ['c1', 'g2', 's1'],
                ['Q'],
                dict(
                    Q=[
                        (Location(1), WAVEFORMS['g2']),
                        (Location(2), WAVEFORMS['s1'])
                    ]
                )
            )
        ]
    )
    def test_get_channel_map(self, waveforms, channels, channel_map):
        
        se = SequenceElement.fromtuples(
            [(i, WAVEFORMS[w]) for i, w in enumerate(waveforms)]
        )

        channel_map = {Channel(c): waves for c, waves in channel_map.items()}
        assert se.get_channel_map(*channels) == channel_map

    @pytest.mark.parametrize(
        'se,se_dict',
        [
            (SequenceElement(), {}),
            (
                SequenceElement.fromtuples(
                    [(1 + Location('width'), WAVEFORMS['s1'])],
                    width=Location(5)
                ),
                dict(
                    locations={
                        '1 + width': [
                            {
                                'channels': ignore_order([{'name': 'I'}, {'name': 'Q'}]),
                                'width': 3.2e-8,
                                '__class__': 'SquareWaveform'
                            }
                        ]
                    },
                    constraints=dict(width=5.0)
                )
            )
        ]
    )
    def test_serialization(self, se, se_dict):

        unstructured = qwip.converter.unstructure(se)
        restructured = qwip.converter.structure(unstructured, SequenceElement)
        
        assert unstructured == se_dict
        assert se == restructured
    