from contextlib import nullcontext as noerror

import numpy as np
import pytest
import sympy as sym

import qwip
from qwip.sequencer.phase_tracker import Frame
from qwip.sequencer.timeline import Timeline
from qwip.sequencer.utils import Location
from qwip.sequencer.waveform import (
    CosineRampWaveform,
    CWWaveform,
    DCWaveform,
    GaussianWaveform,
    Marker,
    ModulatedWaveform,
    SquareWaveform,
    VirtualZWaveform,
    Waveform,
)
from qwip.testing import ignore_order

WAVEFORMS = dict(
    g1=GaussianWaveform(width=32e-9, amplitude=1, channels={"I"}),
    g2=GaussianWaveform(width=32e-9, amplitude=0.5, channels={"Q"}),
    s1=SquareWaveform(width=32e-9, amplitude=1, channels={"I", "Q"}),
    c1=CosineRampWaveform(width=32e-9, amplitude=1, channels={"F1"}),
)


class TestTimeline:
    def test_create(self):
        tmln = Timeline()
        assert tmln.lw_pairs == []
        assert tmln.constraints == []
        assert tmln.channels == set()
        assert tmln.width is None

        tmln = Timeline(
            lw_pairs=[("start", WAVEFORMS["g1"])],
            constraints=["start"],
            width="width",
        )

        assert tmln.lw_pairs == [(sym.Symbol("start"), WAVEFORMS["g1"])]
        assert tmln.constraints == [sym.Symbol("start")]
        assert tmln.channels == set({"I"})
        assert tmln.width == sym.Symbol("width")

    @pytest.mark.parametrize(
        "locations,constraints",
        [
            ([0, 5, 5], ["start"]),
            (["start", "end"], ["start + 5 - end"]),
            ([10, 12], ["start - 10"]),
        ],
    )
    def test_fromtuples(self, locations, constraints):
        wave = SquareWaveform(channels=("CH1",))

        tmln = Timeline.fromtuples(
            [(loc, wave) for loc in locations], constraints=constraints
        )

        assert tmln.channels == {"CH1"}

        for loc in tmln.locations:
            assert isinstance(loc, sym.Expr)

        for constraint in tmln.constraints:
            assert isinstance(constraint, sym.Expr)

    @pytest.mark.parametrize(
        "layers,expect",
        [
            ([], Timeline()),
            (
                [
                    SquareWaveform(width=5e-9),
                    [SquareWaveform(width=10e-9), SquareWaveform(width=20e-9)],
                    SquareWaveform(width=10e-9),
                ],
                Timeline.fromtuples(
                    [
                        (0, SquareWaveform(width=5e-9)),
                        (5e-9, SquareWaveform(width=10e-9)),
                        (5e-9, SquareWaveform(width=20e-9)),
                        (25e-9, SquareWaveform(width=10e-9)),
                    ]
                ),
            ),
            (
                [
                    [SquareWaveform(width=10e-9), 20e-9],
                    SquareWaveform(width=10e-9),
                ],
                Timeline.fromtuples(
                    [
                        (0, SquareWaveform(width=10e-9)),
                        (20e-9, SquareWaveform(width=10e-9)),
                    ]
                ),
            ),
            (
                [
                    Timeline.fromtuples(
                        [
                            (0, VirtualZWaveform(frame="Q0")),
                            (0, SquareWaveform(width=25e-9)),
                            (25e-9, VirtualZWaveform(frame="Q0")),
                        ],
                        width=25e-9,
                    ),
                    "delay",
                    [SquareWaveform(width=10e-9)],
                ],
                Timeline.fromtuples(
                    [
                        (0, VirtualZWaveform(frame="Q0")),
                        (0, SquareWaveform(width=25e-9)),
                        (25e-9, VirtualZWaveform(frame="Q0")),
                        ("25e-9 + delay", SquareWaveform(width=10e-9)),
                    ]
                ),
            ),
            (
                [
                    (20e-9, SquareWaveform(width=10e-9)),
                    "delay",
                    SquareWaveform(width=5e-9),
                ],
                Timeline.fromtuples(
                    [
                        (0, SquareWaveform(width=10e-9)),
                        ("20e-9 + delay", SquareWaveform(width=5e-9)),
                    ]
                ),
            ),
        ],
    )
    def test_from_layers(self, layers, expect):
        tmln = Timeline.from_layers(layers)

        assert tmln == expect

    def test_from_layers_start_time(self):
        tmln = Timeline.from_layers([SquareWaveform(width=20)], t0="start_delay")

        expect = Timeline.fromtuples([("start_delay", SquareWaveform(width=20))])

        assert tmln == expect

    @pytest.mark.parametrize(
        "all_locs,get_loc,expect",
        [
            (("a", "b", "c"), "b", "g2"),
            (("a", "b", "c"), 1, pytest.raises(KeyError)),
            (tuple(), "a", pytest.raises(KeyError)),
        ],
    )
    def test_getitem(self, all_locs, get_loc, expect):
        tmln = Timeline(lw_pairs=[(l, w) for l, w in zip(all_locs, WAVEFORMS.values())])

        context = noerror() if isinstance(expect, str) else expect
        with context:
            assert tmln[get_loc] == [WAVEFORMS[expect]]

    @pytest.mark.parametrize(
        "pairs,wave,expect",
        [
            ([(0, WAVEFORMS["s1"]), (0, (WAVEFORMS["g2"]))], WAVEFORMS["s1"], True),
            ([(0, WAVEFORMS["s1"]), (0, (WAVEFORMS["g2"]))], WAVEFORMS["g1"], False),
        ],
    )
    def test_contains(self, pairs, wave, expect):
        tmln = Timeline(lw_pairs=pairs)
        assert (wave in tmln) == expect

    @pytest.mark.parametrize(
        "locations,constraints,expect",
        [
            ([], [], set()),
            ([0, "a", "b"], [], {"a", "b"}),
            ([0, 1, 2], [], set()),
            ([], ["x - y"], {"x", "y"}),
        ],
    )
    def test_variables(self, locations, constraints, expect):
        s = SquareWaveform()

        tmln = Timeline.fromtuples([(l, s) for l in locations], constraints=constraints)

        assert tmln.variables() == expect

    @pytest.mark.parametrize(
        "locations,constraints,expect",
        [
            (["start"], ["start"], dict(start=0)),
            (
                ["start", "a", "b", "c"],
                ["start", "start + 10 - a", "start + 20 - b", "(a + b) / 2 - c"],
                dict(start=0, a=10, b=20, c=15),
            ),
            (
                ["a", "b", "c"],
                ["b / 2 - c - a", "a + 2 * c + 1 - b", "3 * a + 2 * b - 1"],
                dict(a=1, b=-2, c=-2),
            ),
            (
                ["start", "underconstrained"],
                ["start"],
                pytest.raises(np.linalg.LinAlgError),
            ),
            (
                ["start", "end"],
                ["end + 1 - start", "start - 1 - end"],
                pytest.raises(np.linalg.LinAlgError),
            ),
            (["a"], ["a - b", "b - c", "c - 1"], dict(a=1, b=1, c=1)),
        ],
    )
    def test_solve_constraints(self, locations, constraints, expect):
        m = Marker(name="m1")
        tmln = Timeline(
            lw_pairs=[(loc, m) for loc in locations], constraints=constraints
        )

        context = expect if hasattr(expect, "__enter__") else noerror()
        with context:
            result = tmln.solve_constraints()

            assert result.keys() == expect.keys()
            assert all(np.allclose(result[k], expect[k]) for k in result.keys())

    def test_resolve_locations_negative(self):
        tmln = Timeline.fromtuples(
            [(-20e-9, SquareWaveform(width=30e-9)), (0, GaussianWaveform(width=20e-9))]
        )

        locations = tmln.resolve_locations()

        assert list(locations.keys()) == [Location(0), Location(20e-9), Location(40e-9)]

    def test_resolve_locations_infinite(self):
        tmln = Timeline.fromtuples(
            [
                (-10e-9, DCWaveform()),
                (0, GaussianWaveform(width=20e-9)),
                (10e-9, GaussianWaveform(width=30e-9)),
            ]
        )

        locations = tmln.resolve_locations()

        assert list(locations.keys()) == [Location(t) for t in (0, 10e-9, 20e-9, 50e-9)]

    @pytest.mark.parametrize(
        "locs,end",
        [
            ([(10e-9, GaussianWaveform(width=20e-9))], Location(30e-9)),
            ([(-10e-9, GaussianWaveform(width=20e-9))], Location(20e-9)),
        ],
    )
    def test_resolve_locations_marker(self, locs, end):
        tmln = Timeline.fromtuples(locs)

        markers = {}
        tmln.resolve_locations(markers=markers)

        assert markers["end"].almost_equal(end)

    def test_rename_variables(self):
        tmln = Timeline().fromtuples(
            [
                (0, SquareWaveform(amplitude="amp", width="t")),
                ("t", GaussianWaveform(width="t")),
            ]
        )

        tmln.rename_variables(lambda n: "tgate" if n == "t" else n)

        assert tmln == Timeline().fromtuples(
            [
                (0, SquareWaveform(amplitude="amp", width="tgate")),
                ("tgate", GaussianWaveform(width="tgate")),
            ]
        )

        tmln.width = "tgate + tbuffer"
        tmln.rename_variables(lambda n: n + "1" if n != "tgate" else n)

        assert tmln == Timeline().fromtuples(
            [
                (0, SquareWaveform(amplitude="amp1", width="tgate")),
                ("tgate", GaussianWaveform(width="tgate")),
            ],
            width="tgate + tbuffer1",
        )

    def test_add_timeline(self):
        tmln1 = Timeline.fromtuples([("a", None), ("b", None)])
        tmln2 = Timeline.fromtuples([("c", None), ("d", None)])

        tmln1.add_timeline(tmln2)

        assert tmln1.locations == {Location(l): [] for l in "abcd"}

    @pytest.mark.parametrize(
        "vars1,vars2,name,expect",
        [
            (["a", "b", "c"], ["b", "c", "d"], None, set("abcd")),
        ],
    )
    def test_add_timeline_shared_variables(self, vars1, vars2, name, expect):
        tmln1 = Timeline.fromtuples([(v, None) for v in vars1])
        tmln2 = Timeline.fromtuples([(v, None) for v in vars2])

        if hasattr(expect, "__enter__"):
            with expect:
                tmln1.add_timeline(tmln2, name=name)
        else:
            tmln1.add_timeline(tmln2, name=name)
            tmln1.variables() == expect

    @pytest.mark.parametrize(
        "self_loc,other_loc,name",
        [
            (Location(), Location(), "seq2/start"),
            (Location(), Location(5), "seq2/start"),
            (Location(5), Location(), "seq2/start"),
            (Location(), Location(), None),
        ],
    )
    def test_add_timeline_location_name(self, self_loc, other_loc, name):
        tmln1 = Timeline()
        tmln2 = Timeline()

        tmln1.add_timeline(tmln2, self_loc, other_loc, name=name)

        if name:
            assert tmln1.constraints[name] == self_loc - other_loc
        else:
            assert tmln1.constraints == dict()

    @pytest.mark.parametrize(
        "tmln,target,result",
        [
            (Timeline(), Timeline(), Timeline()),
            (
                Timeline(),
                SquareWaveform(),
                Timeline.fromtuples([(Location(), SquareWaveform())]),
            ),
            (Timeline(), t := Timeline.fromtuples([(Location(), SquareWaveform())]), t),
        ],
    )
    def test_add(self, tmln, target, result):
        assert tmln.add(target) == result

    @pytest.mark.parametrize(
        "tmln1,tmln2,result",
        [
            (Timeline(), Timeline(), Timeline()),
            (
                Timeline.fromtuples(
                    [(Location("start"), SquareWaveform(channels=["a"]))],
                    constraints=dict(start=Location()),
                ),
                Timeline.fromtuples(
                    [(Location("start"), SquareWaveform(channels=["b"]))],
                    constraints=dict(start=Location(), width=Location(10)),
                ),
                Timeline.fromtuples(
                    [
                        (Location("start"), SquareWaveform(channels=["a"])),
                        (Location("start"), SquareWaveform(channels=["b"])),
                    ],
                    constraints=dict(start=Location(), width=Location(10)),
                ),
            ),
            (
                Timeline.fromtuples([], constraints=dict(start=Location())),
                Timeline.fromtuples([], constraints=dict(start=Location(1))),
                pytest.raises(ValueError),
            ),
        ],
    )
    def test_add_operator(self, tmln1, tmln2, result):
        context = result if hasattr(result, "__enter__") else noerror()
        with context:
            assert tmln1 + tmln2 == result

    def test_transform_waveforms(self):
        lws = [
            (Location(), VirtualZWaveform(frame="mod_Q0_GE", phase="zphase")),
            (
                Location(),
                ModulatedWaveform(
                    envelope=GaussianWaveform(amplitude="amplitude", width="width"),
                    modulation=CWWaveform(
                        frequency="mod_Q0_GE", channels=("Q0_I", "Q0_Q")
                    ),
                ),
            ),
            (Location("width"), VirtualZWaveform(frame="mod_Q0_GE", phase="zphase")),
        ]
        tmln = Timeline.fromtuples(lws, width="width")

        def transformer(loc, wave):
            match wave:
                case VirtualZWaveform():
                    new_wave = wave.evolve(frame="Q0.mod_GE")
                case ModulatedWaveform():
                    new_wave = wave.evolve(modulation_frequency="Q0.mod_GE")
                case _:
                    new_wave = wave

            return new_wave

        assert tmln.transform_waveforms(transformer) == 3

        for _, wave in tmln.get_location_pairs():
            match wave:
                case VirtualZWaveform():
                    assert wave.frame == Frame("Q0.mod_GE")
                case ModulatedWaveform():
                    assert wave.modulation.frequency == Frame("Q0.mod_GE")

    @pytest.mark.parametrize(
        "tmln",
        [
            Timeline(),
            Timeline.fromtuples([("a", WAVEFORMS["c1"]), ("b", WAVEFORMS["g1"])]),
            Timeline.fromtuples(
                [("a", WAVEFORMS["c1"]), ("a", WAVEFORMS["g1"])],
                constraints=dict(a=Location()),
            ),
        ],
    )
    def test_deep_copy(self, tmln):
        tmlncopy = tmln.copy()

        assert tmln == tmlncopy
        assert tmln.locations is not tmlncopy.locations
        assert tmln.constraints is not tmlncopy.constraints
        assert tmln.channels is not tmlncopy.channels

        for loc in tmln.locations:
            assert tmln[loc] is not tmlncopy[loc]

    @pytest.mark.parametrize(
        "waveforms,channels,channel_map",
        [
            (
                ["c1", "s1"],
                [],
                dict(
                    I=[(Location(1), WAVEFORMS["s1"])],
                    Q=[(Location(1), WAVEFORMS["s1"])],
                    F1=[(Location(0), WAVEFORMS["c1"])],
                ),
            ),
            (
                ["c1", "g2", "s1"],
                ["Q"],
                dict(
                    Q=[(Location(1), WAVEFORMS["g2"]), (Location(2), WAVEFORMS["s1"])]
                ),
            ),
        ],
    )
    def test_get_channel_map(self, waveforms, channels, channel_map):
        tmln = Timeline.fromtuples([(i, WAVEFORMS[w]) for i, w in enumerate(waveforms)])

        channel_map = {c: waves for c, waves in channel_map.items()}
        assert tmln.get_channel_map(*channels) == channel_map

    @pytest.mark.parametrize(
        "tmln,tmln_dict",
        [
            (Timeline(), {}),
            (
                Timeline.fromtuples(
                    [(1 + Location("width"), WAVEFORMS["s1"])],
                    constraints=dict(width=Location(5)),
                ),
                dict(
                    locations={
                        "1 + width": [
                            {
                                "channels": ignore_order(["I", "Q"]),
                                "width": 3.2e-8,
                                "__class__": "SquareWaveform",
                            }
                        ]
                    },
                    constraints=dict(width=5.0),
                ),
            ),
        ],
    )
    def test_serialization(self, tmln, tmln_dict):
        unstructured = qwip.converter.unstructure(tmln)
        restructured = qwip.converter.structure(unstructured, Timeline)

        assert unstructured == tmln_dict
        assert tmln == restructured
