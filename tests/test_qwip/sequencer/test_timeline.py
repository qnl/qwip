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
    g1=GaussianWaveform(width=32e-9, amplitude=1, channel="I"),
    g2=GaussianWaveform(width=32e-9, amplitude=0.5, channel="Q"),
    s1=SquareWaveform(width=32e-9, amplitude=1, channel="IQ"),
    c1=CosineRampWaveform(width=32e-9, amplitude=1, channel="F1"),
)


class TestTimeline:
    def test_create(self):
        tmln = Timeline()
        assert tmln.lw_pairs == []
        assert tmln.constraints == set()
        assert tmln.channels == set()
        assert tmln.width is None

        tmln = Timeline(
            lw_pairs=[("start", WAVEFORMS["g1"])],
            constraints=["start"],
            width="width",
        )

        assert tmln.lw_pairs == [(sym.Symbol("start"), WAVEFORMS["g1"])]
        assert tmln.constraints == {sym.Symbol("start")}
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
        wave = SquareWaveform(channel="CH1")

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

    def test_iter(self):
        markers = [Marker(name=letter) for letter in "abcde"]
        tmln = Timeline(lw_pairs=list(enumerate(markers)))

        assert list(tmln) == tmln.lw_pairs
        assert list(Timeline()) == []

    @pytest.mark.parametrize(
        "locations,op,constraints,width,expect",
        [
            ([], SquareWaveform(), [], None, set()),
            ([0, 1, 2], SquareWaveform(), [], None, set()),
            ([0, "a", "b"], SquareWaveform(), [], None, {"a", "b"}),
            ([], SquareWaveform(), ["x - y"], None, {"x", "y"}),
            (
                [0, 1, 2],
                SquareWaveform(width="width", amplitude="amp"),
                [],
                None,
                {"width", "amp"},
            ),
        ],
    )
    def test_variables(self, locations, op, constraints, width, expect):
        tmln = Timeline(
            lw_pairs=[(loc, op) for loc in locations],
            constraints=constraints,
            width=width,
        )

        assert tmln.variables() == expect

    @pytest.mark.parametrize(
        "constraints,substitutions,expect",
        [
            (["a + b / 2"], dict(), {sym.parse_expr("a + b / 2")}),
            ([], dict(a="-b/2"), {sym.parse_expr("a + b / 2")}),
            (
                ["a - b"],
                dict(a="b", c="d"),
                {sym.parse_expr("a - b"), sym.parse_expr("c - d")},
            ),
        ],
    )
    def test_add_constraints(self, constraints, substitutions, expect):
        tmln = Timeline()
        tmln.add_constraints(*constraints, **substitutions)

        assert tmln.constraints == expect

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
                ["b / 2 - c - a", "a + 2 * c + 1 - b", "3 * a + 2 * b - 1 - c"],
                dict(a=1, b=-2, c=-2),
            ),
            (
                ["start", "underconstrained"],
                ["start"],
                dict(start=0),
            ),
            (["a"], ["a - b", "b - c", "c - 1"], dict(a=1, b=1, c=1)),
        ],
    )
    def test_solve_constraints(self, locations, constraints, expect):
        m = Marker(name="m1")
        tmln = Timeline(
            lw_pairs=[(loc, m) for loc in locations], constraints=constraints
        )
        result = tmln.solve_constraints()

        for k in result:
            assert result[k] == expect.pop(k)

        assert expect == {}

    def test_solve_undersconstrained(self):
        m = Marker(name="m1")
        tmln = Timeline(
            lw_pairs=[("start", m), ("end", m)],
            constraints=["end + 1 - start", "start - 1 - end"],
        )

        result = tmln.solve_constraints()

        if "start" in result:
            assert result["start"] == sym.parse_expr("end + 1")
        else:
            assert result["end"] == sym.parse_expr("start - 1")

    def test_resolve_sort(self):
        def _sort_key(loc_op):
            loc, op = loc_op
            return (loc, op.name)

        markers = {letter: Marker(name=letter) for letter in "abcdefg"}

        tmln = Timeline(
            lw_pairs=[
                (2, markers["b"]),
                (3, markers["a"]),
                (0, markers["f"]),
                (0, markers["e"]),
            ]
        )

        lw_pairs = tmln.resolve(inplace=False, sort=True)
        assert lw_pairs == [
            (0, markers["f"]),
            (0, markers["e"]),
            (2, markers["b"]),
            (3, markers["a"]),
        ]

        lw_pairs = tmln.resolve(inplace=False, sort=_sort_key)
        assert lw_pairs == [
            (0, markers["e"]),
            (0, markers["f"]),
            (2, markers["b"]),
            (3, markers["a"]),
        ]

    def test_resolve_negative(self):
        tmln = Timeline(
            lw_pairs=[
                (-20e-9, SquareWaveform(width=30e-9)),
                (0, GaussianWaveform(width=20e-9)),
            ]
        )

        lw_pairs = tmln.resolve()

        assert lw_pairs == tmln.lw_pairs
        assert tmln.locations == [0, 20e-9]
        assert tmln.width == 40e-9

    def test_resolve_infinite_waveform(self):
        tmln = Timeline(
            lw_pairs=[
                (-10e-9, DCWaveform()),
                (0, GaussianWaveform(width=20e-9)),
                (10e-9, GaussianWaveform(width=30e-9)),
            ]
        )

        tmln.resolve()
        assert tmln.locations == [0, 10e-9, 20e-9]
        assert tmln.width == 50e-9

    @pytest.mark.parametrize(
        "substitutions,expect",
        [
            (
                dict(amp=0.5),
                Timeline.from_layers(
                    [
                        SquareWaveform(width="tau", amplitude=0.5),
                        "wait",
                        SquareWaveform(width="tau", amplitude=-0.5),
                    ],
                    width="wait + 2*tau",
                ),
            ),
            (
                dict(tau=30e-9),
                Timeline.from_layers(
                    [
                        SquareWaveform(width=30e-9, amplitude="amp"),
                        "wait",
                        SquareWaveform(width=30e-9, amplitude="-amp"),
                    ],
                    width="wait + 60e-9",
                ),
            ),
            (
                dict(amp=0.1, wait=5e-9, tau=30e-9),
                Timeline.from_layers(
                    [
                        SquareWaveform(width=30e-9, amplitude=0.1),
                        5e-9,
                        SquareWaveform(width=30e-9, amplitude=-0.1),
                    ],
                    width=65e-9,
                ),
            ),
        ],
    )
    def test_substitute(self, substitutions, expect):
        tmln = Timeline.from_layers(
            [
                SquareWaveform(width="tau", amplitude="amp"),
                "wait",
                SquareWaveform(width="tau", amplitude="-amp"),
            ],
            width="wait + 2*tau",
        )

        assert tmln.substitute(**substitutions) == expect

    def test_rename_variables(self):
        tmln = Timeline(
            lw_pairs=[
                (0, SquareWaveform(amplitude="amp", width="t")),
                ("t", GaussianWaveform(width="t")),
            ]
        )

        tmln.rename_variables(lambda n: "tgate" if n == "t" else n)

        assert tmln == Timeline(
            lw_pairs=[
                (0, SquareWaveform(amplitude="amp", width="tgate")),
                ("tgate", GaussianWaveform(width="tgate")),
            ]
        )

        tmln.width = "tgate + tbuffer"
        tmln.rename_variables(lambda n: n + "1" if n != "tgate" else n)

        assert tmln == Timeline(
            lw_pairs=[
                (0, SquareWaveform(amplitude="amp1", width="tgate")),
                ("tgate", GaussianWaveform(width="tgate")),
            ],
            width="tgate + tbuffer1",
        )

    def test_add_timeline(self):
        m = Marker(name="m1")
        tmln1 = Timeline(lw_pairs=[("a", m), ("b", m)])
        tmln2 = Timeline.fromtuples([("c", m), ("d", m)])

        tmln1.add_timeline(tmln2)

        assert tmln1.locations == [sym.Symbol(s) for s in "abcd"]

    @pytest.mark.parametrize(
        "vars1,vars2,name,expect",
        [
            (["a", "b", "c"], ["b", "c", "d"], None, set("abcd")),
        ],
    )
    def test_add_timeline_shared_variables(self, vars1, vars2, name, expect):
        m = Marker(name="m1")
        tmln1 = Timeline(lw_pairs=[(v, m) for v in vars1])
        tmln2 = Timeline(lw_pairs=[(v, m) for v in vars2])

        tmln1.add_timeline(tmln2, name=name)
        assert tmln1.variables() == expect

    @pytest.mark.parametrize(
        "self_loc,other_loc,name",
        [
            (0, 0, "t2_start"),
            (0, 5, "t2_start"),
            (5, 0, "t2_start"),
            (0, 0, None),
        ],
    )
    def test_add_timeline_location_name(self, self_loc, other_loc, name):
        tmln1 = Timeline()
        tmln2 = Timeline()

        tmln1.add_timeline(tmln2, self_loc, other_loc, name=name)

        if name:
            assert tmln1.constraints.pop() == sym.parse_expr(
                f"{name} - ({float(self_loc)} - {float(other_loc)})"
            )
        else:
            assert tmln1.constraints == set()

    @pytest.mark.parametrize(
        "tmln,target,result",
        [
            (Timeline(), Timeline(), Timeline()),
            (
                Timeline(),
                SquareWaveform(),
                Timeline(lw_pairs=[(0, SquareWaveform())]),
            ),
            (
                Timeline(),
                [SquareWaveform(width=10e-9), SquareWaveform(width=20e-9)],
                Timeline(
                    lw_pairs=[
                        (0, SquareWaveform(width=10e-9)),
                        (0, SquareWaveform(width=20e-9)),
                    ]
                ),
            ),
            (Timeline(), t := Timeline(lw_pairs=[(0, SquareWaveform())]), t),
        ],
    )
    def test_add(self, tmln, target, result):
        assert tmln.add(target) == result

    @pytest.mark.parametrize(
        "tmln1,tmln2,result",
        [
            (Timeline(), Timeline(), Timeline()),
            (
                Timeline(
                    lw_pairs=[("start", SquareWaveform(channel="a"))],
                    constraints=dict(start=Location()),
                ),
                Timeline(
                    lw_pairs=[("start", SquareWaveform(channel="b"))],
                    constraints=["start", "width - 10.0"],
                ),
                Timeline(
                    lw_pairs=[
                        ("start", SquareWaveform(channel="a")),
                        ("start", SquareWaveform(channel="b")),
                    ],
                    constraints=["start", "width - 10.0"],
                ),
            ),
        ],
    )
    def test_add_operator(self, tmln1, tmln2, result):
        assert tmln1 + tmln2 == result

    def test_transform_waveforms(self):
        lws = [
            (0, VirtualZWaveform(frame="mod_Q0_GE", phase="zphase")),
            (
                0,
                ModulatedWaveform(
                    envelope=GaussianWaveform(amplitude="amplitude", width="width"),
                    modulation=CWWaveform(frequency="mod_Q0_GE", channel="Q0"),
                ),
            ),
            ("width", VirtualZWaveform(frame="mod_Q0_GE", phase="zphase")),
        ]
        tmln = Timeline(lw_pairs=lws, width="width")

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

        for wave in tmln.operations:
            match wave:
                case VirtualZWaveform():
                    assert wave.frame == Frame("Q0.mod_GE")
                case ModulatedWaveform():
                    assert wave.modulation.frequency == Frame("Q0.mod_GE")

    @pytest.mark.parametrize(
        "tmln",
        [
            Timeline(),
            Timeline(lw_pairs=[("a", WAVEFORMS["c1"]), ("b", WAVEFORMS["g1"])]),
            Timeline(
                lw_pairs=[("a", WAVEFORMS["c1"]), ("a", WAVEFORMS["g1"])],
                constraints=["a"],
            ),
        ],
    )
    def test_deep_copy(self, tmln):
        tmlncopy = tmln.copy()

        assert tmln == tmlncopy
        assert tmln.lw_pairs is not tmlncopy.lw_pairs
        assert tmln.constraints is not tmlncopy.constraints
        assert tmln.channels is not tmlncopy.channels

    @pytest.mark.parametrize(
        "waveforms,channels,channel_map",
        [
            (
                ["c1", "s1"],
                [],
                dict(
                    IQ=[(1.0, WAVEFORMS["s1"])],
                    F1=[(0.0, WAVEFORMS["c1"])],
                ),
            ),
            (
                ["c1", "g2", "s1"],
                ["Q"],
                dict(Q=[(1.0, WAVEFORMS["g2"])]),
            ),
        ],
    )
    def test_get_channel_map(self, waveforms, channels, channel_map):
        tmln = Timeline(lw_pairs=[(i, WAVEFORMS[w]) for i, w in enumerate(waveforms)])

        channel_map = {c: waves for c, waves in channel_map.items()}
        assert tmln.get_channel_map(*channels) == channel_map

    @pytest.mark.parametrize(
        "tmln,tmln_dict",
        [
            (Timeline(), {}),
            (
                Timeline(
                    lw_pairs=[
                        (
                            0,
                            SquareWaveform(
                                amplitude="amp", width="tau/2", channel="Q0"
                            ),
                        ),
                        ("tau / 2", SquareWaveform(amplitude="-amp", width="tau/2")),
                        ("tau", VirtualZWaveform(frame="Q0", phase=90)),
                    ],
                ),
                dict(
                    lw_pairs=[
                        [
                            0.0,
                            dict(
                                channel="Q0",
                                width="tau/2",
                                amplitude="amp",
                                __class__="SquareWaveform",
                            ),
                        ],
                        [
                            "tau/2",
                            dict(
                                width="tau/2",
                                amplitude="-amp",
                                __class__="SquareWaveform",
                            ),
                        ],
                        [
                            "tau",
                            dict(frame="Q0", phase=90, __class__="VirtualZWaveform"),
                        ],
                    ]
                ),
            ),
            (
                Timeline(constraints=["10e-9 - tau", "tau / 2 - mid"]),
                dict(constraints=ignore_order(["1.0e-8 - tau", "-mid + tau/2"])),
            ),
            (Timeline(width=20e-9), dict(width=20e-9)),
        ],
    )
    def test_serialization(self, tmln, tmln_dict):
        unstructured = qwip.converter.unstructure(tmln)
        restructured = qwip.converter.structure(unstructured, Timeline)

        assert unstructured == tmln_dict
        assert tmln == restructured

    @pytest.mark.parametrize(
        "tmln_dict,tmln",
        [
            (
                dict(
                    locations={
                        "1 + width": [
                            {
                                "channels": ["I", "Q"],
                                "width": 3.2e-8,
                                "__class__": "SquareWaveform",
                            }
                        ]
                    },
                    constraints=dict(width=5.0),
                ),
                Timeline(
                    lw_pairs=[(1 + Location("width"), WAVEFORMS["s1"])],
                    constraints={"width - 5.0"},
                ),
            ),
        ],
    )
    def test_legacy_serialization(self, tmln_dict, tmln):
        structured = qwip.converter.structure(tmln_dict, Timeline)
        assert tmln == structured
