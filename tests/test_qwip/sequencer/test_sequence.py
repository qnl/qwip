import itertools as it
from contextlib import nullcontext as noerror

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import qwip
from qwip.sequencer.sequence import (
    Sequence,
    _combine_indices,
    _is_advanced_index,
    _is_default_label,
    broadcast_labels,
    stack,
)
from qwip.sequencer.timeline import Timeline
from qwip.sequencer.waveform import SquareWaveform


class TestSequenceConstruction:
    @pytest.mark.parametrize(
        "elements,labels,names,shape,error",
        [
            ([], dict(), ("d0",), (0,), noerror()),
            ([Timeline() for _ in range(10)], dict(), ("d0",), (10,), noerror()),
            (
                [[Timeline() for _ in range(4)] for _ in range(3)],
                dict(d0=None, d1=None),
                ("d0", "d1"),
                (3, 4),
                noerror(),
            ),
            (
                [[Timeline() for _ in range(4)] for _ in range(15)],
                dict(a=np.arange(15), b=np.arange(4)),
                ("a", "b"),
                (15, 4),
                noerror(),
            ),
            (
                [Timeline() for _ in range(10)],
                dict(a=None, b=None),
                ("a", "b"),
                (10,),
                pytest.raises(ValueError),
            ),
            (
                [Timeline() for _ in range(5)],
                dict(a=np.arange(10)),
                ("a",),
                (5),
                pytest.raises(ValueError),
            ),
            (
                [[Timeline() for _ in range(5)] for _ in range(4)],
                dict(d1=None),
                ("d0", "d1"),
                (4, 5),
                pytest.raises(ValueError),
            ),
        ],
    )
    def test_explicit_constructor(self, elements, labels, names, shape, error):
        with error:
            s = Sequence(elements, **labels)

            # Check shape matches
            assert s.shape == shape
            assert tuple(lb.shape[0] for lb in s.labels) == s.shape

            # Check names
            assert s.names == names

    def test_label_conversion(self):
        s = Sequence([Timeline() for _ in range(3)], a=[1, 2, 3])

        assert isinstance(s["a"], pd.Index)
        assert s["a"].equals(pd.Index([1, 2, 3], name="a"))

    def test_multiindex_name(self):
        s = Sequence(
            [Timeline() for _ in range(2)],
            d0=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["l", "n"]),
        )

        assert s.names == ("l_n",)

    @pytest.mark.parametrize(
        "arr",
        [
            np.array(
                [[Timeline() for _ in range(3)] for _ in range(2)],
            ),
            Sequence.empty((2, 3), a=np.arange(2)),
        ],
    )
    def test_constructor_pass_through(self, arr):
        s = Sequence(arr, c=np.zeros(2), d=np.ones(3))

        assert s.names == ("c", "d")
        assert s["c"].equals(pd.Index(np.zeros(2), name="c"))
        assert s["d"].equals(pd.Index(np.ones(3), name="d"))

        assert_array_equal(s, arr)
        assert s.base is arr

    def test_constructor_pass_through_no_labels(self):
        s1 = Sequence.empty((2, 3), a=None, b=None)
        s2 = Sequence(s1)

        assert s1.names == s2.names
        assert s1["a"] is s2["a"]
        assert s1["b"] is s2["b"]

    @pytest.mark.parametrize(
        "sequence",
        [
            Sequence.empty((1, 2, 3, 4)),
            Sequence.empty(
                (4, 5), names=("a", "b"), a=np.linspace(0, 1, 4), b=np.zeros(5)
            ),
            Sequence.empty(
                10,
                names=("d0",),
            ),
        ],
    )
    def test_view(self, sequence):
        s1 = sequence.view()

        # assert_array_equal(sequence, s1)
        assert sequence.names == s1.names
        # Labels are not copied
        assert sequence.labels is s1.labels
        assert s1.base is sequence

    @pytest.mark.parametrize(
        "shape,labels",
        [
            (10, dict()),
            (
                (3, 4, 5),
                dict(a=np.arange(3), b=np.arange(4), c=np.arange(5)),
            ),
            ((1, 1), dict()),
        ],
    )
    def test_empty(self, shape, labels):
        sequence = Sequence.empty(shape, **labels)

        if not isinstance(shape, tuple):
            shape = (shape,)

        assert sequence.shape == shape

        obj_ids = set()
        for index in np.ndindex(*shape):
            se = sequence[index]
            obj_ids.add(id(se))

        # Checks that all timelines are unique/separate objects
        assert len(obj_ids) == np.prod(shape)

        for name, label in labels.items():
            assert sequence[name].equals(pd.Index(label, name=name))

    @pytest.mark.parametrize(
        "params,shape,expect",
        [
            (dict(tau=[0, 1, 2]), 3, pd.Index([0, 1, 2], name="tau")),
            (dict(tau=[0, 1, 2], wait=1), 3, pd.Index([0, 1, 2], name="tau")),
            (
                dict(tau=[0, 1, 2], wait=[0]),
                3,
                pd.MultiIndex.from_tuples(
                    [(0, 0), (1, 0), (2, 0)], names=("tau", "wait")
                ),
            ),
            (
                dict(tau=[0, 1], wait=[1, 2], amp1=[2, 3], amp2="amp3"),
                2,
                pd.MultiIndex.from_tuples(
                    [(0, 1, 2), (1, 2, 3)], names=("tau", "wait", "amp1")
                ),
            ),
        ],
    )
    def test_sweep(self, params, shape, expect):
        tmln = Timeline.from_layers(
            [
                SquareWaveform(width="tau", amplitude="amp1"),
                "wait",
                SquareWaveform(width="tau", amplitude="amp2"),
            ]
        )

        seq = Sequence.sweep(tmln, **params)
        assert seq.shape[0] == shape
        assert seq.labels[0].equals(expect)
        assert seq.labels[0].name == (expect.name or "_".join(expect.names))

        for tmln in seq.flat:
            assert set(tmln.constraints) == set(params)

    @pytest.mark.parametrize(
        "params,shape,expected",
        [
            (
                dict(tau=[0, 1], wait=np.linspace(1, 2, 3)),
                (2, 3),
                dict(
                    tau=pd.RangeIndex(2, name="tau"),
                    wait=pd.Index(np.linspace(1, 2, 3), name="wait"),
                ),
            ),
            (dict(tau=[0, 1], wait=0), (2,), dict(tau=pd.RangeIndex(2, name="tau"))),
            (
                dict(
                    tau=[0, 1],
                    wait=[0, 1, 2],
                    amp1_amp2=(
                        idx := pd.MultiIndex.from_tuples(
                            [(-1, 1), (-1.5, 1.5), (-2, 2)], names=("amp1", "amp2")
                        )
                    ),
                ),
                (2, 3, 3),
                dict(
                    tau=pd.RangeIndex(2, name="tau"),
                    wait=pd.RangeIndex(3),
                    amp1_amp2=idx,
                ),
            ),
        ],
    )
    def test_product(self, params, shape, expected):
        tmln = Timeline.from_layers(
            [
                SquareWaveform(width="tau", amplitude="amp1"),
                "wait",
                SquareWaveform(width="tau", amplitude="amp2"),
            ]
        )

        seq = Sequence.product(tmln, **params)

        assert seq.shape == shape
        assert seq.names == tuple(expected)

        for label, expect in zip(seq.labels, expected.values()):
            assert label.equals(expect)

        for tmln in seq.flat:
            assert set(tmln.constraints) == set(
                it.chain.from_iterable(n.split("_") for n in params)
            )


class TestBroadcastLabels:
    def test_is_default_label(self):
        seq = Sequence.empty(
            (1, 2, 3, 4, 5),
            d0=None,
            d1=np.ones(2),
            d2c=None,
            d03=None,
            axis4=np.r_[:5] ** 2,
        )

        expected = [True, False, False, True, False]

        for axis, expect in enumerate(expected):
            assert _is_default_label(seq.labels[axis], seq.shape[axis]) is expect

    @pytest.mark.parametrize(
        "indices,dim,expect",
        [
            (
                (
                    pd.Index(["a", "b", "c"], name="letters"),
                    pd.RangeIndex(0, 3, name="numbers"),
                ),
                3,
                pd.MultiIndex.from_tuples(
                    [("a", 0), ("b", 1), ("c", 2)], names=["letters", "numbers"]
                ),
            ),
            ((pd.RangeIndex(10, name="range"),), 10, pd.RangeIndex(10, name="range")),
            (
                (pd.Index(["a"], name="letters"), pd.RangeIndex(5, name="numbers")),
                5,
                pd.MultiIndex.from_product(
                    [["a"], np.arange(5)], names=["letters", "numbers"]
                ),
            ),
            (
                (
                    pd.RangeIndex(10, name="first"),
                    pd.MultiIndex.from_product(
                        [np.arange(2), np.arange(5)], names=["second", "third"]
                    ),
                    pd.Index(["fourth"], name="fourth"),
                ),
                10,
                pd.MultiIndex.from_arrays(
                    [
                        np.arange(10),
                        np.repeat(np.arange(2), 5),
                        np.tile(np.arange(5), 2),
                        ["fourth"] * 10,
                    ],
                    names=["first", "second", "third", "fourth"],
                ),
            ),
            (
                (pd.RangeIndex(10, name="first"), pd.RangeIndex(10, name="second")),
                10,
                pd.MultiIndex.from_tuples(
                    [(i, i) for i in range(10)], names=["first", "second"]
                ),
            ),
            (
                (pd.RangeIndex(10, name="first"), pd.RangeIndex(10, name="first")),
                10,
                pd.RangeIndex(10, name="first"),
            ),
            (
                (
                    pd.RangeIndex(5, name="duplicate"),
                    pd.RangeIndex(5, name="duplicate"),
                    pd.Index(["a"], name="unique"),
                ),
                5,
                pd.MultiIndex.from_product(
                    [np.arange(5), ["a"]], names=["duplicate", "unique"]
                ),
            ),
        ],
    )
    def test_combine_indices(self, indices, dim, expect):
        combined = _combine_indices(*indices, dim=dim)
        assert combined.equals(expect)

        if isinstance(expect, pd.MultiIndex):
            assert combined.names == expect.names
            assert combined.levshape == expect.levshape
        else:
            assert combined.name == expect.name

    def test_broadcast_labels_default(self):
        s1 = Sequence.empty((4,))
        s2 = Sequence.empty((5, 1))
        s3 = Sequence.empty((2, 1, 1))

        labels = broadcast_labels(s1, s2, s3)

        dims = (2, 5, 4)
        names = ("d0", "d1", "d2")
        for label, name, dim in zip(labels, names, dims):
            assert label.equals(pd.RangeIndex(dim))
            assert label.name == name

    def test_broadcast_labels_simple(self):
        s1 = Sequence.empty((4,))
        s2 = Sequence.empty((5, 1), first=np.arange(5), second=["a"])

        labels = broadcast_labels(s1, s2)

        assert [lb.name for lb in labels] == ["first", "second"]
        assert labels[0].equals(pd.RangeIndex(5))
        assert labels[1].equals(pd.Index(["a"] * 4))

    def test_broadcast_labels_duplicate(self):
        s1 = Sequence.empty((2, 2), d0=None, second=[2, 4])
        s2 = Sequence.empty((2,), second=[2, 4])

        labels = broadcast_labels(s1, s2)

        assert [lb.name for lb in labels] == ["d0", "second"]
        assert labels[0].equals(pd.RangeIndex(2))
        assert labels[1].equals(pd.Index([2, 4]))

    def test_broadcast_labels_multiindex(self):
        prep = [0, 1, 2, 3]
        post = ["I", "X", "Y", "Z"]
        s1 = Sequence.empty(4, prep=prep)
        s2 = Sequence.empty(4, post=post)

        labels = broadcast_labels(s1, s2)
        assert [lb.name for lb in labels] == ["prep_post"]
        assert labels[0].names == ["prep", "post"]
        assert labels[0].equals(pd.MultiIndex.from_arrays([prep, post]))


class TestSequenceIndexing:
    @pytest.mark.parametrize(
        "index,is_advanced",
        [
            ((1, 2, 3), False),
            (((1, 2, 3),), True),
            ((slice(None), ..., np.newaxis), False),
            (np.arange(10), True),
        ],
    )
    def test_is_advanced_index(self, index, is_advanced):
        assert _is_advanced_index(index) == is_advanced

    @pytest.mark.parametrize(
        "start_shape,index,expected_shape",
        [
            ((3, 4, 5), np.s_[0, :, :], (4, 5)),
            ((3, 4, 5), np.s_[:2, ...], (2, 4, 5)),
            ((3, 4, 5), np.s_[..., np.newaxis, ::2], (3, 4, 1, 3)),
            ((3, 4, 5), np.s_[..., :, :, :], (3, 4, 5)),
        ],
    )
    def test_basic_indexing(self, start_shape, index, expected_shape):
        s = Sequence.empty(start_shape)

        view = s[index]
        assert view.shape == expected_shape
        assert len(view.labels) == view.ndim
        assert view.names == tuple(f"d{i}" for i in range(view.ndim))

        for axis, dim in enumerate(view.shape):
            assert len(view.labels[axis]) == dim

    @pytest.mark.parametrize(
        "shape,labels,index,expected",
        [
            (
                (3, 4, 5),
                (
                    pd.Index(["a", "b", "c"], name="letters"),
                    pd.RangeIndex(4, name="numbers"),
                    pd.Index([5, 4, 3, 2, 1], name="reversed"),
                ),
                np.s_[0, :2, 1::2],
                (pd.RangeIndex(2, name="numbers"), pd.Index([4, 2], name="reversed")),
            ),
            (
                (3, 4),
                (pd.RangeIndex(3, name="a"), pd.RangeIndex(4, name="b")),
                np.s_[:, np.newaxis, :],
                (
                    pd.RangeIndex(3, name="a"),
                    pd.RangeIndex(1, name="d1"),
                    pd.RangeIndex(4, name="b"),
                ),
            ),
            (
                (5,),
                (pd.RangeIndex(5, name="a"),),
                np.s_[::-1],
                (pd.RangeIndex(4, -1, -1, name="a"),),
            ),
            (
                (3, 4, 5),
                (
                    pd.RangeIndex(3, name="d0"),
                    pd.RangeIndex(4, name="d1"),
                    pd.RangeIndex(5, name="d2"),
                ),
                np.s_[np.newaxis, :, 0, np.newaxis, ...],
                (
                    pd.RangeIndex(1, name="d0"),
                    pd.RangeIndex(3, name="d1"),
                    pd.RangeIndex(1, name="d2"),
                    pd.RangeIndex(5, name="d3"),
                ),
            ),
            (
                (2, 3, 4),
                (
                    pd.RangeIndex(2, name="a"),
                    2 + pd.RangeIndex(3, name="d1"),
                    pd.RangeIndex(4, name="c"),
                ),
                np.s_[0],
                (
                    2 + pd.RangeIndex(3, name="d0"),
                    pd.RangeIndex(4, name="c"),
                ),
            ),
        ],
    )
    def test_labels_from_index(self, shape, labels, index, expected):
        s = Sequence.empty(shape, **{lb.name: lb for lb in labels})

        expanded = s._expand_basic_index(index)
        labels = s._get_labels_from_index(expanded)

        assert len(labels) == len(expected)

        for lb, expect in zip(labels, expected):
            assert lb.name == expect.name
            assert lb.equals(expect)

    @pytest.mark.parametrize(
        "shape,labels,index,expected",
        [
            (
                (2, 5, 4, 5),
                (
                    pd.RangeIndex(2, name="d0"),
                    pd.Index([3, 1, 4, 1, 5], name="pi"),
                    pd.Index(list("abcd"), name="letters"),
                    -pd.RangeIndex(5, name="negative"),
                ),
                np.s_[np.newaxis, :-1, -4:3, np.newaxis, 1::2, ::-1],
                (
                    pd.RangeIndex(1, name="d0"),
                    pd.RangeIndex(1, name="d1"),
                    pd.Index([1, 4], name="pi"),
                    pd.RangeIndex(1, name="d3"),
                    pd.Index(["b", "d"], name="letters"),
                    pd.Index([-4, -3, -2, -1, 0], name="negative"),
                ),
            ),
            (
                (10, 2),
                (pd.RangeIndex(10, name="a"), pd.Index(["a", "b"], name="alphabet")),
                np.s_[4],
                (pd.Index(["a", "b"], name="alphabet"),),
            ),
            (
                (2, 3, 4),
                (
                    pd.RangeIndex(2, name="d0"),
                    pd.RangeIndex(3, name="d1"),
                    pd.RangeIndex(4, name="d2"),
                ),
                np.s_[...],
                (
                    pd.RangeIndex(2, name="d0"),
                    pd.RangeIndex(3, name="d1"),
                    pd.RangeIndex(4, name="d2"),
                ),
            ),
            (
                (10,),
                (
                    pd.MultiIndex.from_product(
                        [["a", "b"], list(range(5))], names=["letters", "numbers"]
                    ),
                ),
                np.s_[::5],
                (
                    pd.MultiIndex.from_tuples(
                        [("a", 0), ("b", 0)], names=["letters", "numbers"]
                    ),
                ),
            ),
        ],
    )
    def test_basic_indexing_with_labels(self, shape, labels, index, expected):
        s = Sequence.empty(
            shape, **{lb.name or f"d{i}": lb for i, lb in enumerate(labels)}
        )
        view = s[index]

        assert view.names == tuple(lb.name or "_".join(lb.names) for lb in expected)
        assert view.shape == tuple(len(lb) for lb in expected)

        for lb, expect in zip(view.labels, expected):
            assert lb.equals(expect)


class TestSequenceShaping:
    def test_reshape(self):
        s = Sequence.empty((4, 5, 3), names=("a", "b", "c"), a=np.arange(4))

        r = s.reshape(2, -1)
        assert r.shape == (2, 30)
        assert r.names == ("d0", "d1")
        assert r["d0"].equals(pd.RangeIndex(2))
        assert r["d1"].equals(pd.RangeIndex(30))
        assert r.base is s

    @pytest.mark.parametrize(
        "seq,axes,shape,names",
        [
            (
                Sequence.empty((1, 2, 3, 4), a=None, b=None, c=None, d=None),
                None,
                (4, 3, 2, 1),
                ("d", "c", "b", "a"),
            ),
            (
                Sequence.empty((1, 2, 3, 4), a=None, b=None, c=None, d=None),
                (2, 0, 1, 3),
                (3, 1, 2, 4),
                ("c", "a", "b", "d"),
            ),
        ],
    )
    def test_transpose(self, seq, axes, shape, names):
        r = seq.transpose(axes)
        r1 = np.transpose(seq, axes)

        assert r.names == names
        assert r.shape == shape
        assert r1.names == names
        assert r1.shape == shape

        for axis in range(r.ndim):
            assert len(r.labels[axis]) == len(r1.labels[axis]) == r.shape[axis]

    @pytest.mark.parametrize(
        "shape,names", [((2,), ("a",)), ((1, 2, 3), ("a", "b", "c"))]
    )
    def test_T(self, shape, names):
        s = Sequence.empty(shape, **{n: None for n in names})

        assert s.T.shape == tuple(reversed(shape))
        assert s.T.names == tuple(reversed(names))

    @pytest.mark.parametrize(
        "shape,expect",
        [
            (
                (2, 3),
                pd.MultiIndex.from_product((np.r_[:2], np.r_[:3]), names=("d0", "d1")),
            ),
            (
                (2, 4, 3),
                pd.MultiIndex.from_product(
                    (np.r_[:2], np.r_[:4], np.r_[:3]), names=("d0", "d1", "d2")
                ),
            ),
        ],
    )
    def test_flatten(self, shape, expect):
        seq = Sequence.empty(shape)

        flattened = seq.flatten()
        assert len(flattened.labels) == 1
        assert flattened.names[0] == "_".join(expect.names)
        assert flattened.labels[0].equals(expect)

    def test_flatten_1D(self):
        seq = Sequence.empty(10, a=pd.RangeIndex(10, 20))
        flattened = seq.flatten()

        assert seq.names == flattened.names
        assert seq.labels[0].equals(flattened.labels[0])

    def test_flatten_multiindex(self):
        seq = Sequence.empty(
            (2, 4),
            sign=["pos", "neg"],
            prep_post=pd.MultiIndex.from_product(
                (["I", "X"], ["I", "X"]), names=("prep", "post")
            ),
        )
        flattened = seq.flatten()

        assert flattened.shape == (8,)
        assert len(flattened.labels) == 1
        assert flattened.names[0] == "sign_prep_post"
        assert flattened["sign_prep_post"].equals(
            pd.MultiIndex.from_product(
                (["pos", "neg"], ["I", "X"], ["I", "X"]), names=("sign", "prep", "post")
            )
        )


class TestSequenceJoins:
    @pytest.mark.parametrize(
        "seqs,axis,expected",
        [
            (
                [Sequence.empty(5), Sequence.empty(6), Sequence.empty(7)],
                0,
                dict(d0=pd.RangeIndex(18)),
            ),
            (
                [
                    Sequence.empty((5, 2), d0=None, b=None),
                    Sequence.empty((5, 3), d0=None, b=None),
                ],
                1,
                dict(
                    d0=pd.RangeIndex(5, name="d0"),
                    b=pd.Index([0, 1, 0, 1, 2], name="b"),
                ),
            ),
            (
                [
                    Sequence.empty(
                        (2, 2),
                        letters=pd.Index(["a", "b"], name="letters"),
                        label0=None,
                    ),
                    Sequence.empty(
                        (2, 3), numbers=pd.Index([0, 1], name="numbers"), label1=None
                    ),
                ],
                1,
                dict(
                    letters_numbers=pd.MultiIndex.from_tuples(
                        [("a", 0), ("b", 1)], names=["letters", "numbers"]
                    ),
                    d1=pd.Index([0, 1, 0, 1, 2], name="d0"),
                ),
            ),
            (
                [
                    Sequence.empty((5, 2), d0=None, d1=None),
                    Sequence.empty((5, 3), d0=None, d1=None),
                ],
                -1,
                dict(d0=pd.RangeIndex(5), d1=pd.RangeIndex(5)),
            ),
        ],
    )
    def test_concatenate(self, seqs, axis, expected):
        c = np.concatenate(seqs, axis=axis)

        if axis < 0:
            axis = seqs[0].ndim + axis

        shape = []
        for dim in range(len(seqs[0].shape)):
            if dim == axis:
                shape.append(sum(s.shape[dim] for s in seqs))
            else:
                shape.append(seqs[0].shape[dim])

        assert c.shape == tuple(shape)
        assert c.names == tuple(expected)

        for i, (label, expect) in enumerate(zip(c.labels, expected.values())):
            assert label.equals(expect)
            assert len(label) == shape[i]

    @pytest.mark.parametrize(
        "seqs,axis,expected",
        [
            (
                [Sequence.empty(10, a=None), Sequence.empty(10), Sequence.empty(10)],
                -1,
                dict(a=pd.RangeIndex(10, name="a"), d1=pd.RangeIndex(3, name="d1")),
            ),
            (
                [Sequence.empty(5), Sequence.empty(5)],
                0,
                dict(d0=pd.RangeIndex(2, name="d0"), d1=pd.RangeIndex(5, name="d1")),
            ),
            (
                [
                    Sequence.empty((4, 2), f1=pd.RangeIndex(4), I=[0, 2]),
                    Sequence.empty((4, 2), f2=pd.RangeIndex(4, 8)),
                ],
                0,
                dict(
                    d0=pd.RangeIndex(2, name="d0"),
                    f1_f2=pd.MultiIndex.from_tuples(
                        [(i, i + 4) for i in range(4)], names=["f1", "f2"]
                    ),
                    I=pd.Index([0, 2], name="I"),
                ),
            ),
        ],
    )
    def test_stack(self, seqs, axis, expected):
        c = np.stack(seqs, axis=axis)

        shape = list(seqs[0].shape)
        axis = len(shape) + 1 + axis if axis < 0 else axis
        shape.insert(axis, len(seqs))

        assert c.shape == tuple(shape)
        assert c.names == tuple(expected)

        for i, (label, expect) in enumerate(zip(c.labels, expected.values())):
            assert label.equals(expect)
            assert len(label) == shape[i]

    @pytest.mark.parametrize(
        "label,expect",
        [
            ("seq", pd.RangeIndex(2, name="seq")),
            (pd.Index([0.5, 0.6], name="seq"), pd.Index([0.5, 0.6], name="seq")),
        ],
    )
    def test_stack_with_label(self, label, expect):
        a = Sequence.empty((3, 4), d0=None, d1=list("abcd"))
        b = Sequence.empty((3, 4), d0=[0, 2, 4])

        c = stack((a, b), axis=0, label=label)

        assert c.shape == (2, 3, 4)
        assert c.names == (expect.name, "d1", "d2")
        assert c["d1"].equals(pd.Index([0, 2, 4]))
        assert c["d2"].equals(pd.Index(["a", "b", "c", "d"]))
        assert c[expect.name].equals(expect)

    def test_stack_label_error(self):
        with pytest.raises(ValueError):
            stack([Sequence.empty(2), Sequence.empty(2)], label=pd.RangeIndex(10))


class TestSequenceUniversalFunctions:
    @pytest.mark.parametrize(
        "a,b,shape,expected",
        [
            (
                Sequence.empty((10,), a=np.arange(10)),
                Sequence.empty((10,)),
                (10,),
                dict(a=pd.RangeIndex(10, name="a")),
            ),
            (
                Sequence.empty((3, 2, 1), a=np.arange(3)),
                Sequence.empty((1, 2, 3), d0=None, b=None, c=np.arange(3)),
                (3, 2, 3),
                dict(
                    a=pd.RangeIndex(3, name="a"),
                    b=pd.RangeIndex(2, name="b"),
                    c=pd.RangeIndex(3, name="c"),
                ),
            ),
            (
                Sequence.empty((2, 3, 1), d0=None, d1=None, c=np.ones(1)),
                Sequence.empty((4,), c=np.ones(4)),
                (2, 3, 4),
                dict(
                    d0=pd.RangeIndex(2, name="d0"),
                    d1=pd.RangeIndex(3, name="d1"),
                    c=pd.Index(np.ones(4), name="c"),
                ),
            ),
            (
                Sequence.empty(4, a=np.arange(4)),
                Sequence.empty(4, b=np.ones(4)),
                (4,),
                dict(
                    a_b=pd.MultiIndex.from_tuples(
                        list(enumerate([1] * 4)), names=["a", "b"]
                    )
                ),
            ),
        ],
    )
    def test_ufunc_call(self, a, b, shape, expected):
        c = a + b

        assert c.shape == shape
        assert c.names == tuple(expected)

        for label, expect in zip(c.labels, expected.values()):
            assert label.equals(expect)

    def test_ufunc_non_sequence(self):
        seq = Sequence.empty((4, 2))
        equals = seq == seq

        assert not isinstance(equals, Sequence)
        assert np.all(equals)

    @pytest.mark.parametrize(
        "seq,kwargs,shape,expected",
        [
            (
                Sequence.empty((10,), a=None),
                {},
                tuple(),
                {},
            ),
            (
                Sequence.empty((3, 2, 1), a=None, b=None, c=None),
                dict(axis=1),
                (3, 1),
                dict(a=pd.RangeIndex(3, name="a"), c=pd.RangeIndex(1, name="c")),
            ),
            (
                Sequence.empty((3, 2), a=None, b=None),
                dict(axis=1, keepdims=True),
                (3, 1),
                dict(a=pd.RangeIndex(3, name="a"), b=pd.RangeIndex(1, name="d1")),
            ),
        ],
    )
    def test_ufunc_reduce(self, seq, kwargs, shape, expected):
        result = np.sum(seq, **kwargs)

        assert result.shape == shape
        assert result.names == tuple(expected)

        for label, expect in zip(result.labels, expected.values()):
            assert label.equals(expect)


class TestSequenceSerialization:
    def test_sequence_unstructure(self):
        s = Sequence.empty((2, 2), a=None, b=None)

        unstructured = qwip.converter.unstructure(s)

        assert unstructured == dict(
            names=["a", "b"],
            labels=dict(a=[0, 1], b=[0, 1]),
            shape=(2, 2),
            data=[[{}, {}], [{}, {}]],
        )

    def test_sequence_structure(self):
        unstructured = dict(
            names=("iter", "time", "frequency"),
            labels=dict(
                iter=[0, 1, 2, 3], time=[10e-9, 20e-9, 30e-9], frequency=[5.4, 5.5]
            ),
            shape=(4, 3, 2),
            data=[
                [[{}, {}], [{}, {}], [{}, {}]],
                [[{}, {}], [{}, {}], [{}, {}]],
                [[{}, {}], [{}, {}], [{}, {}]],
                [[{}, {}], [{}, {}], [{}, {}]],
            ],
        )

        seq = qwip.converter.structure(unstructured, Sequence)
        assert isinstance(seq, Sequence)
        assert seq.names == unstructured["names"]
        assert seq.shape == unstructured["shape"]

        for n in seq.names:
            assert seq[n].equals(pd.Index(unstructured["labels"][n]))
