from contextlib import nullcontext as noerror

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from qwip.sequencer.elements import SequenceElement
from qwip.sequencer.sequence import (
    Sequence,
    _is_advanced_index,
    broadcast_names_and_labels,
    stack,
)


class TestSequenceConstruction:
    @pytest.mark.parametrize(
        "elements,names,labels,shape,error",
        [
            ([], None, dict(), (0,), noerror()),
            ([SequenceElement() for _ in range(10)], None, dict(), (10,), noerror()),
            (
                [[SequenceElement() for _ in range(4)] for _ in range(3)],
                (None, None),
                dict(),
                (3, 4),
                noerror(),
            ),
            (
                [[SequenceElement() for _ in range(4)] for _ in range(15)],
                ("a", "b"),
                dict(a=np.arange(15), b=np.arange(4)),
                (15, 4),
                noerror(),
            ),
            (
                [SequenceElement() for _ in range(10)],
                ("a", "b"),
                dict(),
                (10,),
                pytest.raises(ValueError),
            ),
            (
                [SequenceElement() for _ in range(5)],
                ("a",),
                dict(a=np.arange(10)),
                (5),
                pytest.raises(ValueError),
            ),
            (
                [[SequenceElement() for _ in range(5)] for _ in range(4)],
                ("a", "a"),
                dict(),
                (4, 5),
                pytest.raises(ValueError),
            ),
        ],
    )
    def test_explicit_constructor(self, elements, names, labels, shape, error):
        with error:
            s = Sequence(elements, names=names, **labels)

            # Check shape matches
            assert s.shape == shape

            if names:
                assert s.names == names

            if labels:
                assert s.labels.keys() == labels.keys()
                # Labels are not copied
                for l1, l2 in zip(s.labels.values(), labels.values()):
                    assert l1 is l2

    def test_label_conversion(self):
        s = Sequence([SequenceElement() for _ in range(3)], names=("a",), a=[1, 2, 3])

        assert isinstance(s.labels["a"], np.ndarray)
        assert_array_equal(s.labels["a"], np.array([1, 2, 3]))

    @pytest.mark.parametrize(
        "arr",
        [
            np.array(
                [[SequenceElement() for _ in range(3)] for _ in range(2)],
            ),
            Sequence.empty((2, 3), names=("a", "b"), a=np.arange(2)),
        ],
    )
    def test_constructor_pass_through(self, arr):
        s = Sequence(arr, names=("c", "d"), c=np.zeros(2), d=np.ones(3))

        assert s.names == ("c", "d")
        assert_array_equal(s.labels["c"], np.zeros(2))
        assert_array_equal(s.labels["d"], np.ones(3))

        assert_array_equal(s, arr)
        assert s.base is arr

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

        assert_array_equal(sequence, s1)
        assert sequence.names == s1.names
        assert sequence.labels.keys() == s1.labels.keys()

        for l1, l2 in zip(sequence.labels.values(), s1.labels.values()):
            # Labels are not copied
            assert l1 is l2

        assert s1.base is sequence

    @pytest.mark.parametrize(
        "shape,names,labels",
        [
            (10, None, dict()),
            (
                (3, 4, 5),
                ("a", "b", "c"),
                dict(a=np.arange(3), b=np.arange(4), c=np.arange(5)),
            ),
            ((1, 1), None, dict()),
        ],
    )
    def test_empty(self, shape, names, labels):
        sequence = Sequence.empty(shape, names=names, **labels)

        if not isinstance(shape, tuple):
            shape = (shape,)

        assert sequence.shape == shape

        obj_ids = set()
        for index in np.ndindex(*shape):
            se = sequence[index]
            obj_ids.add(id(se))

        # Checks that all sequence elements are unique/separate objects
        assert len(obj_ids) == np.product(shape)

        if names:
            assert sequence.names == names

        if labels:
            for v1, v2 in zip(sequence.labels.values(), labels.values()):
                assert v1 is v2


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

        assert s[index].shape == expected_shape
        assert s.labels == dict()

    @pytest.mark.parametrize(
        "shape,names,index,expected",
        [
            ((3, 4, 5), ("a", "b", "c"), np.s_[0, :, :], ("b", "c")),
            ((3, 4), ("a", "b"), np.s_[:, np.newaxis, :], ("a", None, "b")),
            (
                (3, 4, 5),
                ("a", "b", "c"),
                np.s_[np.newaxis, 0, ..., :],
                (None, "b", "c"),
            ),
        ],
    )
    def test_names_from_index(self, shape, names, index, expected):
        s = Sequence.empty(shape, names=names)

        expanded = s._expand_basic_index(index)
        updated_names = s._get_names_from_index(expanded)

        assert updated_names == expected

    @pytest.mark.parametrize(
        "shape,names,index,expected",
        [
            ((5,), ("a",), np.s_[:2], dict(a=np.arange(5)[:2])),
            (
                (1, 2, 10),
                (None, None, "c"),
                np.s_[:, :, 1::2],
                dict(c=np.arange(10)[1::2]),
            ),
            ((5, 4, 3), ("a", "b", "c"), np.s_[-3:, 1, 1], dict(a=np.arange(5)[-3:])),
            ((10, 10, 5), ("a", None, "c"), np.s_[0], dict(c=np.arange(5))),
            (
                (10, 11, 12),
                ("a", "b", "c"),
                np.s_[...],
                dict(a=np.arange(10), b=np.arange(11), c=np.arange(12)),
            ),
            ((3, 4, 5), ("a", "b", "c"), np.s_[:, 2, ..., 3], dict(a=np.arange(3))),
            (
                (4, 8, 2),
                ("a", "b", "c"),
                np.s_[0, :, :],
                dict(b=np.arange(8), c=np.arange(2)),
            ),
            (
                (4, 4),
                ("a", "b"),
                np.s_[np.newaxis, :, ::2],
                dict(a=np.arange(4), b=np.arange(4)[::2]),
            ),
        ],
    )
    def test_basic_indexing_with_names_and_labels(self, shape, names, index, expected):
        labels = {n: np.arange(dim) for n, dim in zip(names, shape) if n}

        s = Sequence.empty(shape, names=names, **labels)
        view = s[index]

        assert view.labels.keys() == expected.keys()
        for n in view.labels:
            assert_array_equal(view.labels[n], expected[n])

        # Check that view labels are also views of the original labels
        for n in view.labels:
            assert s.labels[n] is view.labels[n].base


class TestSequenceShaping:
    def test_reshape(self, request):
        s = Sequence.empty((4, 5, 3), names=("a", "b", "c"), a=np.arange(4))

        r = s.reshape(2, -1)
        assert r.shape == (2, 30)
        assert r.names == (None, None)
        assert r.labels == {}
        assert r.base is s

    @pytest.mark.parametrize(
        "seq,axes,shape,names",
        [
            (
                Sequence.empty((1, 2, 3, 4), ("a", "b", "c", "d")),
                None,
                (4, 3, 2, 1),
                ("d", "c", "b", "a"),
            ),
            (
                Sequence.empty((1, 2, 3, 4), ("a", "b", "c", "d")),
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

    @pytest.mark.parametrize(
        "shape,names", [((2,), ("a",)), ((1, 2, 3), ("a", "b", "c"))]
    )
    def test_T(self, shape, names):
        s = Sequence.empty(shape, names=names)

        assert s.T.shape == tuple(reversed(shape))
        assert s.T.names == tuple(reversed(names))

    def test_broadcast(self):
        s = Sequence.empty((2, 4), names=("x", "a"), a=np.arange(4), x=np.arange(2))
        r = Sequence.empty((4, 1, 1), names=("b", "x", ...), x=np.arange(1))

        names, labels = broadcast_names_and_labels(s, r)
        assert names == ["b", "x", "a"]

        expected_labels = dict(a=np.arange(4))
        assert labels.keys() == expected_labels.keys()

        for k in labels:
            assert_array_equal(labels[k], expected_labels[k])


class TestSequenceJoins:
    @pytest.mark.parametrize(
        "seqs,axis,names,labels,error",
        [
            (
                [Sequence.empty(5), Sequence.empty(6), Sequence.empty(7)],
                0,
                (None,),
                dict(),
                noerror(),
            ),
            (
                [
                    Sequence.empty((5, 2), names=("d0", ...), d0=np.arange(5)),
                    Sequence.empty((5, 3)),
                ],
                1,
                ("d0", None),
                dict(d0=np.arange(5)),
                noerror(),
            ),
            (
                [
                    Sequence.empty((5, 2), names=(..., "d1"), d1=np.arange(2)),
                    Sequence.empty((5, 3), names=("d0", "d1"), d1=np.arange(2, 5)),
                ],
                1,
                ("d0", "d1"),
                dict(d1=np.arange(5)),
                noerror(),
            ),
            (
                [
                    Sequence.empty((5, 1, 2), names=("d0", ..., "d2"), d0=np.arange(5)),
                    Sequence.empty(
                        (5, 1, 3), names=("d0", "d1", "d2"), d0=np.arange(5)
                    ),
                    Sequence.empty((5, 1, 4), names=("d0", ...), d0=np.arange(5)),
                ],
                2,
                ("d0", "d1", "d2"),
                dict(d0=np.arange(5)),
                noerror(),
            ),
            (
                [
                    Sequence.empty((5, 2), names=("d0", "d1"), d1=np.arange(2)),
                    Sequence.empty((5, 3), names=("d0", "d1"), d1=np.arange(2, 5)),
                ],
                -1,
                ("d0", "d1"),
                dict(d1=np.arange(5)),
                noerror(),
            ),
        ],
    )
    def test_concatenate(self, seqs, axis, names, labels, error):
        with error:
            c = np.concatenate(seqs, axis=axis)

            if axis < 0:
                axis = len(seqs[0].shape) + axis
            shape = []
            for dim in range(len(seqs[0].shape)):
                if dim == axis:
                    shape.append(sum(s.shape[dim] for s in seqs))
                else:
                    shape.append(seqs[0].shape[dim])

            assert c.shape == tuple(shape)
            assert c.names == names
            assert c.labels.keys() == labels.keys()

            for v1, v2 in zip(c.labels.values(), labels.values()):
                assert_array_equal(v1, v2)

    @pytest.mark.parametrize(
        "seqs,axis,names,labels,error",
        [
            (
                [Sequence.empty(10, names=("a",), a=np.arange(10)), Sequence.empty(10)],
                -1,
                ("a", None),
                dict(a=np.arange(10)),
                noerror(),
            ),
            (
                [
                    Sequence.empty((2, 4, 5), names=("a", ...), a=np.arange(2)),
                    Sequence.empty((2, 4, 5), names=(..., "d"), d=np.arange(5)),
                    Sequence.empty((2, 4, 5), names=(None, "c", None), c=np.arange(4)),
                ],
                1,
                ("a", None, "c", "d"),
                dict(a=np.arange(2), c=np.arange(4), d=np.arange(5)),
                noerror(),
            ),
        ],
    )
    def test_stack(self, seqs, axis, names, labels, error):
        with error:
            c = np.stack(seqs, axis=axis)

            shape = list(seqs[0].shape)
            axis = len(shape) + 1 + axis if axis < 0 else axis
            shape.insert(axis, len(seqs))

            assert c.shape == tuple(shape)
            assert c.names == names

            # Ordering is not guaranteed
            assert set(c.labels.keys()) == set(labels.keys())

            for n, label in c.labels.items():
                assert_array_equal(label, labels[n])

    def test_stack_with_sequence_data(self):
        a = Sequence.empty((3, 4), names=("b", "c"), b=np.arange(3), c=np.arange(4))
        b = Sequence.empty((3, 4), names=("b", "c"), c=np.arange(4))

        c = stack((a, b), axis=-3, name="a", label=np.arange(2))

        assert c.shape == (2, 3, 4)
        assert c.names == ("a", "b", "c")
        assert set(c.labels.keys()) == set("abc")

        for i, n in enumerate("abc"):
            assert_array_equal(c.labels[n], np.arange(i + 2))


class TestSequenceUniversalFunctions:
    @pytest.mark.parametrize(
        "a,b,shape,names,labels",
        [
            (
                Sequence.empty((10,), names=("a",), a=np.arange(10)),
                Sequence.empty((10,)),
                (10,),
                ("a",),
                dict(a=np.arange(10)),
            ),
            (
                Sequence.empty((3, 2, 1), names=("a", ...), a=np.arange(3)),
                Sequence.empty((1, 2, 3), names=(..., "b", "c"), c=np.arange(3)),
                (3, 2, 3),
                ("a", "b", "c"),
                dict(a=np.arange(3), c=np.arange(3)),
            ),
            (
                Sequence.empty((2, 3, 1), names=(..., "b", "c"), c=np.ones(1)),
                Sequence.empty((4,), names=("c",), c=np.ones(4)),
                (2, 3, 4),
                (None, "b", "c"),
                dict(c=np.ones(4)),
            ),
            (
                Sequence.empty(4, names=("a",), a=np.arange(4)),
                Sequence.empty(4, names=("a",), a=np.ones(4)),
                (4,),
                ("a",),
                dict(),
            ),
        ],
    )
    def test_ufunc_call(self, a, b, shape, names, labels):
        c = a + b

        assert c.shape == shape
        assert c.names == names
        assert set(c.labels.keys()) == set(labels.keys())

        for n, label in c.labels.items():
            assert_array_equal(label, labels[n])

    @pytest.mark.parametrize(
        "seq,kwargs,shape,names,labels",
        [
            (
                Sequence.empty((10,), names=("a",), a=np.arange(10)),
                {},
                tuple(),
                tuple(),
                {},
            ),
            (
                Sequence.empty(
                    (3, 2, 1), names=("a", "b", "c"), a=np.arange(3), b=np.arange(2)
                ),
                dict(axis=1),
                (3, 1),
                ("a", "c"),
                dict(a=np.arange(3)),
            ),
            (
                Sequence.empty(
                    (3, 2), names=("a", "b"), a=np.arange(3), b=np.arange(2)
                ),
                dict(axis=1, keepdims=True),
                (3, 1),
                ("a", "b"),
                dict(a=np.arange(3)),
            ),
        ],
    )
    def test_ufunc_reduce(self, seq, kwargs, shape, names, labels):
        result = np.sum(seq, **kwargs)

        assert result.shape == shape
        assert result.names == names
        assert set(result.labels.keys()) == set(labels.keys())

        for n, label in result.labels.items():
            assert_array_equal(label, labels[n])
