from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, ForwardRef, Optional, Union, get_args

import numpy as np
import pytest
from numpy.typing import NDArray
from typing_extensions import Self

from qwip import FlatDict
from qwip.processing.processors import Labeled, StatePopulations
from qwip.typing import (
    generic_to_string,
    is_annotated_type,
    is_callable_type,
    is_generic_type,
    is_iterable_type,
    is_mapping_type,
    is_ndarray_type,
    is_optional_type,
    is_union_type,
    issubtype,
    replace_self_type,
    typedispatch,
)


class TestTypeDispatch:
    @pytest.mark.parametrize(
        "tp,expect",
        [
            (int, "number"),
            (str, "string"),
            (float, "number"),
            (complex, "default"),
            (Any, "anything"),
        ],
    )
    def test_simple(self, tp, expect):
        @typedispatch
        def get_type(fieldtype):
            return "default"

        @get_type.register(int)
        @get_type.register(float)
        def _(fieldtype):
            return "number"

        @get_type.register(str)
        def _(fieldtype):
            return "string"

        @get_type.register(Any)
        def _(fieldtype):
            return "anything"

        assert get_type(tp) == expect

    @pytest.mark.parametrize(
        "tp,expect",
        [
            (dict[str, int], "dictionary"),
            (dict[int, str], "dictionary"),
            (Mapping[str, str], "generic"),
            (NDArray[np.float64], "numpy"),
            (Annotated[str, "annotation"], "annotated"),
            (Callable[..., int], "function"),
            (str | int, "generic"),
        ],
    )
    def test_generics_dispatch(self, tp, expect):
        @typedispatch
        def get_origin_type(fieldtype):
            # Check that typedispatch passes the original type arg to the
            # implemented function
            assert fieldtype == tp
            return "generic"

        @get_origin_type.register(dict)
        def _(fieldtype):
            assert fieldtype == tp
            return "dictionary"

        @get_origin_type.register(np.ndarray)
        def _(fieldtype):
            assert fieldtype == tp
            return "numpy"

        @get_origin_type.register(Callable)
        def _(fieldtype):
            assert fieldtype == tp
            return "function"

        @get_origin_type.register(Annotated)
        def _(fieldtype):
            assert fieldtype == tp
            return "annotated"

        assert get_origin_type(tp) == expect

    @pytest.mark.parametrize(
        "tp,expect",
        [
            (dict[str, int], "dictionary"),
            (dict[int, str], "dictionary"),
            (Mapping[str, str], "generic"),
            (NDArray[np.float64], "numpy"),
            (Annotated[str, "annotation"], "annotated"),
            (Callable[..., int], "function"),
        ],
    )
    def test_generics_register(self, tp, expect):
        @typedispatch
        def get_origin_type(fieldtype):
            return "generic"

        @get_origin_type.register(dict[tuple, tuple])
        def _(fieldtype):
            return "dictionary"

        @get_origin_type.register(NDArray[np.float64])
        def _(fieldtype):
            return "override me"

        @get_origin_type.register(NDArray[np.int32])
        def _(fieldtype):
            return "numpy"

        @get_origin_type.register(Callable[..., str])
        def _(fieldtype):
            return "function"

        @get_origin_type.register(Annotated[int, "annotation"])
        def _(fieldtype):
            return "annotated"

        assert get_origin_type(tp) == expect

    @pytest.mark.parametrize(
        "tp,expect",
        [
            (list | dict, "types.UnionType"),
            (set | None, "optional types.UnionType"),
            (Optional[str], "optional typing.Union"),
            (Union[bool, int], "typing.Union"),
        ],
    )
    def test_union(self, tp, expect):
        @typedispatch
        def get_origin_type(fieldtype):
            return "default"

        @get_origin_type.register(Union[str, int])
        def _(fieldtype):
            return "override me"

        @get_origin_type.register(Optional[int])
        def _(fieldtype):
            if type(None) in get_args(fieldtype):
                return "optional typing.Union"
            else:
                return "typing.Union"

        @get_origin_type.register(str | int)
        def _(fieldtype):
            if type(None) in get_args(fieldtype):
                return "optional types.UnionType"
            else:
                return "types.UnionType"

        assert get_origin_type(tp) == expect


@pytest.mark.parametrize(
    "subtype,tp,expect",
    [(dict, Mapping, True), (bool, int, True), (Annotated, Mapping, False)],
)
def test_issubtype(subtype, tp, expect):
    assert issubtype(subtype, tp) == expect


@pytest.mark.parametrize(
    "tp,expect",
    [
        (Annotated[str, "annotated"], True),
        (Annotated[Annotated[str, "nested"], "annotation"], True),
    ],
)
def test_annotated(tp, expect):
    assert is_annotated_type(tp) == expect


@pytest.mark.parametrize(
    "tp,expect",
    [(Union[int, str], True), (int | str, True), (Optional[bool], True), (int, False)],
)
def test_union(tp, expect):
    assert is_union_type(tp) == expect


@pytest.mark.parametrize(
    "tp,expect",
    [
        (Optional[str], True),
        (Union[int, None], True),
        (bool | None, True),
        (str, False),
        (str | int, False),
    ],
)
def test_optional(tp, expect):
    assert is_optional_type(tp) == expect


@pytest.mark.parametrize(
    "tp,expect",
    [
        (Optional[str], True),
        (str, False),
        (str | int, True),
        (Annotated[str, "annotation"], True),
        (Callable[..., Any], True),
        (Callable, False),
        (dict[str, int], True),
        (dict, False),
    ],
)
def test_generic(tp, expect):
    assert is_generic_type(tp) == expect


@pytest.mark.parametrize(
    "tp,expect",
    [
        (list, True),
        (set, True),
        (dict, True),
        (list[int], True),
        (tuple[str, ...], True),
        (NDArray[np.float32], True),
        (bool, False),
        (Union[int, float], False),
    ],
)
def test_iterable(tp, expect):
    assert is_iterable_type(tp) == expect


@pytest.mark.parametrize(
    "tp,expect",
    [
        (dict, True),
        (Mapping, True),
        (dict[str, int], True),
        (Union[str, int], False),
        (tuple[str, ...], False),
        (Sequence, False),
    ],
)
def test_mapping(tp, expect):
    assert is_mapping_type(tp) == expect


@pytest.mark.parametrize(
    "tp,expect",
    [
        (np.ndarray, True),
        (NDArray, True),
        (NDArray[np.complex128], True),
        (list[int], False),
        (tuple[str, ...], False),
        (bool, False),
        (Union[int, float], False),
    ],
)
def test_ndarray(tp, expect):
    assert is_ndarray_type(tp) == expect


@pytest.mark.parametrize(
    "tp,expect",
    [
        (Callable, True),
        (Callable[..., int], True),
        (Callable[[str, int], int], True),
        (int, False),
        (int | str, False),
    ],
)
def test_callable(tp, expect):
    assert is_callable_type(tp) == expect


@pytest.mark.parametrize(
    "tp,expected",
    [
        (str, "str"),
        (list[str], "list[str]"),
        (tuple[bool, int, str], "tuple[bool, int, str]"),
        (FlatDict[str, int], "FlatDict[str, int]"),
        (Labeled[StatePopulations], "Labeled[StatePopulations]"),
    ],
)
def test_generic_to_string(tp, expected):
    assert generic_to_string(tp) == expected


TypeA = ForwardRef("A", module=__name__, is_class=True)


@pytest.mark.parametrize(
    "tp,expect",
    [
        (int, int),
        (Self, TypeA),
        (tuple[Self, ...], tuple[TypeA, ...]),
        (tuple[list[Self]], tuple[list[TypeA]]),
        (dict[str, Self], dict[str, TypeA]),
        (str | Self, Union[str, TypeA]),
    ],
)
def test_replace_self_type(tp, expect):
    class A: ...

    assert replace_self_type(tp, A) == expect
