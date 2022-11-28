import pytest

from collections.abc import Sequence, Callable, Mapping
from types import UnionType
from typing import (
    Annotated,
    Optional,
    Union,
    get_origin,
    get_args
)

import numpy as np
from numpy.typing import NDArray

from qwip.typing import (
    typedispatch,
    is_annotated_type,
    is_callable_type,
    is_union_type,
    is_optional_type,
    is_iterable_type,
    is_mapping_type,
    is_ndarray_type
)

def test_typedispatch():

    @typedispatch
    def get_type(fieldtype):
        return 'default'

    @get_type.register(int)
    @get_type.register(float)
    def _(fieldtype):
        return 'number'

    @get_type.register(str)
    def _(fieldtype):
        return 'string'

    assert get_type(object) == 'default'
    assert get_type(int) == 'number'
    assert get_type(float) == 'number'
    assert get_type(str) == 'string'

    # Note that bool is a subclass of int
    assert get_type(bool) == 'number'

@pytest.mark.parametrize(
    'tp,expect',
    [
        (Annotated[str, 'annotated'], True),
        (Annotated[Annotated[str, 'nested'], 'annotation'], True)
    ]
)
def test_annotated(tp, expect):
    assert is_annotated_type(tp) == expect

@pytest.mark.parametrize(
    'tp,expect',
    [
        (Union[int, str], True),
        (int | str, True),
        (Optional[bool], True),
        (int, False)
    ]
)
def test_union(tp, expect):
    assert is_union_type(tp) == expect

@pytest.mark.parametrize(
    'tp,expect',
    [
        (Optional[str], True),
        (Union[int, None], True),
        (bool | None, True),
        (str, False),
        (str | int, False)
    ]
)
def test_optional(tp, expect):
    assert is_optional_type(tp) == expect

@pytest.mark.parametrize(
    'tp,expect',
    [
        (list, True),
        (set, True),
        (dict, True),
        (list[int], True),
        (tuple[str, ...], True),
        (NDArray[np.float32], True),
        (bool, False),
        (Union[int, float], False)
    ]
)
def test_iterable(tp, expect):
    assert is_iterable_type(tp) == expect

@pytest.mark.parametrize(
    'tp,expect',
    [
        (dict, True),
        (Mapping, True),
        (dict[str, int], True),
        (Union[str, int], False),
        (tuple[str, ...], False),
        (Sequence, False)
    ]
)
def test_mapping(tp, expect):
    assert is_mapping_type(tp) == expect

@pytest.mark.parametrize(
    'tp,expect',
    [
        (np.ndarray, True),
        (NDArray, True),
        (NDArray[np.complex128], True),
        (list[int], False),
        (tuple[str, ...], False),
        (bool, False),
        (Union[int, float], False)
    ]
)
def test_ndarray(tp, expect):
    assert is_ndarray_type(tp) == expect

@pytest.mark.parametrize(
    'tp,expect',
    [
        (Callable, True),
        (Callable[..., int], True),
        (Callable[[str, int], int], True),
        (int, False),
        (int | str, False),
    ]
)
def test_callable(tp, expect):
    assert is_callable_type(tp) == expect
