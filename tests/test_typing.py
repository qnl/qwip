import pytest

from collections.abc import Sequence, Callable, Mapping
from typing import (
    Any,
    Annotated,
    Optional,
    Union,
    get_args
)

import numpy as np
from numpy.typing import NDArray

from qwip.typing import (
    typedispatch,
    issubtype,
    is_annotated_type,
    is_callable_type,
    is_union_type,
    is_optional_type,
    is_generic_type,
    is_iterable_type,
    is_mapping_type,
    is_ndarray_type,
)

class TestTypeDispatch:

    @pytest.mark.parametrize(
        'tp,expect',
        [
            (int, 'number'),
            (str, 'string'),
            (float, 'number'),
            (complex, 'default'),
            (Any, 'anything')
        ]
    )
    def test_simple(self, tp, expect):
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

        @get_type.register(Any)
        def _(fieldtype):
            return 'anything'

        assert get_type(tp) == expect

    @pytest.mark.parametrize(
        'tp,expect',
        [
            (dict[str, int], 'dictionary'),
            (dict[int, str], 'dictionary'),
            (Mapping[str, str], 'generic'),
            (NDArray[np.float64], 'numpy'),
            (Annotated[str, 'annotation'], 'annotated'),
            (Callable[..., int], 'function'),
            (str | int, 'generic')
        ]
    )
    def test_generics_dispatch(self, tp, expect):
        @typedispatch
        def get_origin_type(fieldtype):
            # Check that typedispatch passes the original type arg to the
            # implemented function
            assert fieldtype == tp
            return 'generic'

        @get_origin_type.register(dict)
        def _(fieldtype):
            assert fieldtype == tp
            return 'dictionary'

        @get_origin_type.register(np.ndarray)
        def _(fieldtype):
            assert fieldtype == tp
            return 'numpy'

        @get_origin_type.register(Callable)
        def _(fieldtype):
            assert fieldtype == tp
            return 'function'

        @get_origin_type.register(Annotated)
        def _(fieldtype):
            assert fieldtype == tp
            return 'annotated'

        assert get_origin_type(tp) == expect

    @pytest.mark.parametrize(
        'tp,expect',
        [
            (dict[str, int], 'dictionary'),
            (dict[int, str], 'dictionary'),
            (Mapping[str, str], 'generic'),
            (NDArray[np.float64], 'numpy'),
            (Annotated[str, 'annotation'], 'annotated'),
            (Callable[..., int], 'function'),
        ]
    )
    def test_generics_register(self, tp, expect):
        @typedispatch
        def get_origin_type(fieldtype):
            return 'generic'

        @get_origin_type.register(dict[tuple, tuple])
        def _(fieldtype):
            return 'dictionary'

        @get_origin_type.register(NDArray[np.float64])
        def _(fieldtype):
            return 'override me'

        @get_origin_type.register(NDArray[np.int32])
        def _(fieldtype):
            return 'numpy'

        @get_origin_type.register(Callable[..., str])
        def _(fieldtype):
            return 'function'

        @get_origin_type.register(Annotated[int, 'annotation'])
        def _(fieldtype):
            return 'annotated'

        assert get_origin_type(tp) == expect

    @pytest.mark.parametrize(
        'tp,expect',
        [
            (list | dict, 'types.UnionType'),
            (set | None, 'optional types.UnionType'),
            (Optional[str], 'optional typing.Union'),
            (Union[bool, int], 'typing.Union')
        ]
    )
    def test_union(self, tp, expect):
        @typedispatch
        def get_origin_type(fieldtype):
            return 'default'

        @get_origin_type.register(Union[str, int])
        def _(fieldtype):
            return 'override me'

        @get_origin_type.register(Optional[int])
        def _(fieldtype):
            if type(None) in get_args(fieldtype):
                return 'optional typing.Union'
            else:
                return 'typing.Union'


        @get_origin_type.register(str | int)
        def _(fieldtype):
            if type(None) in get_args(fieldtype):
                return 'optional types.UnionType'
            else:
                return 'types.UnionType'

        assert get_origin_type(tp) == expect

@pytest.mark.parametrize(
    'subtype,tp,expect',
    [
        (dict, Mapping, True),
        (bool, int, True),
        (Annotated, Mapping, False)
    ]
)
def test_issubtype(subtype, tp, expect):
    assert issubtype(subtype, tp) == expect

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
        (Optional[str], True),
        (str, False),
        (str | int, True),
        (Annotated[str, 'annotation'], True),
        (Callable[..., Any], True),
        (Callable, False),
        (dict[str, int], True),
        (dict, False)
    ]
)
def test_generic(tp, expect):
    assert is_generic_type(tp) == expect

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
