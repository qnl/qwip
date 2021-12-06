import pytest

from typing import Any, Annotated, Optional, Union

import attr
import numpy as np

from numpy.random import default_rng
from numpy.testing import assert_array_equal
from attr import attrs, define, field
from cattr.converters import GenConverter
from cattr.gen import make_mapping_structure_fn, make_mapping_unstructure_fn

import qwip
from qwip.flatdict import FlatDict
from qwip.typing import NDArray

# https://docs.python.org/3/library/json.html
JSON_SERIALIZABLE = (dict, list, tuple, str, int, float, bool, type(None))

class SpecialFlatDict(FlatDict): ...

class TestFlatDict:
    TYPE_HINTS = [
        (FlatDict, FlatDict, None, None),
        (SpecialFlatDict, SpecialFlatDict, None, None),
        (FlatDict[str, float], FlatDict, str, float),
        (FlatDict[str, str], FlatDict, str, str),
        (FlatDict[str, Any], FlatDict, str, None),
        (FlatDict[str, Optional[int]], FlatDict, str, int),
        (Optional[FlatDict[str, bool]], FlatDict, str, bool)
    ]

    NESTED_OBJS = [
        {'a': 0, 'b': {'c': 1, 'd': 2}},
        {'a': {'b': True, 'c': {'d': {'e': False}}}},
        {'a': {'b': {'c': '1', 'd': 2.0, 'e': {'f': True}}}}
    ]

    CLASSES = [args[0] for args in TYPE_HINTS]

    @pytest.mark.parametrize('cls,mtype,ktype,vtype', TYPE_HINTS)
    @pytest.mark.parametrize('obj', [
        {},
        {'a': 0, 'b': 1},
        {'a': '0', 'b': 2},
        {'a': True, 'b': False}
    ])
    def test_simple_structure(self, obj, cls, mtype, ktype, vtype):
        struct = qwip.converter.structure(obj, cls) 

        assert type(struct) == mtype

        for k, v in struct.items():
            if ktype:
                assert type(k) == ktype
            if vtype:
                assert type(v) == vtype
            else:
                assert v is obj[k]

    @pytest.mark.parametrize('cls,mtype,ktype,vtype', TYPE_HINTS)
    @pytest.mark.parametrize('obj', NESTED_OBJS)
    def test_nested_structure(self, obj, cls, mtype, ktype, vtype):
        struct = qwip.converter.structure(obj, cls)

        assert type(struct) == mtype

        for k, v in struct.flatitems():
            if ktype:
                assert type(k) == ktype
            if vtype:
                assert type(v) == vtype

    @pytest.mark.parametrize('obj', NESTED_OBJS)
    def test_unstructure(self, obj):
        unstruct = qwip.converter.unstructure(FlatDict(obj))
        assert obj == unstruct

    @pytest.mark.parametrize('cls', CLASSES)
    @pytest.mark.parametrize('obj', NESTED_OBJS)
    def test_round_trip(self, obj, cls):
        """
        Unstructured objects from a previously structured input should always
        be structured the same way. This reverse does not necessarily hold.
        """
        flatdict = qwip.converter.structure(obj, cls)

        unstruct = qwip.converter.unstructure(flatdict)
        struct = qwip.converter.structure(unstruct, cls)

        assert flatdict == struct

    @pytest.mark.parametrize('cls', [
        FlatDict,
        Optional[FlatDict],
        FlatDict[str, Any]
    ])
    @pytest.mark.parametrize('obj', NESTED_OBJS)
    def test_pass_through(self, obj, cls):
        from functools import reduce

        struct = qwip.converter.structure(obj, cls)
        
        for k, v in struct.flatitems():
            assert v is reduce(lambda d, k: d[k], k.split('/'), obj)

    @pytest.mark.parametrize('obj', [
        ({'k1': {'a': 1}, 'k2': {'a': 1}}),
        ({'k1': {'a': 1, 'b': {'k': 2}}})
    ])
    def test_container_value(self, obj):
        @define
        class A:
            a: int
            b: FlatDict[str, int] = field(factory=FlatDict)
        
        struct = qwip.converter.structure(obj, FlatDict[str, A])
        
        for k, v in struct.items():
            assert isinstance(v, A)
            assert isinstance(v.b, FlatDict)

    @pytest.mark.parametrize(('obj', 'cls', 'level'), [
        ({'k1': {'a': 1}, 'k2/l1': {'a': 1}}, 'A', -1),
        ({'k1': {'b': {'a': 1}}, 'k2': {'l1': {'b': {'a': 1}}}}, 'B', -2)
    ])
    def test_annotated_class(self, obj, cls, level):
        @define
        class A:
            a: int

        @define
        class B:
            b: A

        cls = eval(cls)

        struct = qwip.converter.structure(
            obj,
            Annotated[FlatDict[str, cls], level]
        )
        
        for v in struct.flatvalues():
            assert isinstance(v, cls)
    
class TestNumpy:
    DTYPES = [
        np.int_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float_,
        np.float16,
        np.float32,
        np.float64,
        np.complex_,
        np.complex64,
        np.complex128,
        np.bool_,
        np.bool8
    ]

    @pytest.fixture
    def rng(self):
        return default_rng(12345)

    @pytest.mark.parametrize('cls', [
        NDArray,
        Optional[NDArray]
    ])
    @pytest.mark.parametrize('dtype', [None, Any] + DTYPES)
    def test_dtypes(self, cls, dtype):
        struct = qwip.converter.structure([1, 2, 3], cls[dtype] if dtype else cls)
        
        assert type(struct) == np.ndarray
        if dtype not in (None, Any):
            assert struct.dtype == dtype

    @pytest.mark.parametrize('dtype', DTYPES)
    def test_no_copy(self, dtype):
        arr = np.empty((5, 4), dtype=dtype)
        struct = qwip.converter.structure(arr, NDArray[dtype])

        assert struct is arr

    @pytest.mark.parametrize(('shape', 'dtype1', 'dtype2', 'copy'), [
        ((3, 3), np.int8, np.int16, True),
        ((4, 4), np.int8, np.int16, False),
        ((3, 3, 3), np.float64, np.int8, False),
    ])
    def test_change_dtype(self, shape, dtype1, dtype2, copy):
        arr = np.empty(shape, dtype=dtype1)
        struct = qwip.converter.structure(arr, NDArray[dtype2])

        assert (struct is arr) == (not copy)

    SIZES = [
        (10,),
        (1,),
        (1, 2, 3, 4, 5),
        (5, 4, 3)
    ]

    @pytest.mark.parametrize('size', SIZES)
    @pytest.mark.parametrize('dtype', [
        np.float64, np.int32, np.bool8
    ])
    def test_unstructure(self, rng, size, dtype):
        from itertools import product

        arr = rng.integers(10, size=size).astype(dtype)
        unstruct = qwip.converter.unstructure(arr)

        for idx in product(*(range(d) for d in size)):
            x = unstruct
            for i in idx:
                x = x[i]
            
            assert x == arr[idx]
            assert isinstance(x, JSON_SERIALIZABLE)

    @pytest.mark.parametrize('size', SIZES)
    @pytest.mark.parametrize('dtype', [
        np.float64, np.int32, np.bool8
    ])
    def test_round_trip(self, rng, size, dtype):
        arr = rng.integers(10, size=size).astype(dtype)
        unstruct = qwip.converter.unstructure(arr)
        struct = qwip.converter.structure(unstruct, NDArray[dtype])

        assert_array_equal(arr, struct)

    def test_numpy_flatdict(self):
        from functools import reduce
        obj = FlatDict({'a': np.zeros(2), 'b': np.ones((3, 3))})

        unstruct = qwip.converter.unstructure(obj)
        
        for flatkey in obj.flatkeys():
            arr = reduce(lambda d, k: d[k], flatkey.split('/'), unstruct)
            assert isinstance(arr, list)

        struct = qwip.converter.structure(unstruct, FlatDict[str, NDArray[np.int32]])

        print(struct['a'] is obj['a'])
        print(struct['b'] is obj['b'])
        print(struct)
        
        for k, arr in struct.flatitems():
            assert isinstance(arr, np.ndarray)
            assert_array_equal(arr, obj[k])

class TestAttrs:
    def test_null(self):
        @define
        class A: ...

        struct = qwip.converter.structure(dict(), A)
        assert struct == A()

        unstruct = qwip.converter.unstructure(struct)
        assert unstruct == dict()

    def test_simple(self):
        @define
        class A:
            a: int
            b: float
            c: bool

        struct = qwip.converter.structure(dict(a=1, b=1, c=1), A)

        for f in attr.fields(A):
            assert isinstance(getattr(struct, f.name), f.type)

        unstruct = qwip.converter.unstructure(struct)

        assert unstruct == dict(a=1, b=1.0, c=True)

    def test_omit_fields(self):
        @define
        class A:
            a: int = 0
            b: bool = field(default=False, metadata=dict(serialize=False))
            c: float = field(init=False)

            @c.default
            def _init_c(self):
                return 0.5 ** self.a

        a = A()

        unstruct = qwip.converter.unstructure(a)
        assert unstruct == dict(a=0)
        
        struct = qwip.converter.structure(unstruct, A)
        assert struct == a
    
    def test_already_structured(self):
        @define
        class A:
            a: str

        a = A(a='a')
        
        struct = qwip.converter.structure(a, A)
        assert struct is a

    def test_attrs_converter(self):
        @define
        class A:
            a: str = field(converter=lambda x: str(x).upper())

        struct = qwip.converter.structure(dict(a='lowercase'), A)

        assert struct.a == 'LOWERCASE'

    def test_optional(self): ...

    def test_numpy_fields(self):
        from typing import Annotated
        from qwip.settings.settings import Settings
        @define
        class A(Settings):
            a: NDArray[np.float32]

        struct = qwip.converter.structure(dict(a=[[0, 0], [0, 0]]), A)
        unstruct = qwip.converter.unstructure(struct)

        assert isinstance(struct.a, np.ndarray)
        assert isinstance(unstruct['a'], list)