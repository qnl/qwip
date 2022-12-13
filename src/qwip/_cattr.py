from typing import Any, ForwardRef, get_origin, get_args
from pathlib import Path
from numbers import Number

import attr
import cattr
from cattr.gen import make_dict_structure_fn, make_dict_unstructure_fn
import pendulum
import numpy as np
from numpy.typing import NDArray
from cattr import GenConverter

converter = GenConverter(prefer_attrib_converters=True)

# ========== builtin types ========== #
converter.register_structure_hook(
    bool,
    lambda v, cls: bool(v) if isinstance(v, (float, str, int)) else v
)

converter.register_structure_hook(
    Number,
    lambda v, cls: cls(v) if isinstance(v, Number) else v
)

converter.register_structure_hook(
    str,
    lambda v, cls: str(v) if isinstance(v, (int, float, bool)) else v
)

# ========== ForwardRef ========== #

def make_forward_ref_structure_fn(cls):
    def forward_ref_structure_fn(obj, cls):
        cls = cls._evaluate(None, None, set())
        return converter.structure(obj, cls)

    return forward_ref_structure_fn

def is_forward_ref(cls):
    return isinstance(cls, ForwardRef)

converter.register_structure_hook_factory(
    is_forward_ref,
    make_forward_ref_structure_fn
)

# ========== stdlib types ========== #

converter.register_unstructure_hook(
    Path,
    lambda p: str(p)
)

converter.register_structure_hook(
    Path,
    lambda p, _: Path(p)
)

# ========== numpy types ========== #
def isndarray(cls):
    return issubclass(get_origin(cls) or cls, np.ndarray)

def get_dtype(cls):
    if not (args := get_args(cls)):
        return None

    _, dtypelike = args

    if dtypelike in get_args(NDArray):
        dtype = None
    else:
        dtype, = get_args(dtypelike)

    if dtype == Any:
        dtype = None

    return dtype

def make_ndarray_structure_fn(cls):
    dtype = get_dtype(cls)

    def ndarray_structure_fn(obj, cls):
        if isinstance(obj, np.ndarray):
            if obj.dtype == dtype:
                return obj
            else:
                try:
                    obj.dtype = dtype
                    return  obj
                except ValueError:
                    pass # Copy array if changing dtype in place fails
        return np.array(obj, dtype=dtype)

    return ndarray_structure_fn

def make_ndarray_unstructure_fn(cls):
    return lambda arr: arr.tolist()

converter.register_structure_hook_factory(
    isndarray,
    make_ndarray_structure_fn
)

# Hook factory necesssary for unstructuring attrs classes with NDArray fields
converter.register_unstructure_hook_factory(
    isndarray,
    make_ndarray_unstructure_fn
)

# ========== pendulum types ========== #

# Pendulum parse only accepts strings.
converter.register_structure_hook(
    pendulum.DateTime,
    lambda v, cls: pendulum.parse(v) if isinstance(v, str) else v
)

converter.register_unstructure_hook(
    pendulum.DateTime,
    lambda dt: dt.isoformat()
)

converter.register_structure_hook(
    pendulum.Date,
    lambda v, cls: pendulum.parse(v).date()
)


converter.register_unstructure_hook(
    pendulum.Date,
    lambda dt: dt.isoformat()
)

# ========== attrs types ========== #

def make_attrs_structure_fn(cls):
    def should_structure(field):
        return field.init and field.metadata.get('serialize', True)

    to_structure = {
        f.name: cattr.override(omit=True) 
            for f in attr.fields(cls) if not should_structure(f)
    }

    structure_from_dict = make_dict_structure_fn(
        cls,
        converter,
        **to_structure
    )

    _cls = cls

    def structure_fn(v, cls):
        if isinstance(v, cls):
            return v
        else:
            return structure_from_dict(v, cls)

    return structure_fn

def make_attrs_unstructure_fn(cls):
    def should_unstructure(field):
        return field.init and field.metadata.get('serialize', True)
    
    to_unstructure = {
        f.name: cattr.override(omit=True) 
            for f in attr.fields(cls) if not should_unstructure(f)
    }
    unstructure_fn = make_dict_unstructure_fn(
        cls,
        converter,
        **to_unstructure
    )

    return unstructure_fn

converter.register_structure_hook_factory(
    attr.has,
    make_attrs_structure_fn
)

converter.register_unstructure_hook_factory(
    attr.has,
    make_attrs_unstructure_fn
)

