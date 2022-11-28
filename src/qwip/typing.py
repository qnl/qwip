from loguru import logger

from inspect import isclass
from collections.abc import Iterable, Mapping, Callable
from types import UnionType
from typing import (
    Annotated,
    Union,
    get_args,
    get_origin,
)
from functools import singledispatch, update_wrapper

import numpy as np
from numpy.typing import NDArray as _NDArray

NDArray = Annotated[_NDArray, '']

def issubtype(tp, cls):
    return isclass(tp) and issubclass(tp, cls)

def is_annotated_type(tp):
    return get_origin(tp) is Annotated

def is_callable_type(tp):
    if o := get_origin(tp):
        return issubtype(o, Callable)
    else:
        return issubtype(tp, Callable)

def is_union_type(tp):
    return get_origin(tp) in (Union, UnionType)

def is_optional_type(tp):
    return is_union_type(tp) and type(None) in get_args(tp)

def is_generic_type(tp, origin_tp=None):
    if origin_tp is None:
        return NotImplementedError
    
    if o := get_origin(tp):
        return issubtype(o, origin_tp)
    else:
        return issubtype(tp, origin_tp)

def is_iterable_type(tp):
    return is_generic_type(tp, Iterable)

def is_mapping_type(tp):
    return is_generic_type(tp, Mapping)

def is_ndarray_type(tp):
    return is_generic_type(tp, np.ndarray)

def typedispatch(func):
    """Type-dispatch generic function decorator.

    This function modifies the singledispatch function from the functools module
    to allow for single dispatch converters based on types. This allows converters
    to be defined for different attributes when attrs classes are being created.
    """

    dispatcher = singledispatch(func)
    
    def wrapper(*args, **kwargs):
        if not args:
            raise TypeError(f'{funcname} requires at least '
                            '1 positional argument')
        return dispatcher.dispatch(args[0])(*args, **kwargs)
    
    funcname = getattr(func, '__name__', 'typedispatch function')
    wrapper.register = dispatcher.register
    wrapper.dispatcher = dispatcher
    update_wrapper(wrapper, func)
    return wrapper