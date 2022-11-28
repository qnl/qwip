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

def is_annotated(tp):
    return get_origin(tp) is Annotated

def is_callable(tp):
    if o := get_origin(tp):
        return issubtype(o, Callable)
    else:
        return issubtype(tp, Callable)

def is_union(tp):
    return get_origin(tp) in (Union, UnionType)

def is_optional(tp):
    return is_union(tp) and type(None) in get_args(tp)

def is_iterable(tp):
    if o := get_origin(tp):
        return issubtype(o, Iterable)
    else:
        return issubtype(tp, Iterable)

def is_mapping(tp):
    if o := get_origin(tp):
        return issubtype(o, Mapping)
    else:
        return issubtype(tp, Mapping)

def is_ndarray(tp):
    if o := get_origin(tp):
        return issubtype(o, np.ndarray)
    else:
        return issubtype(tp, np.ndarray)

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