from loguru import logger

from inspect import isclass
from collections.abc import Iterable, Mapping, Callable
from types import UnionType
from typing import (
    Annotated,
    Union,
    ForwardRef,
    get_args,
    get_origin,
)
from typing_extensions import Self
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
    return get_origin(tp) in {Union, UnionType}

def is_optional_type(tp, arg_tp=None):
    args = get_args(tp)
    is_opt = is_union_type(tp) and type(None) in args

    if arg_tp is None:
        return is_opt
    else:
        return is_opt and all(issubtype(a, arg_tp) for a in args if a is not None)

def is_generic_type(tp, origin_tp=None):
    if origin_tp is None:
        return get_origin(tp) is not None
    
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

def replace_self_type(tp, cls):
    """Replaces instances of Self type with a class ForwardRef

    This function recursively iterations through any arguments of the given
    type and replaces any instances of Self with a forward references with
    the class module specified.

    Args:
        tp: A type to replace instances of Self
        cls: The cls tp replace Self with

    Returns:
        A forward reference that references the class.
    """
    if tp is Self:
        return ForwardRef(cls.__name__, module=cls.__module__, is_class=True)

    if isinstance(tp, (str, ForwardRef)):
        return tp

    args = get_args(tp)
    orig = get_origin(tp)

    if orig:
        orig = replace_self_type(orig, cls)
        args = tuple(replace_self_type(a, cls) for a in args)
        
        if orig is UnionType:
            orig = Union

        return orig[args] # type: ignore

    return tp

def typedispatch(func):
    """Type-dispatch generic function decorator.

    This function modifies the singledispatch function from the functools module
    to allow for single dispatch converters based on types. This allows converters
    to be defined for different attributes when attrs classes are being created.
    """

    _dispatcher = singledispatch(func)
    # _dispatcher.registry is a read-only MappingProxy
    registry = {}

    def dispatch(tp: type) -> Callable:
        if origin := get_origin(tp):
            tp = origin

        try:
            impl = registry[tp]
        except KeyError:
            impl = _dispatcher.dispatch(tp) if isinstance(tp, type) else func

        return impl

    def register(tp: type, func: Callable | None = None) -> Callable:
        if func is None:
            return lambda f: register(tp, f)

        if origin := get_origin(tp):
            tp = origin

        # functools singledispatch can handle normal classes
        if not isinstance(tp, type):
            registry[tp] = func
        else:
            _dispatcher.register(tp, func)

        return func

    def wrapper(*args, **kwargs):
        if not args:
            raise TypeError(f'{funcname} requires at least 1 positional argument')
        return dispatch(args[0])(*args, **kwargs)
    
    funcname = getattr(func, '__name__', 'typedispatch function')
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry
    wrapper._dispatcher = _dispatcher
    update_wrapper(wrapper, func)
    return wrapper