import typing
import collections

from functools import singledispatch, update_wrapper

try:
    # Python >=3.8
    from typing import get_args
    from typing import get_origin
except ImportError:
    if hasattr(typing, '_GenericAlias'):  # Python 3.7

        def get_origin(tp):
            """Copied from the Python 3.8 typing module"""
            if isinstance(tp, typing._GenericAlias):
                return tp.__origin__
            if tp is typing.Generic: # type:ignore
                return typing.Generic # type:ignore
            return None

        def get_args(tp):
            """Copied from the Python 3.8 typing module"""
            if isinstance(tp, typing._GenericAlias):
                res = tp.__args__
                if (
                    get_origin(tp) is collections.abc.Callable
                    and res[0] is not Ellipsis
                ):
                    res = (list(res[:-1]), res[-1])
                return res
            return ()

def is_optional(field):
    return get_origin(field) is typing.Union and type(None) in get_args(field)

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