import typing
import collections

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