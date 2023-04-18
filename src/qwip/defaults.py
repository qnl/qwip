"""Dynamic defaults."""

import functools
import inspect
from typing import Callable, Dict

from attr import field
from loguru import logger

from qwip import qsettings
from qwip.attrs import qdefine
from qwip.settings import Settings


@qdefine(repr=False)
class QWiPDefault:
    """A class for representing dynamic default parameters.

    This enables `help(some_function)` to print both the `qsettings` key
    referenced by the function and the current value stored in `qsettings`.
    """

    settings_: Settings = qsettings
    path: str

    def __repr__(self) -> str:
        return f"QWiPDefault({self.path}={repr(self.settings_[self.path])})"


class dynamic_default:
    """A function decorator for setting configurable function defaults.

    This decorator takes a variable number of keyword arguments.

    Args:
        **kwargs: Keyword arguments specifying the argument to add a dynamic
            default to and the `qsettings` path from which the default should
            be taken from.

    Examples:
        TODO
    """

    def __init__(self, **kwargs):
        self.__settings__ = kwargs.pop("__settings__", qsettings)
        self.dynamic_kwargs = kwargs

    def __call__(self, func: Callable):
        signature = inspect.signature(func)
        new_signature = None

        valid_updates = {}

        for name, default in self.dynamic_kwargs.items():
            if name not in signature.parameters:
                raise ValueError(f'Parameter "{name}" is not a function parameter.')

            if default not in self.__settings__:
                raise ValueError(
                    f'Default "{default}" for parameter "{name}" is not a valid '
                    f"key in {self.__settings__}."
                )

            allowed = [
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ]
            if (k := signature.parameters[name].kind) not in allowed:
                raise ValueError(
                    f"Can only create dynamic default for keyword-only "
                    f'or positional or keyword parameters. "{name}" is '
                    f"{k}."
                )

            valid_updates.update({name: default})

        new_args = []
        for name, arg in signature.parameters.items():
            if name in valid_updates:
                default = QWiPDefault(
                    path=valid_updates[name], settings_=self.__settings__
                )
                arg = arg.replace(default=default)
                logger.debug(
                    f'Adding default {default} for "{name}" in {func.__name__}'
                )

            new_args += [arg]

        new_signature = signature.replace(parameters=new_args)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            arglen = len(args)

            for i, (name, param) in enumerate(new_signature.parameters.items()):
                if (
                    i >= arglen
                    and name not in kwargs
                    and isinstance(param.default, QWiPDefault)
                ):
                    kwargs.update({name: self.__settings__[param.default.path]})

            return func(*args, **kwargs)

        wrapper.__signature__ = new_signature

        return wrapper
