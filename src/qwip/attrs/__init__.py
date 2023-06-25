import functools
from contextlib import contextmanager

import attrs
import numpy as np
import pandas as pd
from attrs import define, frozen
from attrs.validators import set_disabled

from qwip.attrs.schema import schema
from qwip.attrs.serialization import add_type_converters
from qwip.attrs.validation import add_type_validators


def qwip_field_transform(cls, fields):
    fields = add_type_converters(cls, fields)
    fields = add_type_validators(cls, fields)
    return fields


qdefine = functools.partial(
    define,
    auto_attribs=True,
    kw_only=True,
    field_transformer=qwip_field_transform,
    on_setattr=[attrs.setters.convert, attrs.setters.validate],
)

qfrozen = functools.partial(
    frozen,
    auto_attribs=True,
    kw_only=True,
    field_transformer=qwip_field_transform,
)


@contextmanager
def disable_validation():
    """Context manager for temporarily disabling validation on all attrs classes."""
    try:
        set_disabled(True)
        yield
    finally:
        set_disabled(False)


danger = disable_validation


def _dataframe_equals(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Checks if two dataframes are equal.

    This function should only ever be called with two DataFrame arguments.

    Returns:
        True if a and b are equal otherwise False.
    """
    return a.equals(b)


def _numpy_equals(a: np.ndarray, b: np.ndarray) -> bool:
    """Checks if two numpy arrays are equal.

    Returns:
        True if a and b are equal otherwise False.
    """
    if a is None and b is None:
        return True
    return (a == b).all()
