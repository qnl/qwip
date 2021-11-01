import attr
import pendulum

import numpy as np

from loguru import logger

from functools import singledispatch, update_wrapper

from pendulum import Date
from enum import Enum
from collections.abc import Mapping
from numbers import Number

from qwip.settings.base import SettingsBase
from qwip.settings.typing import get_origin, get_args
from qwip.settings.parameters import Parameters

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

@typedispatch
def structure(field_type, field):
    """Generic type based single dispatch converter for settings.

    Args:
        field_type (type): The field type.
        field: The fie
    """
    logger.warning(
        f'No registered structure method for field "{field.name}" of type {field_type.__name__}'
    )
    return None

@structure.register(bool)
@structure.register(str)
@structure.register(Number)
def _(field_type, field):
    logger.debug(
        f'Creating null converter for field "{field.name}" of type {field_type.__name__}'
    )
    return None

@structure.register(Enum)
def _(field_type, field):
    logger.debug(
        f'Creating Enum converter for field "{field.name}" of type {field_type.__name__}'
    )
    def _structure(maybe_str):
        return field_type(maybe_str)
    return _structure

@structure.register(list)
def _(field_type, field):
    logger.debug(
        f'Creating list converter for field "{field.name}" of type {field_type.__name__}'
    )
    element_type = get_args(field.type)[0]
    structure_element = structure(element_type, field)
    def _structure(lst):
        for i, element in enumerate(lst):
            if structure_element:
                lst[i] = structure_element(element) # pylint: disable=not-callable
        return lst
    return _structure

@structure.register(Mapping)
def _(field_type, field):
    logger.debug(
        f'Creating mapping converter for field "{field.name}" of type {field_type.__name__}'
    )
    value_type = get_args(field.type)[1]
    structure_value = structure(value_type, field)
    def _structure(d):
        if isinstance(d, Parameters):
            for k, v in d.items():
                if structure_value:
                    d[k] = structure_value(v) # pylint: disable=not-callable
            return d
        return Parameters({
            k: structure_value(v) if structure_value else v for k, v in d.items() # pylint: disable=not-callable
        })
    return _structure

@structure.register(SettingsBase)
def _(field_type, field):
    logger.debug(
        f'Creating Settings converter for field "{field.name}" of type {field_type.__name__}'
    )
    def _structure(maybe_dict):
        return field_type(**maybe_dict)
    return _structure

@structure.register(Date)
def _(field_type: type, field: Date):
    logger.debug(
        f'Creating Date converter for field "{field.name}" of type {field_type.__name__}'
    )
    def _structure(maybe_str):
        if isinstance(maybe_str, str):
            return pendulum.parse(maybe_str).date()
        else:
            return maybe_str
    return _structure

@structure.register(np.ndarray)
def _(field_type, field):
    logger.debug(
        f'Creating ndarray converter for field "{field.name}" of type {field_type.__name__}'
    )
    def _structure(maybe_ndarray):
        if isinstance(maybe_ndarray, np.ndarray):
            return maybe_ndarray
        else:
            return np.array(maybe_ndarray)
    return _structure

def add_type_converters(cls, fields):
    new_fields = []

    for field in fields:
        field_type = field.type
        if get_origin(field.type) is not None:
            field_type = get_origin(field.type)

        type_converter = structure(field_type, field)

        if field.converter is not None:
            type_converter = attr.converters.pipe(field.converter, type_converter)

        field = field.evolve(converter=type_converter)

        new_fields.append(field)

    return new_fields