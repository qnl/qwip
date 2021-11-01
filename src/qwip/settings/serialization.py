"""Serializers for settings."""

from functools import singledispatch, update_wrapper
from enum import Enum
from collections.abc import Mapping
from numbers import Number
from typing import Union

import attr
import pendulum
import numpy as np

from loguru import logger

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

@structure.register(pendulum.Date)
def _(field_type, field):
    logger.debug(
        f'Creating Date converter for field "{field.name}" of type {field_type.__name__}'
    )
    def _structure(maybe_str):
        if isinstance(maybe_str, str):
            return pendulum.parse(maybe_str).date()
        else:
            return maybe_str
    return _structure

@structure.register(pendulum.DateTime)
def _(field_type, field):
    logger.debug(
        f'Creating Date converter for field "{field.name}" of type {field_type.__name__}'
    )
    def _structure(maybe_str):
        if isinstance(maybe_str, str):
            return pendulum.parse(maybe_str)
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
            return np.array(maybe_ndarray, dtype=field.metadata['dtype'])
    return _structure

def add_type_converters(cls, fields):
    new_fields = []

    for field in fields:
        optional = False

        type_origin = get_origin(field.type)
        type_args = get_args(field.type)

        if type_origin is Union:
            field_types = list(type_args)
            optional = type(None) in field_types
            field_types.remove(type(None))

        elif type_origin is not None:
            field_types = [type_origin]
        else:
            field_types = [field.type]
        
        # Can't meaningfully convert union types
        type_converter = structure(field_types[0], field) if len(field_types) == 1 else None

        if optional and type_converter:
            type_converter = attr.converters.optional(type_converter)

        if field.converter is not None:
            type_converter = attr.converters.pipe(field.converter, type_converter)

        field = field.evolve(converter=type_converter)

        new_fields.append(field)

    return new_fields