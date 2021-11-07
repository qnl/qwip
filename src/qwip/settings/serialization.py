"""Serializers for settings."""

from enum import Enum
from collections.abc import Mapping
from numbers import Number
from typing import Type, TypeVar, Union

import attr
import pendulum
import numpy as np

from loguru import logger

from qwip.settings.base import SettingsBase
from qwip.settings.typing import get_class_from_type, get_args, typedispatch, is_optional
from qwip.settings.parameters import Parameters

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
    structure_element = structure(element_type, field) if not isinstance(element_type, TypeVar) else (lambda x: x)
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

    kvtypes = get_args(field.type)
    structure_value = structure(kvtypes[1], field) if kvtypes and not isinstance(kvtypes[1], TypeVar) else (lambda x: x)
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
            return np.array(maybe_ndarray, dtype=field.metadata.get('dtype', None))
    return _structure

def add_type_converters(cls, fields):
    new_fields = []

    for field in fields:
        class_set = set(get_class_from_type(field.type).keys())

        type_converter = structure(class_set.pop(), field) if len(class_set) == 1 else None

        if is_optional(field.type) and type_converter:
            type_converter = attr.converters.optional(type_converter)

        if field.converter is not None:
            type_converter = attr.converters.pipe(field.converter, type_converter)

        field = field.evolve(converter=type_converter)

        new_fields.append(field)

    return new_fields