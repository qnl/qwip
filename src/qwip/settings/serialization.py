"""Serializers for settings."""

from enum import Enum
from collections.abc import Mapping
from numbers import Number
from typing import Container, Iterable, Type, TypeVar, Union, Any
from pathlib import Path

import attr
import pendulum
import numpy as np

from loguru import logger

from qwip.settings.base import SettingsBase
from qwip.settings.typing import get_class_from_type, get_args, typedispatch, is_optional
from qwip.parameters import Parameters

@typedispatch
def structure(field_type: type, field: attr.Attribute):
    """Generic type based single dispatch converter for settings.

    Args:
        field_type (type): The field type.
        field (Attribute): The `attr` attribute that the converter will be
            added to.
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
@structure.register(Path)
def _(field_type, field):
    logger.debug(
        f'Creating Enum converter for field "{field.name}" of type {field_type.__name__}'
    )
    def _structure(maybe_str):
        return field_type(maybe_str)
    return _structure

@structure.register(tuple)
def _(field_type, field):
    def _structure(maybe_iter):
        if isinstance(maybe_iter, Iterable):
            return tuple(maybe_iter)

        return maybe_iter
    return _structure

@structure.register(list)
def _(field_type, field):
    logger.debug(
        f'Creating list converter for field "{field.name}" of type {field_type.__name__}'
    )
    try:
        element_type = get_args(field.type)[0]
    except IndexError:
        element_type = None
    structure_element = structure(element_type, field) if element_type else (lambda x: x)
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

    structure_value = None
    if kvtypes and not (isinstance(kvtypes[1], TypeVar) or kvtypes[1] is Any):
        class_set = set(get_class_from_type(kvtypes[1]).keys())
        structure_value = structure(class_set.pop(), field) if len(class_set) == 1 else None

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
        if isinstance(maybe_dict, SettingsBase):
            return maybe_dict
        
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
    dtype = None
    if args := get_args(field.type):
        dtype = get_args(args[1])[0]

    def _structure(maybe_ndarray):
        if isinstance(maybe_ndarray, np.ndarray):
            has_correct_dtype = dtype == Any or isinstance(dtype, TypeVar) or maybe_ndarray.dtype == dtype
            if has_correct_dtype:
                return maybe_ndarray
        print(field.type, type(dtype))
        return np.array(maybe_ndarray, dtype=None if dtype == Any else dtype)
    return _structure

def add_type_converters(cls, fields):
    new_fields = []

    for field in fields:
        if field.type == Any:
            new_fields.append(field)
            continue

        class_set = set(get_class_from_type(field.type).keys())

        type_converter = structure(class_set.pop(), field) if len(class_set) == 1 else None

        if is_optional(field.type) and type_converter:
            type_converter = attr.converters.optional(type_converter)

        if field.converter is not None and type_converter is not None:
            type_converter = attr.converters.pipe(field.converter, type_converter)

        if type_converter is not None:
            field = field.evolve(converter=type_converter)

        new_fields.append(field)

    return new_fields