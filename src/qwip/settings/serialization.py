import attr
import pendulum

from loguru import logger

from pendulum import Date
from enum import Enum
from collections.abc import Mapping
from numbers import Number

from qwip.settings.base import SettingsBase
from qwip.settings.typing import get_origin, get_args
from qwip.settings.functools import singledispatch
from qwip.settings.parameters import Parameters

@singledispatch
def structure(field_type, field):
    logger.warning(
        f'No registered structure method for {field.name} of type {type(field_type).__name__}')
    return None

@structure.register(bool)
@structure.register(str)
@structure.register(Number)
def _(field_type, field):
    return None

@structure.register
def _(field_type: Enum, field):
    def _structure(maybe_str):
        return field_type(maybe_str)
    return _structure

@structure.register
def _(field_type: list, field):
    element_type = get_args(field.type)[0]
    structure_element = structure(element_type, field)
    def _structure(lst):
        for i, element in enumerate(lst):
            if structure_element:
                lst[i] = structure_element(element) # pylint: disable=not-callable
        return lst
    return _structure

@structure.register
def _(field_type: Mapping, field):
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

@structure.register
def _(field_type: SettingsBase, field):
    def _structure(maybe_dict):
        return field_type(**maybe_dict)
    return _structure

@structure.register
def _(field_type: Date, field):
    def _structure(maybe_str):
        if isinstance(maybe_str, str):
            return pendulum.parse(maybe_str).date()
        else:
            return maybe_str
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