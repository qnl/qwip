import typing
import datetime
from collections.abc import Mapping
from numbers import Number

import attr
import pendulum
import numpy as np

# from attr import attrib
# from attr.validators import instance_of, in_, deep_iterable, deep_mapping
# from enum import Enum
# from typing import Type, List, Dict, Any

# from pendulum import Date, DateTime, instance

from loguru import logger

from qwip.settings.settings import Settings
from qwip.settings.parameters import Parameters
from qwip.settings.typing import get_class_from_type, is_optional, typedispatch, get_origin, get_args


def schema(cls):
    properties, required, definitions = collect_properties(cls)
        
    cls_schema = {
        'title': cls.__name__,
        'description': cls.__doc__,
        'type': 'object',
        'properties': properties,
    }
    
    if required:
        cls_schema['required'] = required
    
    if definitions:
        cls_schema['definitions'] = definitions
    
    return cls_schema

def collect_properties(cls, definitions=None):
    properties = {}
    required = []
    if definitions is None:
        definitions = {}

    for f in attr.fields(cls):
        fschema = field_schema(f, definitions)

        is_settings = False
        if get_origin(f.type):
            is_settings = is_optional(f.type) and issubclass(get_args(f.type)[0], Settings)
        else:
            is_settings = issubclass(f.type, Settings)

        if is_settings:
            base_type = get_args(f.type)[0] if is_optional(f.type) else f.type
            properties[f.name] = {'$ref': f'#/definitions{base_type.__name__}'}
            if f.name not in definitions:
                definitions[base_type.__name__] = fschema
        else:
            properties[f.name] = fschema
            
        if f.default == attr.NOTHING and not is_optional(f.type):
            required.append(f.name)
            
    return properties, required, definitions

def field_schema(field, definitions=None): 
    fschema = {
        'title': field.name,
    }
    
    add_description(fschema, field)
    process_types(fschema, field, definitions=definitions)
    
    if field.default != attr.NOTHING:
        fschema['default'] = field.default

    return fschema

def add_description(fschema, field):
    fieldtype = None
    if get_origin(field.type):
        if is_optional(field.type):
            fieldtype = get_args(field.type)[0]
    else:
        fieldtype = field.type

    default = fieldtype.__doc__ if fieldtype and issubclass(fieldtype, Settings) else None
    description = field.metadata.get('description', default)
    if description:
        fschema['description'] = description

def process_types(fschema, field, definitions=None):
    class_type_map = get_class_from_type(field.type)
    json_types = set(get_json_type(cls) for cls in class_type_map)

    fschema['type'] = list(json_types) if len(json_types) > 1 else json_types.pop()

    for cls, tp in class_type_map.items():
        add_type_specific_properties(cls, fschema, field, tp, definitions=definitions)


@typedispatch
def get_json_type(field_type):
    return 'object'

@get_json_type.register(type(None))
def _(field_type):
    return 'null'

@get_json_type.register(bool)
def _(field_type):
    return 'boolean'

@get_json_type.register(Number)
def _(field_type):
    return 'number'

@get_json_type.register(str)
@get_json_type.register(datetime.datetime)
@get_json_type.register(datetime.date)
@get_json_type.register(datetime.time)
@get_json_type.register(datetime.timedelta)
def _(field_type):
    return 'string'

@get_json_type.register(np.ndarray)
@get_json_type.register(list)
def _(field_type):
    return 'array'

@typedispatch
def add_type_specific_properties(clstype, fschema, field, tp, **kwargs):
    logger.warning(f'No properties to add for clstype {clstype}')

@add_type_specific_properties.register(Mapping)
def _(clstype, fschema, field, tp, definitions=None):
    ## Add value types
    kt, vt = get_args(tp)

    json_types = set()
    if not isinstance(vt, typing.TypeVar):
        logger.debug(f'Adding types for {vt}')
        class_type_map = get_class_from_type(vt)
        
        for cls in class_type_map:
            json_types.add(get_json_type(cls))
        
    if json_types:
        fschema['additionalProperties'] = {
            'type': list(json_types) if len(json_types) > 1 else json_types.pop()
        }

@add_type_specific_properties.register(Settings)
def _(clstype, fschema, field, tp, definitions=None):
    properties, required, definitions = collect_properties(clstype, definitions)
    if properties:
        fschema['properties'] = properties
        
    if required:
        fschema['required'] = required

    fschema['additionalProperties'] = False

@add_type_specific_properties.register(list)
def _(clstype, fschema, field, tp, definitions=None):
    vt = get_args(tp)[0]

    json_types = set()
    if not isinstance(vt, typing.TypeVar):
        class_type_map = get_class_from_type(vt)

        for cls in class_type_map:
            json_types.add(get_json_type(cls))

    if json_types:
        fschema['items'] = {
            'type': list(json_types) if len(json_types) > 1 else json_types.pop()
        }

@add_type_specific_properties.register(Number)
def _(clstype, fschema, field, tp, definitions=None):
    pass

@add_type_specific_properties.register(bool)
def _(clstype, fschema, field, tp, definitions=None):
    pass

def add_string_format(fschema, format_specifier):
    if 'format' in fschema:
        fschema['anyOf'] = [dict(format=fschema['format'])]
        del fschema['format']
    
    if 'anyOf' in fschema:
        fschema['anyOf'] += [dict(format=format_specifier)]
    else:
        fschema['format'] = format_specifier

def get_regex_validator(field):
    validators = []
    try:
        validators = field.validator._validators
    except AttributeError:
        validators = [field.validator]

    for v in validators:
        if isinstance(v, attr.validators._MatchesReValidator):
            return v

    return None

@add_type_specific_properties.register(str)
def _(clstype, fschema, field, tp, definitions=None):
    regex_validator = get_regex_validator(field)
    if regex_validator:
        fschema['regex'] = regex_validator.regex.pattern

@add_type_specific_properties.register(datetime.datetime)
def _(clstype, fschema, field, tp, definitions=None):
    add_string_format(fschema, 'date-time')

@add_type_specific_properties.register(datetime.date)
def _(clstype, fschema, field, tp, definitions=None):
    add_string_format(fschema, 'date')

@add_type_specific_properties.register(datetime.time)
def _(clstype, fschema, field, tp, definitions=None):
    add_string_format(fschema, 'time')

@add_type_specific_properties.register(datetime.timedelta)
def _(clstype, fschema, field, tp, definitions=None):
    add_string_format(fschema, 'duration')

