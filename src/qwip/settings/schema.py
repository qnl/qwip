import typing
import datetime
from collections.abc import Mapping
from numbers import Number

import attr
import numpy as np

from loguru import logger

# from qwip.settings.settings import Settings
from qwip.settings.base import SettingsBase
from qwip.flatdict import FlatDict
from qwip.settings.typing import get_class_from_type, is_optional, typedispatch, get_origin, get_args


def schema(cls):
    properties, required, definitions = collect_properties(cls)
        
    cls_schema = {
        'title': cls.__name__,
        'description': cls.__doc__,
        'type': 'object',
        'properties': properties,
    }

    if not cls_schema['description']:
        cls_schema.pop('description')
    
    if required:
        cls_schema['required'] = required
    
    if definitions:
        cls_schema['definitions'] = definitions

    cls_schema['additionalProperties'] = False
    
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
            is_settings = is_optional(f.type) and issubclass(get_args(f.type)[0], SettingsBase)
        else:
            is_settings = (f.type != typing.Any) and issubclass(f.type, SettingsBase)

        if is_settings:
            base_type = get_args(f.type)[0] if is_optional(f.type) else f.type
            properties[f.name] = {'$ref': f'#/definitions/{base_type.__name__}'}
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
    add_enum(fschema, field)
    
    if field.default != attr.NOTHING:
        fschema['default'] = field.default

    return fschema

def add_description(fschema, field):
    fieldtype = None
    if get_origin(field.type):
        if is_optional(field.type):
            fieldtype = get_args(field.type)[0]
    elif field.type != typing.Any:
        fieldtype = field.type

    default = fieldtype.__doc__ if fieldtype and issubclass(fieldtype, SettingsBase) else None
    description = field.metadata.get('description', default)
    if description:
        fschema['description'] = description

def get_enum_validator(field):
    validators = []
    try:
        validators = field.validator._validators
    except AttributeError:
        validators = [field.validator]

    for v in validators:
        if isinstance(v, attr.validators._InValidator):
            return v

    return None

def add_enum(fschema, field):
    enum_validator = get_enum_validator(field)

    if enum_validator:
        fschema['enum'] = enum_validator.options

def process_types(fschema, field, definitions=None):
    if field.type == typing.Any:
        return

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


def get_json_validation(validator):
    if isinstance(validator, attr.validators._MatchesReValidator):
        return {'pattern': validator.pattern.pattern}
    elif isinstance(validator, attr.validators._InValidator):
        return {'enum': validator.options}
    # elif isinstance(validator, attr.validators._NumberValidator):
    #     op_text = {
    #         '<': 'exclusiveMaximum',
    #         '>': 'exclusiveMinimum',
    #         '<=': 'maximum',
    #         '>=': 'minimum'
    #     }
    #     return {op_text[validator.compare_op]: validator.bound}
    
    return {}

def get_iterable_validators(field):
    try:
        validators = field.validator._validators
    except AttributeError:
        validators = [field.validator]

    return [v for v in validators if isinstance(v, attr.validators._DeepIterable)]

def get_mapping_validators(field):
    try:
        validators = field.validator._validators
    except AttributeError:
        validators = [field.validator]

    return [v for v in validators if isinstance(v, attr.validators._DeepMapping)]

@add_type_specific_properties.register(Mapping)
def _(clstype, fschema, field, tp, definitions=None):
    """
    """
    ## Add value types
    try:
        kt, vt = get_args(tp)
    except ValueError:
        kt = None
        vt = None

    json_types = set()
    if vt is not None and vt != typing.Any:
        logger.debug(f'Adding types for {vt}')
        class_type_map = get_class_from_type(vt)
        
        for cls in class_type_map:
            json_types.add(get_json_type(cls))

    if json_types:
        json_types = list(json_types) if len(json_types) > 1 else json_types.pop()

    iter_validators = get_iterable_validators(field)
    pattern = None
    enum = None
    for v in iter_validators:
        mv = v.member_validator
        if isinstance(mv, attr.validators._InValidator):
            enum = mv.options
        elif isinstance(mv, attr.validators._MatchesReValidator):
            pattern = mv.pattern.pattern

    if enum and json_types:
        fschema['properties'] = {name: {'type': json_types} for name in enum}
    elif pattern and json_types:
        fschema['patternProperties'] = {pattern: {'type': json_types}}
    elif pattern:
        fschema['propertyNames'] = dict(pattern=pattern)
    
    pattern = None
    value_properties = {}
    map_validators = get_mapping_validators(field)
    for v in map_validators:
        kv = v.key_validator
        vv = v.value_validator
        if isinstance(kv, attr.validators._MatchesReValidator):
            pattern = kv.pattern.pattern
            value_properties = get_json_validation(vv)

    if json_types:
        value_properties.update(dict(type=json_types))
    
    if pattern and value_properties:
        pattern_properties = fschema.get('patternProperties', {})
        pattern_properties.update({pattern: value_properties})
        fschema['patternProperties'] = pattern_properties
    elif pattern:
        fschema['propertyNames'] = dict(pattern=pattern)
    elif value_properties and 'patternProperties' not in fschema:
        fschema['additionalProperties'] = value_properties

@add_type_specific_properties.register(SettingsBase)
def _(clstype, fschema, field, tp, definitions=None):
    properties, required, definitions = collect_properties(clstype, definitions)
    if properties:
        fschema['properties'] = properties
        
    if required:
        fschema['required'] = required

    fschema['additionalProperties'] = False

@add_type_specific_properties.register(list)
def _(clstype, fschema, field, tp, definitions=None):
    try:
        vt = get_args(tp)[0]
    except IndexError:
        vt = None

    json_types = set()
    if vt is not None:
        class_type_map = get_class_from_type(vt)

        for cls in class_type_map:
            json_types.add(get_json_type(cls))

    item_properties = {}
    if json_types:
        json_types = list(json_types) if len(json_types) > 1 else json_types.pop()
        item_properties.update(dict(type=json_types))

    iter_validators = get_iterable_validators(field)
    for v in iter_validators:
        mv = v.member_validator
        item_properties.update(get_json_validation(mv))

    if item_properties:
        fschema['items'] = item_properties

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
        fschema['regex'] = regex_validator.pattern.pattern

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

