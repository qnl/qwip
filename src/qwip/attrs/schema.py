"""Tools for creating JSON schema from attrs classes.

See https://json-schema.org/ for JSON schema specification.
"""

import datetime
import types
import typing
from collections.abc import Collection, Mapping
from numbers import Number
from typing import get_args, get_origin

import attrs
import numpy as np
from attrs import Attribute
from attrs.exceptions import NotAnAttrsClassError
from loguru import logger

from qwip.flatdict import FlatDict
from qwip.settings.base import SettingsBase
from qwip.typing import is_optional_type, is_union_type, typedispatch


def schema(cls):
    definitions = dict()

    properties, required = collect_attrs_properties(cls, definitions)

    cls_schema = {
        "title": cls.__name__,
        "description": cls.__doc__,
        "type": "object",
        "properties": properties,
    }

    if not cls_schema["description"]:
        del cls_schema["description"]

    if required:
        cls_schema["required"] = required

    if definitions:
        cls_schema["$defs"] = definitions

    cls_schema["additionalProperties"] = False

    return cls_schema


def collect_attrs_properties(cls: type, definitions: dict) -> tuple[dict, list]:
    """Collects properties from class.

    This function will recursively collect properties from fields that are also
    attrs decorated classes.
    """
    if not attrs.has(cls):
        raise NotAnAttrsClassError(
            f"Cannot generate schema for non attrs class {repr(cls)}."
        )

    properties = {}
    required = []

    for field in attrs.fields(cls):
        if not field.metadata.get("serialize", True):
            continue

        properties[field.name] = get_field_schema(field, definitions)

        if field.default == attrs.NOTHING:
            required += [field.name]

    return properties, required


def get_field_schema(field: Attribute, definitions: dict) -> list[dict]:
    schema = {}

    if description := get_description(field.type, field):
        schema["description"] = description

    if json_type := get_json_type(field.type):
        schema["type"] = json_type

    if field.default != attrs.NOTHING:
        schema["default"] = field.default

    tp = add_keywords(field.type, field, schema)

    if "$ref" in schema and field.name not in definitions:
        properties, required = collect_attrs_properties(tp, definitions)

        definitions[field.name] = {"properties": properties}

        if required:
            definitions[field.name]["required"] = required

        definitions[field.name]["additionalProperties"] = False

    return schema


## ==== Property Description ===== ##


@typedispatch
def get_description(tp: type, field: Attribute) -> str:
    return field.metadata.get("description", None)


@get_description.register(typing.Annotated)
def _(tp: type, field: Attribute) -> str:
    args = get_args(tp)

    if isinstance(description := args[1], str):
        return description
    else:
        return get_description(args[0], field)


@get_description.register(typing.Union)
@get_description.register(types.UnionType)
def _(tp: type, field: Attribute) -> str:
    if type(None) in (args := get_args(tp)) and len(args) == 2:
        tp = args[0] if args[1] is type(None) else args[1]
        description = get_description(tp)
    else:
        description = field.metadata.get("description", None)

    return description


## ==== Property Type ==== ##


@typedispatch
def get_json_type(tp: type) -> str:
    # Default serialization method is to string unless it is an attrs class
    return "object" if attrs.has(tp) else "string"


@get_json_type.register(typing.Annotated)
def _(tp: type):
    return get_json_type(get_args(tp)[0])


@get_json_type.register(typing.Union)
@get_json_type.register(types.UnionType)
def _(tp: type) -> str:
    args = get_args(tp)
    json_types = set(get_json_type(t) for t in args if t is not type(None))

    if len(json_types) == 1:
        (t,) = json_types
        return t

    # Can't convert union to a single json type
    return None


@get_json_type.register(type(None))
def _(tp: type) -> str:
    return "null"


@get_json_type.register(bool)
def _(tp: type) -> str:
    return "boolean"


@get_json_type.register(Number)
def _(tp: type) -> str:
    return "number"


@get_json_type.register(str)
def _(tp: type) -> str:
    return "string"


@get_json_type.register(np.ndarray)
@get_json_type.register(Collection)
def _(tp: type) -> str:
    return "array"


@get_json_type.register(Mapping)
def _(tp: type) -> str:
    return "object"


@typedispatch
def add_keywords(tp: type, field: Attribute, schema: dict) -> type:
    if attrs.has(tp):
        return add_attrs_keywords(tp, field, schema)
    else:
        ...


def add_attrs_keywords(tp: type, field: Attribute, schema: dict) -> type:
    schema["$ref"] = f"#/$defs/{field.name}"

    return tp


@add_keywords.register(typing.Annotated)
def _(tp: type, field: Attribute, schema: dict) -> type:
    return add_keywords(get_args(tp)[0], field, schema)


# def field_schema(field, definitions=None):
#     fschema = {
#         'title': field.name,
#     }

#     add_description(fschema, field)
#     process_types(fschema, field, definitions=definitions)
#     add_enum(fschema, field)

#     if field.default != attrs.NOTHING:
#         fschema['default'] = field.default

#     return fschema

# def add_description(fschema, field):
#     fieldtype = None
#     if get_origin(field.type):
#         if is_optional_type(field.type):
#             fieldtype = get_args(field.type)[0]
#     elif field.type != typing.Any:
#         fieldtype = field.type

#     default = fieldtype.__doc__ if fieldtype and issubclass(fieldtype, SettingsBase) else None
#     description = field.metadata.get('description', default)
#     if description:
#         fschema['description'] = description

# def get_enum_validator(field):
#     validators = []
#     try:
#         validators = field.validator._validators
#     except AttributeError:
#         validators = [field.validator]

#     for v in validators:
#         if isinstance(v, attrs.validators._InValidator):
#             return v

#     return None

# def add_enum(fschema, field):
#     enum_validator = get_enum_validator(field)

#     if enum_validator:
#         fschema['enum'] = enum_validator.options

# def process_types(fschema, field, definitions=None):
#     if field.type == typing.Any:
#         return

#     class_type_map = get_origin(field.type) or field.type
#     json_types = set(get_json_type(cls) for cls in class_type_map)

#     fschema['type'] = list(json_types) if len(json_types) > 1 else json_types.pop()

#     for cls, tp in class_type_map.items():
#         add_type_specific_properties(cls, fschema, field, tp, definitions=definitions)

# @typedispatch
# def get_json_type(field_type):
#     return 'object'

# @get_json_type.register(type(None))
# def _(field_type):
#     return 'null'

# @get_json_type.register(bool)
# def _(field_type):
#     return 'boolean'

# @get_json_type.register(Number)
# def _(field_type):
#     return 'number'

# @get_json_type.register(str)
# @get_json_type.register(datetime.datetime)
# @get_json_type.register(datetime.date)
# @get_json_type.register(datetime.time)
# @get_json_type.register(datetime.timedelta)
# def _(field_type):
#     return 'string'

# @get_json_type.register(np.ndarray)
# @get_json_type.register(list)
# def _(field_type):
#     return 'array'

# @typedispatch
# def add_type_specific_properties(clstype, fschema, field, tp, **kwargs):
#     logger.warning(f'No properties to add for clstype {clstype}')


# def get_json_validation(validator):
#     if isinstance(validator, attrs.validators._MatchesReValidator):
#         return {'pattern': validator.pattern.pattern}
#     elif isinstance(validator, attrs.validators._InValidator):
#         return {'enum': validator.options}
#     # elif isinstance(validator, attrs.validators._NumberValidator):
#     #     op_text = {
#     #         '<': 'exclusiveMaximum',
#     #         '>': 'exclusiveMinimum',
#     #         '<=': 'maximum',
#     #         '>=': 'minimum'
#     #     }
#     #     return {op_text[validator.compare_op]: validator.bound}

#     return {}

# def get_iterable_validators(field):
#     try:
#         validators = field.validator._validators
#     except AttributeError:
#         validators = [field.validator]

#     return [v for v in validators if isinstance(v, attrs.validators._DeepIterable)]

# def get_mapping_validators(field):
#     try:
#         validators = field.validator._validators
#     except AttributeError:
#         validators = [field.validator]

#     return [v for v in validators if isinstance(v, attrs.validators._DeepMapping)]

# @add_type_specific_properties.register(Mapping)
# def _(clstype, fschema, field, tp, definitions=None):
#     """
#     """
#     ## Add value types
#     try:
#         kt, vt = get_args(tp)
#     except ValueError:
#         kt = None
#         vt = None

#     json_types = set()
#     if vt is not None and vt != typing.Any:
#         logger.debug(f'Adding types for {vt}')
#         class_type_map = get_origin(vt) or vt

#         for cls in class_type_map:
#             json_types.add(get_json_type(cls))

#     if json_types:
#         json_types = list(json_types) if len(json_types) > 1 else json_types.pop()

#     iter_validators = get_iterable_validators(field)
#     pattern = None
#     enum = None
#     for v in iter_validators:
#         mv = v.member_validator
#         if isinstance(mv, attrs.validators._InValidator):
#             enum = mv.options
#         elif isinstance(mv, attrs.validators._MatchesReValidator):
#             pattern = mv.pattern.pattern

#     if enum and json_types:
#         fschema['properties'] = {name: {'type': json_types} for name in enum}
#     elif pattern and json_types:
#         fschema['patternProperties'] = {pattern: {'type': json_types}}
#     elif pattern:
#         fschema['propertyNames'] = dict(pattern=pattern)

#     pattern = None
#     value_properties = {}
#     map_validators = get_mapping_validators(field)
#     for v in map_validators:
#         kv = v.key_validator
#         vv = v.value_validator
#         if isinstance(kv, attrs.validators._MatchesReValidator):
#             pattern = kv.pattern.pattern
#             value_properties = get_json_validation(vv)

#     if json_types:
#         value_properties.update(dict(type=json_types))

#     if pattern and value_properties:
#         pattern_properties = fschema.get('patternProperties', {})
#         pattern_properties.update({pattern: value_properties})
#         fschema['patternProperties'] = pattern_properties
#     elif pattern:
#         fschema['propertyNames'] = dict(pattern=pattern)
#     elif value_properties and 'patternProperties' not in fschema:
#         fschema['additionalProperties'] = value_properties

# @add_type_specific_properties.register(SettingsBase)
# def _(clstype, fschema, field, tp, definitions=None):
#     properties, required, definitions = collect_properties(clstype, definitions)
#     if properties:
#         fschema['properties'] = properties

#     if required:
#         fschema['required'] = required

#     fschema['additionalProperties'] = False

# @add_type_specific_properties.register(list)
# def _(clstype, fschema, field, tp, definitions=None):
#     try:
#         vt = get_args(tp)[0]
#     except IndexError:
#         vt = None

#     json_types = set()
#     if vt is not None:
#         class_type_map = get_origin(vt) or vt

#         for cls in class_type_map:
#             json_types.add(get_json_type(cls))

#     item_properties = {}
#     if json_types:
#         json_types = list(json_types) if len(json_types) > 1 else json_types.pop()
#         item_properties.update(dict(type=json_types))

#     iter_validators = get_iterable_validators(field)
#     for v in iter_validators:
#         mv = v.member_validator
#         item_properties.update(get_json_validation(mv))

#     if item_properties:
#         fschema['items'] = item_properties

# @add_type_specific_properties.register(Number)
# def _(clstype, fschema, field, tp, definitions=None):
#     pass

# @add_type_specific_properties.register(bool)
# def _(clstype, fschema, field, tp, definitions=None):
#     pass

# def add_string_format(fschema, format_specifier):
#     if 'format' in fschema:
#         fschema['anyOf'] = [dict(format=fschema['format'])]
#         del fschema['format']

#     if 'anyOf' in fschema:
#         fschema['anyOf'] += [dict(format=format_specifier)]
#     else:
#         fschema['format'] = format_specifier

# def get_regex_validator(field):
#     try:
#         validators = field.validator._validators
#     except AttributeError:
#         validators = [field.validator]

#     for v in validators:
#         if isinstance(v, attrs.validators._MatchesReValidator):
#             return v

#     return None

# @add_type_specific_properties.register(str)
# def _(clstype, fschema, field, tp, definitions=None):
#     regex_validator = get_regex_validator(field)
#     if regex_validator:
#         fschema['regex'] = regex_validator.pattern.pattern

# @add_type_specific_properties.register(datetime.datetime)
# def _(clstype, fschema, field, tp, definitions=None):
#     add_string_format(fschema, 'date-time')

# @add_type_specific_properties.register(datetime.date)
# def _(clstype, fschema, field, tp, definitions=None):
#     add_string_format(fschema, 'date')

# @add_type_specific_properties.register(datetime.time)
# def _(clstype, fschema, field, tp, definitions=None):
#     add_string_format(fschema, 'time')

# @add_type_specific_properties.register(datetime.timedelta)
# def _(clstype, fschema, field, tp, definitions=None):
#     add_string_format(fschema, 'duration')
