from typing import Union, Any
from collections.abc import Mapping

import attr
from attr.validators import instance_of, deep_iterable, deep_mapping, and_

from qwip.settings.typing import get_origin, get_args

from loguru import logger

def add_type_validators(cls, fields):
    new_fields = []
    for field in fields:
        optional = False

        if field.type == Any:
            new_fields.append(field)
            continue

        type_origin = get_origin(field.type)
        type_args = get_args(field.type)

        if type_origin is Union:
            field_type = type_args
            optional = type(None) in field_type
            field_type = tuple(t for t in field_type if t is not type(None))
        elif type_origin is not None:
            field_type = type_origin
        else:
            field_type = field.type

        type_validator = instance_of(field_type)

        if optional:
            type_validator = attr.validators.optional(type_validator)

        if isinstance(field_type, tuple) and len(field_type) == 1:
            field_type = field_type[0]
        
        if not isinstance(field_type, tuple):
            if issubclass(field_type, Mapping) and len(type_args) > 0:
                ktype, vtype = get_args(field.type)
                type_validator = and_(
                    type_validator,
                    deep_mapping(instance_of(ktype), instance_of(vtype))
                )
            elif issubclass(field_type, list) and len(type_args) > 0:
                etype = type_args[0]
                type_validator = and_(
                    type_validator,
                    deep_iterable(instance_of(etype))
                )
            
        if field.validator is not None:
            type_validator = attr.validators.and_(field.validator, type_validator)
        
        field = field.evolve(validator=type_validator)
        new_fields.append(field)
    return new_fields
