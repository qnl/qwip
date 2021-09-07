import attr
from attr.validators import instance_of, deep_iterable, deep_mapping, and_

from collections.abc import Mapping
from qwip.settings.typing import get_origin, get_args

def add_type_validators(cls, fields):
    new_fields = []
    for field in fields:
        field_type = field.type
        if get_origin(field.type) is not None:
            field_type = get_origin(field.type)

        args = get_args(field.type)

        type_validator = instance_of(field_type)

        if issubclass(field_type, Mapping) and len(args) > 0:
            ktype, vtype = get_args(field.type)
            type_validator = and_(
                type_validator,
                deep_mapping(instance_of(ktype), instance_of(vtype))
            )
        elif issubclass(field_type, list):
            etype = get_args(field.type)[0]
            type_validator = and_(
                type_validator,
                deep_iterable(instance_of(etype))
            )
            
        if field.validator is not None:
            type_validator = attr.validators.and_(field.validator, type_validator)
        
        field = field.evolve(validator=type_validator)
        new_fields.append(field)
    return new_fields