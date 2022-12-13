"""Serializers for settings."""
from attrs import frozen
from loguru import logger

import qwip
from qwip.typing import replace_self_type


@frozen
class _TypeConverter:
    fieldtype: type

    def __call__(self, v):
        try:
            return qwip.converter.structure(v, self.fieldtype)
        except Exception as e:
            logger.debug(f'Conversion failed due to {e}.')
            return v

def add_type_converters(cls, fields):
    new_fields = []

    for field in fields:
        if (
            field.metadata.get('auto_convert', True) and 
            field.init and 
            (
                field.converter is None or
                (
                    isinstance(field.converter, _TypeConverter) and 
                    cls != field.converter.fieldtype
                )
            )
        ):
            converter = _TypeConverter(replace_self_type(field.type, cls))
            field = field.evolve(converter=converter)

        new_fields.append(field)

    return new_fields