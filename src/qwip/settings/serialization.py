"""Serializers for settings."""

import qwip

def add_type_converters(cls, fields):
    new_fields = []

    for field in fields:
        def type_converter(cls):
            return lambda v: qwip.converter.structure(v, cls)

        if field.metadata.get('auto_convert', True) and field.init:
            field = field.evolve(converter=type_converter(field.type))

        new_fields.append(field)

    return new_fields