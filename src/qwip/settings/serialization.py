"""Serializers for settings."""

from loguru import logger

import qwip

def get_type_converter(cls):
    def convert(v):
        """Attempts type structuring.
        
        We want most errors to be caught by validation since this provides a
        clearer error message"""
        try:
            return qwip.converter.structure(v, cls)
        except Exception as e:
            logger.debug(f'Conversion failed due to {e}.')
            return v

    return convert

def add_type_converters(cls, fields):
    new_fields = []

    for field in fields:
        if field.metadata.get('auto_convert', True) and field.init:
            field = field.evolve(converter=get_type_converter(field.type))

        new_fields.append(field)

    return new_fields