import json

from typing import Mapping, Optional, Dict, List, Union

import pytest

import attr
import numpy as np
import pendulum

from attr import attrib
from loguru import logger

from qwip.settings.settings import Settings, qattrs
from qwip.settings.parameters import Parameters
from qwip.settings.schema import schema


def test_properties():
    @qattrs
    class FlatSettings(Settings):
        """A simple settings class."""

        number_property: float = attrib(metadata=dict(description='A number.'))
        string_property: str
        bool_property: bool = False

    s = {
        'title': 'FlatSettings',
        'description': 'A simple settings class.',
        'type': 'object',
        'properties': {
            'number_property': {
                'title': 'number_property', 'type': 'number', 'description': 'A number.'
            },
            'string_property': {
                'title': 'string_property',
                'type': 'string'
            },
            'bool_property': {
                'title': 'bool_property', 'type': 'boolean', 'default': False
            },
        },
        'required': ['number_property', 'string_property']
    }

    assert schema(FlatSettings) == s

def test_optional():
    @qattrs
    class OptionalSettings(Settings):
        required_int: int
        optional_str: Optional[str]
        default_bool: bool = False

    s = schema(OptionalSettings)

    logger.debug(json.dumps(s, indent=4))

    assert 'required_int' in s['required']
    assert 'optional_str' not in s['required']
    assert 'default_bool' not in s['required']

def test_union():
    @qattrs
    class UnionSettings(Settings):
        union_property: Union[str, int]
        unrealistic_property: Union[bool, Dict[str, str]]

    s = schema(UnionSettings)
    logger.debug(json.dumps(s, indent=4))

    assert set(s['properties']['union_property']['type']) == set(['number', 'string'])
    assert set(s['properties']['unrealistic_property']['type']) == set(['boolean', 'object'])
    assert s['properties']['unrealistic_property']['additionalProperties']['type'] == 'string'

def test_mapping_properties():
    @qattrs
    class MappingSettings(Settings):
        """A settings class with mappings."""
        dict_property: Dict[str, int]
        mapping_property: Mapping
        parameters: Parameters[str, str]

    s = schema(MappingSettings)
    properties = {
        'dict_property': {
            'title': 'dict_property',
            'type': 'object',
            'additionalProperties': {'type': 'number'}
        },
        'mapping_property': {'title': 'mapping_property', 'type': 'object'},
        'parameters': {
            'title': 'parameters',
            'type': 'object',
            'additionalProperties': {'type': 'string'}
        }
    }  
    logger.debug(json.dumps(s, indent=4))
    assert s['properties'] == properties

def test_settings_properties():
    @qattrs
    class ChildA(Settings):
        """A child settings class"""
        string_property: str

    @qattrs
    class ParentSettings(Settings):
        """A settings class with subsettings"""

        @qattrs
        class ChildB(Settings):
            property_A: ChildA
        
        property_A: ChildA
        property_B: ChildB

    s = schema(ParentSettings)
    logger.debug(json.dumps(s, indent=4))
    assert 'ChildA' in s['definitions']
    assert 'ChildB' in s['definitions']
    assert (s['definitions']['ChildA']['type'] ==
            s['definitions']['ChildB']['type'] ==
            'object')
    assert (s['definitions']['ChildA']['additionalProperties'] ==
            s['definitions']['ChildB']['additionalProperties'] == False)

def test_list_properties():
    @qattrs
    class ListSettings(Settings):
        """A settings class with a mapping."""
        list_property: List
        list_of_str: List[str]

    s = schema(ListSettings)
    logger.debug(json.dumps(s, indent=4))
    assert (s['properties']['list_property']['type'] ==
            s['properties']['list_of_str']['type'] == 'array')
    assert s['properties']['list_of_str']['items']['type'] == 'string' 

### String specific properties

def test_format():
    @qattrs
    class FormatSettings(Settings):
        datetime: Optional[pendulum.DateTime]
        date_or_time: Union[pendulum.Date, pendulum.Time]

    s = schema(FormatSettings)
    logger.debug(json.dumps(s, indent=4))

    assert (s['properties']['datetime']['type'] ==
            s['properties']['date_or_time']['type'] == 'string')
    assert s['properties']['datetime']['format'] == 'date-time'
    assert dict(format='date') in s['properties']['date_or_time']['anyOf']
    assert dict(format='time') in s['properties']['date_or_time']['anyOf']

def test_regex():
    regex = '(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)'
    @qattrs
    class RegexSetting(Settings):
        email: str = attrib(
            validator=[
                attr.validators.matches_re(regex),
            ]
        )
        

    RegexSetting(email='abc@def.com')

    s = schema(RegexSetting)
    logger.debug(json.dumps(s, indent=4))
