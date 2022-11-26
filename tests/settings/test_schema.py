import json

from typing import Any, Optional, Union
from collections.abc import Mapping

import pytest

import attr
import numpy as np
import pendulum

from attr import attrib
from loguru import logger

from qwip.settings.settings import Settings, qdefine
from qwip.flatdict import FlatDict
from qwip.settings.schema import schema


def test_properties():
    @qdefine
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
        'required': ['number_property', 'string_property'],
        'additionalProperties': False
    }

    assert schema(FlatSettings) == s

def test_optional():
    @qdefine
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
    @qdefine
    class UnionSettings(Settings):
        union_property: Union[str, int]
        unrealistic_property: Union[bool, dict[str, str]]

    s = schema(UnionSettings)
    logger.debug(json.dumps(s, indent=4))

    assert set(s['properties']['union_property']['type']) == set(['number', 'string'])
    assert set(s['properties']['unrealistic_property']['type']) == set(['boolean', 'object'])
    assert s['properties']['unrealistic_property']['additionalProperties']['type'] == 'string'

def test_mapping_properties():
    @qdefine
    class MappingSettings(Settings):
        """A settings class with mappings."""
        dict_property: dict[str, int]
        mapping_property: Mapping
        parameters: FlatDict[str, str]

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
    @qdefine
    class ChildA(Settings):
        """A child settings class"""
        string_property: str

    @qdefine
    class ParentSettings(Settings):
        """A settings class with subsettings"""

        @qdefine
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
    @qdefine
    class ListSettings(Settings):
        """A settings class with a mapping."""
        list_property: list
        list_of_str: list[str]

    s = schema(ListSettings)
    logger.debug(json.dumps(s, indent=4))
    assert (s['properties']['list_property']['type'] ==
            s['properties']['list_of_str']['type'] == 'array')
    assert s['properties']['list_of_str']['items']['type'] == 'string' 


### String specific properties

def test_format():
    @qdefine
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
    regex = r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)'
    @qdefine
    class RegexSetting(Settings):
        email: str = attrib(
            validator=[
                attr.validators.matches_re(regex),
            ]
        )
        

    RegexSetting(email='abc@def.com')

    s = schema(RegexSetting)
    logger.debug(json.dumps(s, indent=4))

def test_in_validator():
    @qdefine
    class InSettings(Settings):
        option_property: Any = attrib(
            validator=attr.validators.in_([0, 'str', False])
        )

    s = schema(InSettings)
    logger.debug(json.dumps(s, indent=4))


def test_property_names():
    @qdefine
    class ConstrainedSettings(Settings):
        names: FlatDict[str, int] = attrib(
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.matches_re('[a-zA-Z][0-9]')
            )
        )

    s = schema(ConstrainedSettings)

    logger.debug(json.dumps(s, indent=4))

def test_zurich():
    @qdefine
    class ZurichDACSettings(Settings):
        """Configuration for the Zurich HDAWGs."""
        @qdefine
        class HardwareSettings(Settings):
            ac_coupling: int
            range: dict

        @qdefine
        class ReadoutSettings(Settings):
            herald_delay: int
            readout_delay: float
            reset_delay: float
        
        disconnect: str = attrib(
            validator=attr.validators.in_(['yes', 'no'])
        )
        experiment_setup: str = attrib(
            validator=attr.validators.in_([
                'multiqubitbasic',
                'multiqubitdig',
                'multiqubitfeedback',
                'multiqubitnohandshake',
                'blizzardpqsc'
            ])
        )
        hardware_settings: HardwareSettings
        hdawgs: list[str] = attrib(
            validator=attr.validators.deep_iterable(attr.validators.matches_re('dev[0-9]{4}'))
        )
        readout: ReadoutSettings
        replay: str = attrib(
            validator=attr.validators.in_(['yes', 'no'])
        )
        software_modulation: str = attrib(
            validator=attr.validators.in_(['yes', 'no'])
        )

    s = schema(ZurichDACSettings)

    logger.debug(json.dumps(s, indent=4))