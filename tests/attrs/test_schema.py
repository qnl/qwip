import json
from collections.abc import Mapping
from typing import Annotated

import attrs
import numpy as np
import pendulum
import pytest
from attr import field
from loguru import logger

from qwip.attrs import qdefine
from qwip.attrs.schema import get_description, get_field_schema, get_json_type, schema
from qwip.flatdict import FlatDict


class TestGetDescription:
    @pytest.mark.parametrize(
        "name,expected",
        [("annotated", "annotation"), ("metadata", "metadata"), ("none", None)],
    )
    def test_simple(self, name, expected):
        @qdefine
        class A:
            annotated: Annotated[int, "annotation"]
            metadata: str = field(metadata=dict(description="metadata"))
            none: bool

        f = getattr(attrs.fields(A), name)
        assert get_description(f.type, f) == expected

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("annotated", "annotation"),
            ("metadata", "metadata"),
        ],
    )
    def test_annotated(self, name, expected):
        @qdefine
        class A:
            annotated: Annotated[int, "annotation"] = field(
                metadata=dict(description="override description")
            )
            metadata: Annotated[str, 1] = field(metadata=dict(description="metadata"))

        f = getattr(attrs.fields(A), name)
        assert get_description(f.type, f) == expected


class TestGetJSONType:
    @qdefine
    class A:
        property: str

    @pytest.mark.parametrize(
        "tp,expected",
        [
            (type(None), "null"),
            (int, "number"),
            (float, "number"),
            (complex, "number"),
            (np.int64, "number"),
            (bool, "boolean"),
            (str, "string"),
            (object, "string"),
            (A, "object"),
            (list, "array"),
            (set, "array"),
            (dict, "object"),
        ],
    )
    def test_basic_types(self, tp, expected):
        assert get_json_type(tp) == expected

    @pytest.mark.parametrize(
        "tp,expected",
        [
            (dict[str, int], "object"),
            (Annotated[float, ""], "number"),
            (list | None, "array"),
            (complex | int | float, "number"),
            (dict | Mapping | None, "object"),
            (bool | int, None),
        ],
    )
    def test_generic_types(self, tp, expected):
        assert get_json_type(tp) == expected

    from pendulum import Date, DateTime, Duration, Period, Time

    @pytest.mark.parametrize("tp", [(Date, Time, DateTime, Duration, Period)])
    def test_pendulum_types(self, tp):
        assert get_json_type(tp) == "string"


class TestGetFieldSchema:
    def test_schema(self):
        import json

        @qdefine
        class B:
            c: str
            d: str

        @qdefine
        class A:
            a: int
            b: B

        s = schema(A)

        print()
        print(json.dumps(s, indent=4))


# def test_properties():
#     @qdefine
#     class FlatSettings(Settings):
#         """A simple settings class."""

#         number_property: float = field(metadata=dict(description='A number.'))
#         string_property: str
#         bool_property: bool = False

#     s = {
#         'title': 'FlatSettings',
#         'description': 'A simple settings class.',
#         'type': 'object',
#         'properties': {
#             'number_property': {
#                 'title': 'number_property', 'type': 'number', 'description': 'A number.'
#             },
#             'string_property': {
#                 'title': 'string_property',
#                 'type': 'string'
#             },
#             'bool_property': {
#                 'title': 'bool_property', 'type': 'boolean', 'default': False
#             },
#         },
#         'required': ['number_property', 'string_property'],
#         'additionalProperties': False
#     }

#     assert schema(FlatSettings) == s

# def test_optional():
#     @qdefine
#     class OptionalSettings(Settings):
#         required_int: int
#         optional_str: Optional[str]
#         default_bool: bool = False

#     s = schema(OptionalSettings)

#     logger.debug(json.dumps(s, indent=4))

#     assert 'required_int' in s['required']
#     assert 'optional_str' not in s['required']
#     assert 'default_bool' not in s['required']

# def test_union():
#     @qdefine
#     class UnionSettings(Settings):
#         union_property: Union[str, int]
#         unrealistic_property: Union[bool, dict[str, str]]

#     s = schema(UnionSettings)
#     logger.debug(json.dumps(s, indent=4))

#     assert set(s['properties']['union_property']['type']) == set(['number', 'string'])
#     assert set(s['properties']['unrealistic_property']['type']) == set(['boolean', 'object'])
#     assert s['properties']['unrealistic_property']['additionalProperties']['type'] == 'string'

# def test_mapping_properties():
#     @qdefine
#     class MappingSettings(Settings):
#         """A settings class with mappings."""
#         dict_property: dict[str, int]
#         mapping_property: Mapping
#         parameters: FlatDict[str, str]

#     s = schema(MappingSettings)
#     properties = {
#         'dict_property': {
#             'title': 'dict_property',
#             'type': 'object',
#             'additionalProperties': {'type': 'number'}
#         },
#         'mapping_property': {'title': 'mapping_property', 'type': 'object'},
#         'parameters': {
#             'title': 'parameters',
#             'type': 'object',
#             'additionalProperties': {'type': 'string'}
#         }
#     }
#     logger.debug(json.dumps(s, indent=4))
#     assert s['properties'] == properties

# def test_settings_properties():
#     @qdefine
#     class ChildA(Settings):
#         """A child settings class"""
#         string_property: str

#     @qdefine
#     class ParentSettings(Settings):
#         """A settings class with subsettings"""

#         @qdefine
#         class ChildB(Settings):
#             property_A: ChildA

#         property_A: ChildA
#         property_B: ChildB

#     s = schema(ParentSettings)
#     logger.debug(json.dumps(s, indent=4))
#     assert 'ChildA' in s['definitions']
#     assert 'ChildB' in s['definitions']
#     assert (s['definitions']['ChildA']['type'] ==
#             s['definitions']['ChildB']['type'] ==
#             'object')
#     assert (s['definitions']['ChildA']['additionalProperties'] ==
#             s['definitions']['ChildB']['additionalProperties'] == False)

# def test_list_properties():
#     @qdefine
#     class ListSettings(Settings):
#         """A settings class with a mapping."""
#         list_property: list
#         list_of_str: list[str]

#     s = schema(ListSettings)
#     logger.debug(json.dumps(s, indent=4))
#     assert (s['properties']['list_property']['type'] ==
#             s['properties']['list_of_str']['type'] == 'array')
#     assert s['properties']['list_of_str']['items']['type'] == 'string'


# ### String specific properties

# def test_format():
#     @qdefine
#     class FormatSettings(Settings):
#         datetime: Optional[pendulum.DateTime]
#         date_or_time: Union[pendulum.Date, pendulum.Time]

#     s = schema(FormatSettings)
#     logger.debug(json.dumps(s, indent=4))

#     assert (s['properties']['datetime']['type'] ==
#             s['properties']['date_or_time']['type'] == 'string')
#     assert s['properties']['datetime']['format'] == 'date-time'
#     assert dict(format='date') in s['properties']['date_or_time']['anyOf']
#     assert dict(format='time') in s['properties']['date_or_time']['anyOf']

# def test_regex():
#     regex = r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)'
#     @qdefine
#     class RegexSetting(Settings):
#         email: str = field(
#             validator=[
#                 attr.validators.matches_re(regex),
#             ]
#         )


#     RegexSetting(email='abc@def.com')

#     s = schema(RegexSetting)
#     logger.debug(json.dumps(s, indent=4))

# def test_in_validator():
#     @qdefine
#     class InSettings(Settings):
#         option_property: Any = field(
#             validator=attr.validators.in_([0, 'str', False])
#         )

#     s = schema(InSettings)
#     logger.debug(json.dumps(s, indent=4))


# def test_property_names():
#     @qdefine
#     class ConstrainedSettings(Settings):
#         names: FlatDict[str, int] = field(
#             validator=attr.validators.deep_iterable(
#                 member_validator=attr.validators.matches_re('[a-zA-Z][0-9]')
#             )
#         )

#     s = schema(ConstrainedSettings)

#     logger.debug(json.dumps(s, indent=4))

# def test_zurich():
#     @qdefine
#     class ZurichDACSettings(Settings):
#         """Configuration for the Zurich HDAWGs."""
#         @qdefine
#         class HardwareSettings(Settings):
#             ac_coupling: int
#             range: dict

#         @qdefine
#         class ReadoutSettings(Settings):
#             herald_delay: int
#             readout_delay: float
#             reset_delay: float

#         disconnect: str = field(
#             validator=attr.validators.in_(['yes', 'no'])
#         )
#         experiment_setup: str = field(
#             validator=attr.validators.in_([
#                 'multiqubitbasic',
#                 'multiqubitdig',
#                 'multiqubitfeedback',
#                 'multiqubitnohandshake',
#                 'blizzardpqsc'
#             ])
#         )
#         hardware_settings: HardwareSettings
#         hdawgs: list[str] = field(
#             validator=attr.validators.deep_iterable(attr.validators.matches_re('dev[0-9]{4}'))
#         )
#         readout: ReadoutSettings
#         replay: str = field(
#             validator=attr.validators.in_(['yes', 'no'])
#         )
#         software_modulation: str = field(
#             validator=attr.validators.in_(['yes', 'no'])
#         )

#     s = schema(ZurichDACSettings)

#     logger.debug(json.dumps(s, indent=4))
