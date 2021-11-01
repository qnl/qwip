import pytest

from enum import Enum
from typing import Dict, List, Optional
from attr import attrib

import numpy as np
from pendulum import Date, DateTime

from qwip.settings.settings import Settings, qattrs
from qwip.settings.parameters import Parameters

@qattrs
class SimpleSettings(Settings):
    int_field: int
    str_field: str
    bool_field: bool
    float_field: float
    datetime_field: DateTime
    numpy_field: np.ndarray = attrib(metadata={'dtype': np.int32})

@qattrs
class NestedSettings(SimpleSettings):
    @qattrs
    class ChildSetting(Settings):
        def positive(instance, attribute, value):
            if value <= 0:
                raise ValueError(f'"{attribute.name}" must be positive, got {value}.')

        positive_int_field: int = attrib(validator=positive)

    setting_field: ChildSetting
    param_field: Dict[str, str]
    list_field: List[int]
    optional_str: Optional[str]

@pytest.fixture
def simpledict():
    return {
        'int_field': 0,
        'str_field': 'abc',
        'bool_field': True,
        'float_field': 0.1,
        'datetime_field': '2006-01-02T15:04:05',
        'numpy_field': [i for i in range(10)],
    }

@pytest.fixture
def nesteddict():
    return {
        'int_field': 0,
        'str_field': 'abc',
        'bool_field': True,
        'float_field': 0.1,
        'datetime_field': '2006-01-02T15:04:05',
        'numpy_field': [i for i in range(10)],
        'setting_field': {
            'positive_int_field': -1
        },
        'param_field': Parameters({'key1': 'value1', 'key2': 2}),
        'list_field': [1, 2, 3],
        'optional_str': 1
    }

def test_structure(simpledict):
    s = SimpleSettings(**simpledict)

    for k, v in simpledict.items():
        assert hasattr(s, k)

    assert s['numpy_field'].dtype == np.int32

def test_validate(nesteddict):
    with pytest.raises(ValueError):
        s = NestedSettings(**nesteddict)

    nesteddict['setting_field']['positive_int_field'] = 1
    
    with pytest.raises(TypeError):
        s = NestedSettings(**nesteddict)

    nesteddict['param_field'].key2 = 'value2'

    with pytest.raises(TypeError):
        s = NestedSettings(**nesteddict)
    
    nesteddict['optional_str'] = None

    s = NestedSettings(**nesteddict)

    assert s.setting_field.positive_int_field > 0