import pytest

from attr import attrib

import qwip

from qwip import qsettings
from qwip._qsettings import DefaultSettings
from qwip.settings.settings import qattrs

@qattrs
class ExampleSettings(DefaultSettings):
    int_field: int = 1
    str_field: str = 'string'
    bool_field: bool = True

@qattrs
class NestedSettings(DefaultSettings):
    float_field: float = 0.1
    child: ExampleSettings = attrib(factory=ExampleSettings)

@pytest.fixture
def example():
    return ExampleSettings()

@pytest.fixture
def nested_example():
    return NestedSettings()

def test_set_defaults(example):
    example['int_field'] = 0
    example['bool_field'] = False

    example.set_defaults()

    assert tuple(example._defaults.values()) == (0, 'string', False)

def test_nested_defaults(nested_example):
    nested_example['child/int_field'] = 2

    nested_example.set_defaults()
    
    assert nested_example._defaults['float_field'] == 0.1
    assert nested_example._defaults['child/int_field'] == 2
    assert nested_example._defaults['child/str_field'] == 'string'
    assert nested_example._defaults['child/bool_field'] == True

def test_reset_no_defaults(example):
    with pytest.raises(ValueError):
        example.reset()

def test_reset_key(example):
    example.set_defaults()

    example['int_field'] = 0
    example['str_field'] = 'new_string'
    example.reset('int_field')

    assert example['int_field'] == 1
    assert example['str_field'] == 'new_string'

def test_reset_all(example):
    example.set_defaults()

    example['int_field'] = 0
    example['str_field'] = 'new_string'
    example['bool_field'] = False
    
    example.reset()
    
    assert example['int_field'] == 1
    assert example['str_field'] == 'string'
    assert example['bool_field'] == True

def test_nested_reset_all(nested_example):
    nested_example.set_defaults()

    nested_example['child/int_field'] = 2
    nested_example.reset()

    assert nested_example['child/int_field'] == 1

def test_nested_reset_key(nested_example):
    nested_example.set_defaults()

    nested_example['child/int_field'] = 2
    nested_example['child/str_field'] = 'new_string'

    nested_example.reset('child/int_field')
    
    assert nested_example['child/int_field'] == 1
    assert nested_example['child/str_field'] == 'new_string'

def test_nested_reset_child(nested_example):
    nested_example.set_defaults()

    nested_example['child/int_field'] = 2
    nested_example['child/bool_field'] = False
    
    nested_example.reset('child')

    assert nested_example['child/int_field'] == 1
    assert nested_example['child/bool_field'] == True

def test_qsettings():
    assert qsettings['version'] == qwip.__version__