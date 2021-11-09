from typing import Type
import pytest

from qwip.parameters import Parameters

@pytest.fixture
def simple_dict():
    return {'str_val': 'string', 'int_val': 1, 'bool_val': True}

@pytest.fixture
def nested_dict():
    return {
        'list_of_ints': [i for i in range(5)],
        'mapping_of_literal': {'str_val': 'string', 'int_val': 1},
        'list_of_mapping': [
            {'str_val': 'string1', 'bool_val': True},
            {'str_val': 'string2', 'bool_val': False}
        ]
    }

@pytest.fixture
def simple_parameters(simple_dict):
    return Parameters(simple_dict)

@pytest.fixture
def nested_parameters(nested_dict):
    return Parameters(nested_dict)

@pytest.fixture
def flatkeys():
    return [f'list_of_ints/{i}' for i in range(5)] + [
        'mapping_of_literal/str_val',
        'mapping_of_literal/int_val',
        'list_of_mapping/0/str_val',
        'list_of_mapping/0/bool_val',
        'list_of_mapping/1/str_val',
        'list_of_mapping/1/bool_val',
    ]

## Initialization

def test_empty_initialization():
    p = Parameters()
    assert not p

def test_dict_initialization(simple_dict, nested_dict):
    p = Parameters(simple_dict)
    assert ('str_val' in p) and ('int_val' in p)
    
    p = Parameters(nested_dict)
    assert isinstance(p['list_of_mapping'][0], Parameters)

def test_keyword_initialization(simple_dict, nested_dict):
    p = Parameters(**simple_dict)
    assert ('str_val' in p) and ('int_val' in p)
    
    p = Parameters(**nested_dict)
    assert isinstance(p['list_of_mapping'][0], Parameters)

## Contains

def test_nested_contains(nested_parameters):
    p = nested_parameters

    assert 'mapping_of_literal/str_val' in p
    assert 'list_of_mapping/0/str_val' in p

## Getters and setters

def test_simple_key_get(simple_parameters):
    p = simple_parameters
    
    assert p['str_val'] == 'string'
    assert p['int_val'] == 1
    assert p['bool_val'] == True

def test_simple_key_get_nonexistent(simple_parameters):
    p = simple_parameters
    
    with pytest.raises(KeyError):
        p['nonexistent_key']

def test_simple_key_set(simple_parameters):
    p = simple_parameters

    p['str_val'] = 'newstring'
    assert p['str_val'] == 'newstring'

    p['int_val'] = 2
    assert p['int_val'] == 2

def test_simple_attr_get(simple_parameters):
    p = simple_parameters

    assert p.str_val == 'string'
    assert p.int_val == 1
    assert p.bool_val == True

def test_simple_attr_get_nonexistent(simple_parameters):
    p = simple_parameters

    with pytest.raises(AttributeError):
        p.nonexistent_key

def test_simple_attr_set(simple_parameters):
    p = simple_parameters

    p.str_val = 'newstring'
    assert p['str_val'] == 'newstring'

    p.int_val = 2
    assert p['int_val'] == 2

def test_nested_key_get(nested_parameters):
    p = nested_parameters

    assert p['list_of_ints/0'] == 0
    assert p['mapping_of_literal/str_val'] == 'string'
    assert p['list_of_mapping/0/str_val'] == 'string1'

def test_nested_key_get_nonexistent(nested_parameters):
    p = nested_parameters

    with pytest.raises(KeyError):
        p['mapping_of_literal/nonexistent_key']

    with pytest.raises(KeyError):
        p['nonexistent_mapping/nonexistent_key']

def test_nested_key_set(nested_parameters):
    p = nested_parameters

    p['list_of_ints/0'] = 6
    assert p['list_of_ints/0'] == 6

    p['mapping_of_literal/int_val'] = 10
    assert p['mapping_of_literal/int_val'] == 10

    p['list_of_mapping/0/bool_val'] = False
    assert p['list_of_mapping/0/bool_val'] == False

def test_nested_attr_get(nested_parameters):
    p = nested_parameters

    assert p.list_of_ints[0] == 0
    assert p.mapping_of_literal.str_val == 'string'
    assert p.list_of_mapping[0].str_val == 'string1'

def test_nested_attr_get_nonexistent(nested_parameters):
    p = nested_parameters

    with pytest.raises(AttributeError):
        p.mapping_of_literal.nonexistent_key

    with pytest.raises(AttributeError):
        p.nonexistent_mapping.nonexistent_key

def test_nested_attr_set(nested_parameters):
    p = nested_parameters

    p.list_of_ints[0] = 6
    assert p['list_of_ints/0'] == 6

    p.mapping_of_literal.int_val = 10
    assert p['mapping_of_literal/int_val'] == 10

    p.list_of_mapping[0].bool_val = False
    assert p['list_of_mapping/0/bool_val'] == False

## Update

def test_update_with_flat_dict(nested_parameters):
    p = nested_parameters
    
    p.update({'list_of_ints/0': 10, 'list_of_mapping/0/str_val': 'new_string'})

    assert p['list_of_ints/0'] == 10
    assert p['list_of_mapping/0/str_val'] == 'new_string'

def test_update_with_nested_dict(nested_parameters):
    p = nested_parameters

    p.update({
        'mapping_of_literal': {'str_val': 'new_string', 'int_val': 0}
    })

    assert p['mapping_of_literal/str_val'] == 'new_string'
    assert p['mapping_of_literal/int_val'] == 0

def test_update_with_keyword(nested_parameters):
    p = nested_parameters
    p.update(**{'mapping_of_literal/str_val': 'new_string'})

    assert p['mapping_of_literal/str_val'] == 'new_string'

## Create

def test_simple_key_create(simple_parameters):
    p = simple_parameters

    p['new_key'] = 'new_string'
    assert p['new_key'] == 'new_string'

def test_simple_attr_create(simple_parameters):
    p = simple_parameters

    p.new_key = 'new_string'
    assert p['new_key'] == 'new_string'

def test_nested_key_create(nested_parameters):
    p = nested_parameters
    
    p['mapping_of_literal/new_str_val'] = 'new_string'
    assert p['mapping_of_literal/new_str_val'] == 'new_string'

    p['list_of_mapping/1/new_str_val'] = 'new_nested_string'
    assert p['list_of_mapping/1/new_str_val'] == 'new_nested_string'

def test_nested_attr_create(nested_parameters):
    p = nested_parameters

    p.mapping_of_literal.new_str_val = 'new_string'
    assert p['mapping_of_literal/new_str_val'] == 'new_string'

    p.list_of_mapping[1].new_str_val = 'new_nested_string'
    assert p['list_of_mapping/1/new_str_val'] == 'new_nested_string'

def test_key_create_hierarchy(simple_parameters):
    p = simple_parameters

    p['new_mapping_of_mapping/child_mapping/string_val'] = 'string'
    assert p.new_mapping_of_mapping.child_mapping.string_val == 'string'
    assert isinstance(p['new_mapping_of_mapping'], Parameters)

def test_key_create_incompatible(simple_parameters):
    p = simple_parameters
    
    with pytest.raises(TypeError):
        p['str_val/impossible_create'] = 1

def test_attr_create_incompatible(simple_parameters):
    p = simple_parameters

    with pytest.raises(AttributeError):
        p.str_val.impossible_create = 1

## Delete

def test_simple_delete(simple_parameters):
    p = simple_parameters

    del p['str_val']

    assert 'str_val' not in p

def test_nested_delete(nested_parameters):
    p = nested_parameters

    del p['mapping_of_literal/str_val']
    del p['mapping_of_literal/int_val']

    assert 'mapping_of_literal' in p
    assert 'mapping_of_literal/str_val' not in p
    assert 'int_val' not in p['mapping_of_literal']

## Flat Iteration

def test_flat_iteration(nested_parameters, flatkeys):
    p = nested_parameters

    for key, expected_key in zip(p.flatkeys(), flatkeys):
        assert key == expected_key

    # def nested_dict():

## Test Deep Copy

def test_deep_copy(nested_parameters):
    p1 = nested_parameters
    p2 = p1.copy()

    def recursive_check(obj1, obj2):
        it1 = obj1.items() if isinstance(obj1, dict) else enumerate(obj1)

        for k1, v1 in it1:
            v2 = obj2[k1]

            if isinstance(v2, (dict, list)):
                assert id(v1) != id(v2)
                recursive_check(v1, v2)
            else:
                assert v1 == v2

    recursive_check(p1, p2)

## Test Context Manager

def test_simple_context(simple_parameters):
    p = simple_parameters

    with p.context():
        p['str_val'] = 'temporary_string'

        assert p['str_val'] == 'temporary_string'

    assert p['str_val'] == 'string'

def test_nested_context(nested_parameters):
    p = nested_parameters

    with p.context():
        p['mapping_of_literal/str_val'] = 'temporary_string'
        p['list_of_mapping/0/bool_val'] = False

        assert p['mapping_of_literal/str_val'] == 'temporary_string'
        assert p['list_of_mapping/0/bool_val'] == False

    assert p['mapping_of_literal/str_val'] == 'string'
    assert p['list_of_mapping/0/bool_val'] == True
