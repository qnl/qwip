from typing import Type
import pytest

from qwip.flatdict import FlatDict

@pytest.fixture
def simple_dict():
    return {'str_val': 'string', 'int_val': 1, 'bool_val': True}

@pytest.fixture
def nested_dict():
    return {
        'list_of_ints': [i for i in range(5)],
        'mapping_of_literal': {'str_val': 'string', 'int_val': 1},
        'mapping_of_mapping': {
            'key1': {'str_val': 'string1', 'bool_val': True},
            'key2': {'str_val': 'string2', 'bool_val': False}
        }
    }

@pytest.fixture
def simple_flatdict(simple_dict):
    return FlatDict(simple_dict)

@pytest.fixture
def nested_flatdict(nested_dict):
    return FlatDict(nested_dict)

@pytest.fixture
def flatkeys():
    return [
        'list_of_ints',
        'mapping_of_literal/str_val',
        'mapping_of_literal/int_val',
        'mapping_of_mapping/key1/str_val',
        'mapping_of_mapping/key1/bool_val',
        'mapping_of_mapping/key2/str_val',
        'mapping_of_mapping/key2/bool_val',
    ]

## Initialization

def test_empty_initialization():
    p = FlatDict()
    assert not p

def test_dict_initialization(simple_dict, nested_dict):
    p = FlatDict(simple_dict)
    assert ('str_val' in p) and ('int_val' in p)
    
    p = FlatDict(nested_dict)
    assert isinstance(p['mapping_of_mapping']['key1'], FlatDict)

def test_keyword_initialization(simple_dict, nested_dict):
    p = FlatDict(**simple_dict)
    assert ('str_val' in p) and ('int_val' in p)
    
    p = FlatDict(**nested_dict)
    assert isinstance(p['mapping_of_mapping']['key1'], FlatDict)

## Contains

def test_nested_contains(nested_flatdict):
    p = nested_flatdict

    assert 'mapping_of_literal/str_val' in p
    assert 'mapping_of_mapping/key1/str_val' in p

## Length

def test_nested_lens(nested_flatdict):
    assert len(nested_flatdict) == 3
## Getters and setters

def test_simple_key_get(simple_flatdict):
    p = simple_flatdict
    
    assert p['str_val'] == 'string'
    assert p['int_val'] == 1
    assert p['bool_val'] == True

def test_simple_key_get_nonexistent(simple_flatdict):
    p = simple_flatdict
    
    with pytest.raises(KeyError):
        p['nonexistent_key']

def test_simple_key_set(simple_flatdict):
    p = simple_flatdict

    p['str_val'] = 'newstring'
    assert p['str_val'] == 'newstring'

    p['int_val'] = 2
    assert p['int_val'] == 2

def test_simple_attr_get(simple_flatdict):
    p = simple_flatdict

    assert p.str_val == 'string'
    assert p.int_val == 1
    assert p.bool_val == True

def test_simple_attr_get_nonexistent(simple_flatdict):
    p = simple_flatdict

    with pytest.raises(AttributeError):
        p.nonexistent_key

def test_simple_attr_set(simple_flatdict):
    p = simple_flatdict

    p.int_val = 2
    assert p['int_val'] == 2

    p.str_val = 'newstring'
    assert p['str_val'] == 'newstring'

def test_nested_key_get(nested_flatdict):
    p = nested_flatdict

    assert p['list_of_ints'] == list(range(5))
    assert p['mapping_of_literal/str_val'] == 'string'
    assert p['mapping_of_mapping/key1/str_val'] == 'string1'

def test_nested_key_get_nonexistent(nested_flatdict):
    p = nested_flatdict

    with pytest.raises(KeyError):
        p['mapping_of_literal/nonexistent_key']

    with pytest.raises(KeyError):
        p['nonexistent_mapping/nonexistent_key']

def test_nested_key_set(nested_flatdict):
    p = nested_flatdict

    p['list_of_ints'][0] = 6
    assert p['list_of_ints'][0] == 6

    p['mapping_of_literal/int_val'] = 10
    assert p['mapping_of_literal/int_val'] == 10

    p['mapping_of_mapping/key1/bool_val'] = False
    assert p['mapping_of_mapping/key1/bool_val'] == False

def test_nested_attr_get(nested_flatdict):
    p = nested_flatdict

    assert p.list_of_ints[0] == 0
    assert p.mapping_of_literal.str_val == 'string'
    assert p.mapping_of_mapping['key1'].str_val == 'string1'

def test_nested_attr_get_nonexistent(nested_flatdict):
    p = nested_flatdict

    with pytest.raises(AttributeError):
        p.mapping_of_literal.nonexistent_key

    with pytest.raises(AttributeError):
        p.nonexistent_mapping.nonexistent_key

def test_nested_attr_set(nested_flatdict):
    p = nested_flatdict

    p.list_of_ints[0] = 6
    assert p['list_of_ints'][0] == 6

    p.mapping_of_literal.int_val = 10
    assert p['mapping_of_literal/int_val'] == 10

    p.mapping_of_mapping['key1'].bool_val = False
    assert p['mapping_of_mapping/key1/bool_val'] == False

## Update

def test_update_with_flat_dict(nested_flatdict):
    p = nested_flatdict
    
    p.update({'list_of_ints': [10], 'mapping_of_mapping/key1/str_val': 'new_string'})

    assert p['list_of_ints'] == [10]
    assert p['mapping_of_mapping/key1/str_val'] == 'new_string'

def test_update_with_nested_dict(nested_flatdict):
    p = nested_flatdict

    p.update({
        'mapping_of_literal': {'str_val': 'new_string', 'int_val': 0}
    })

    assert p['mapping_of_literal/str_val'] == 'new_string'
    assert p['mapping_of_literal/int_val'] == 0

def test_update_with_keyword(nested_flatdict):
    p = nested_flatdict
    p.update(**{'mapping_of_literal/str_val': 'new_string'})

    assert p['mapping_of_literal/str_val'] == 'new_string'

## Create

def test_simple_key_create(simple_flatdict):
    p = simple_flatdict

    p['new_key'] = 'new_string'
    assert p['new_key'] == 'new_string'

def test_simple_attr_create(simple_flatdict):
    p = simple_flatdict

    p.new_key = 'new_string'
    assert p['new_key'] == 'new_string'

def test_nested_key_create(nested_flatdict):
    p = nested_flatdict
    
    p['mapping_of_literal/new_str_val'] = 'new_string'
    assert p['mapping_of_literal/new_str_val'] == 'new_string'

    p['mapping_of_mapping/key2/new_str_val'] = 'new_nested_string'
    assert p['mapping_of_mapping/key2/new_str_val'] == 'new_nested_string'

def test_nested_attr_create(nested_flatdict):
    p = nested_flatdict

    p.mapping_of_literal.new_str_val = 'new_string'
    assert p['mapping_of_literal/new_str_val'] == 'new_string'

    p.mapping_of_mapping['key2'].new_str_val = 'new_nested_string'
    assert p['mapping_of_mapping/key2/new_str_val'] == 'new_nested_string'

def test_key_create_hierarchy(simple_flatdict):
    p = simple_flatdict

    p['new_mapping_of_mapping/child_mapping/string_val'] = 'string'
    assert p.new_mapping_of_mapping.child_mapping.string_val == 'string'
    assert isinstance(p['new_mapping_of_mapping'], FlatDict)

def test_key_create_incompatible(simple_flatdict):
    p = simple_flatdict
    
    with pytest.raises(TypeError):
        p['str_val/impossible_create'] = 1

def test_attr_create_incompatible(simple_flatdict):
    p = simple_flatdict

    with pytest.raises(AttributeError):
        p.str_val.impossible_create = 1

## Delete

def test_simple_delete(simple_flatdict):
    p = simple_flatdict

    del p['str_val']

    assert 'str_val' not in p

def test_nested_delete(nested_flatdict):
    p = nested_flatdict

    del p['mapping_of_literal/str_val']
    del p['mapping_of_literal/int_val']

    assert 'mapping_of_literal' in p
    assert 'mapping_of_literal/str_val' not in p
    assert 'int_val' not in p['mapping_of_literal']

## Flat Iteration

def test_flat_iteration(nested_flatdict, flatkeys):
    p = nested_flatdict

    for key, expected_key in zip(p.flatkeys(), flatkeys):
        assert key == expected_key

@pytest.mark.parametrize(('d', 'expected'), [
    ({'a': {'b': {'k0': 0, 'k1': '1', 'k2': 2.0, 'k3': {'c': True}}}}, ('a/b/k0', 'a/b/k1', 'a/b/k2', 'a/b/k3/c')),
    ({'a': {'k0': 1, 'k1': {'b': True}, 'k2': {}}}, ('a/k0', 'a/k1/b', 'a/k2'))
])
def test_flat_keys(d, expected):
    flatkeys = tuple(FlatDict(d).flatkeys())
    
    assert flatkeys == expected


def test_flatiter_levels():
    x = FlatDict({
        'a1/b2/c3': {'int': 1, 'str': 'a'},
        'a1/b2/d3': {'int': 1, 'str': 'a'},
        'b1/c2': {'int': 1, 'str': 'a'},
        'random': 1
    })

    assert list(x.flatkeys(levels=1)) == ['a1', 'b1', 'random']
    assert list(x.flatkeys(levels=-1)) == [
        'a1/b2/c3',
        'a1/b2/d3',
        'b1/c2',
        'random'
    ]

## Test Deep Copy

def test_deep_copy(nested_flatdict):
    p1 = nested_flatdict
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

def test_simple_context(simple_flatdict):
    p = simple_flatdict

    with p.context():
        p['str_val'] = 'temporary_string'

        assert p['str_val'] == 'temporary_string'

    assert p['str_val'] == 'string'

def test_nested_context(nested_flatdict):
    p = nested_flatdict

    with p.context():
        p['mapping_of_literal/str_val'] = 'temporary_string'
        p['mapping_of_mapping/key1/bool_val'] = False

        assert p['mapping_of_literal/str_val'] == 'temporary_string'
        assert p['mapping_of_mapping/key1/bool_val'] == False

    assert p['mapping_of_literal/str_val'] == 'string'
    assert p['mapping_of_mapping/key1/bool_val'] == True
