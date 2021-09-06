import pytest

from qwip.settings.parameters import Parameters

@pytest.fixture
def simple_dict():
    return {'num_qubits': 1, 'awg': 'Zurich'}

@pytest.fixture
def nested_dict():
    nested = {
        'num_qubits': 1, 
        'qubits': [
            {'name': 'Q1', 'freq': 5.0},
            {'name': 'Q2', 'freq': 5.2, 'pulses': ['X', 'Y', 'Z']}
        ],
        'fridge': {'name': 'snowball', 'location': 'Campbell'}
    }
    return nested

@pytest.fixture
def flattened_keys():
    return [
        'num_qubits',
        'qubits/0/name',
        'qubits/0/freq',
        'qubits/1/name',
        'qubits/1/freq',
        'qubits/1/pulses/0',
        'qubits/1/pulses/1',
        'qubits/1/pulses/2',
        'fridge/name',
        'fridge/location'
    ]

@pytest.fixture
def nested_parameters(nested_dict):
    return Parameters(nested_dict)

def test_initialization(simple_dict, nested_dict):
    p = Parameters()
    assert not bool(p)

    p = Parameters(simple_dict)
    assert p['num_qubits'] == 1
    assert p['awg'] == 'Zurich'
    assert p.num_qubits == 1

    p = Parameters(**simple_dict)
    assert p['num_qubits'] == 1
    assert p['awg'] == 'Zurich'
    assert p.awg == 'Zurich'

    p = Parameters(nested_dict)
    assert isinstance(p['qubits'][0], Parameters)
    assert isinstance(p['fridge'], Parameters)

def test_get_set(nested_parameters):
    p = nested_parameters

    # Getters
    assert p['qubits/0/name'] == 'Q1'
    assert p['fridge/name'] == 'snowball'
    assert p['/fridge/location/'] == 'Campbell'

    p['qubits/0/name'] = 'Q0'
    assert p['qubits'][0].name == 'Q0'

    p['qubits/1/pulses/0'] = 'Y'

    for actual, expected in zip(p['qubits/1/pulses'], ['Y', 'Y', 'Z']):
        assert actual == expected
    

def test_delete(nested_parameters):
    p = nested_parameters

    del p['fridge']

    with pytest.raises(AttributeError):
        p['fridge']

def test_context(nested_parameters):
    p = nested_parameters

    with p.context():
        p['fridge/name'] = 'blizzard'
        assert p['fridge/name'] == 'blizzard'

    assert p['fridge/name'] == 'snowball'
    
    with p.context(settings={'fridge': {'temperature': 10}}):
        assert p['fridge/temperature'] == 10

        with pytest.raises(AttributeError):
            p['fridge/name']

    assert p['fridge/name'] == 'snowball'
    with pytest.raises(AttributeError):
        p['fridge/temperature']

def test_iter(nested_dict, nested_parameters, flattened_keys):
    d = nested_dict
    p = nested_parameters

    for dkeys, pkeys in zip(d, p):
        assert dkeys == pkeys

    for dkeys, pkeys in zip(d.keys(), p.keys()):
        assert dkeys == pkeys

    for pkeys, fkeys in zip(p.flatkeys(), flattened_keys):
        assert pkeys == fkeys

def test_copy(nested_parameters):
    p1 = nested_parameters
    p2 = p1.copy()

    def recursive_check(obj1, obj2):
        it1 = obj1.items() if isinstance(obj1, dict) else enumerate(obj1)
        
        for k1, v1 in it1:
            v2 = obj2[k1]

            if isinstance(v2, (dict, list)):
                assert v1 is not v2
                recursive_check(v1, v2)
            else:
                assert v1 == v2

    recursive_check(p1, p2)