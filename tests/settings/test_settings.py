from enum import Enum
from typing import Dict, List, Optional

import attr
import numpy as np
import pendulum
import pytest
from attr import field, s
from numpy.typing import NDArray

from qwip.flatdict import FlatDict
from qwip.settings.settings import Settings, qdefine


@qdefine
class SimpleSettings(Settings):
    int_field: int
    str_field: str
    bool_field: bool
    float_field: float


@qdefine
class NestedSettings(SimpleSettings):
    @qdefine
    class ChildSetting(Settings):
        def positive(instance, attribute, value):
            if value <= 0:
                raise ValueError(f'"{attribute.name}" must be positive, got {value}.')

        positive_int_field: int = field(validator=positive)

    setting_field: ChildSetting
    param_field: FlatDict[str, str]
    list_field: list[int]
    optional_str: str | None


@qdefine
class NumpySettings(Settings):
    # Numpy will set dtype = int64 for integer only arrays
    float_field: NDArray[np.float64]
    int_field: np.ndarray
    numpy_field: np.ndarray
    matrix_field: np.ndarray


@qdefine
class PendulumSettings(Settings):
    date_field: pendulum.Date
    datetime_field: pendulum.DateTime


@pytest.fixture
def simpledict():
    return {
        "int_field": 0,
        "str_field": "abc",
        "bool_field": True,
        "float_field": 0.1,
    }


@pytest.fixture
def numpydict():
    return {
        "float_field": [i for i in range(5)],
        "int_field": [i for i in range(10)],
        "numpy_field": np.zeros((3, 4)),
        "matrix_field": [[y + x for x in range(4)] for y in range(5)],
    }


@pytest.fixture
def pendulumdict():
    return {"date_field": "2006-01-02", "datetime_field": "2006-01-02T15:04:05"}


@pytest.fixture
def nesteddict():
    return {
        "int_field": 0,
        "str_field": "abc",
        "bool_field": True,
        "float_field": 0.1,
        "setting_field": {"positive_int_field": 1},
        "param_field": FlatDict({"key1": "value1", "key2": "value2"}),
        "list_field": [1, 2, 3],
        "optional_str": None,
    }


@pytest.fixture
def typefaildict(nesteddict):
    nesteddict["param_field"]["key2"] = list()
    nesteddict["optional_str"] = list()
    return nesteddict


## Context


def test_context(nesteddict):
    pass


## Structure


def test_structure(simpledict):
    s = SimpleSettings(**simpledict)

    for k, v in simpledict.items():
        assert hasattr(s, k)


def test_nested_structure(nesteddict):
    s = NestedSettings(**nesteddict)

    for k, v in nesteddict.items():
        assert hasattr(s, k)


def test_numpy_structure(numpydict):
    s = NumpySettings(**numpydict)
    assert s["float_field"].dtype == np.float64
    assert s["int_field"].dtype == np.int_

    assert id(s["numpy_field"]) == id(numpydict["numpy_field"])

    matrix = numpydict["matrix_field"]
    assert s["matrix_field"].shape == (len(matrix), len(matrix[0]))


def test_pendulum_structure(pendulumdict):
    s = PendulumSettings(**pendulumdict)
    assert isinstance(s["date_field"], pendulum.Date)
    assert isinstance(s["datetime_field"], pendulum.DateTime)


## Validation


def test_type_validate(typefaildict):
    with pytest.raises(TypeError):
        s = NestedSettings(**typefaildict)
    typefaildict["param_field"]["key2"] = "value2"

    with pytest.raises(TypeError):
        s = NestedSettings(**typefaildict)
    typefaildict["optional_str"] = None

    s = NestedSettings(**typefaildict)


def test_onset_validate(nesteddict):
    s = NestedSettings(**nesteddict)

    with pytest.raises(ValueError):
        s.setting_field.positive_int_field = -1


def test_disable_validate(typefaildict):
    from qwip.settings.settings import danger, disable_validation

    with disable_validation():
        s = NestedSettings(**typefaildict)

    assert s["param_field"]["key2"] == list()
    assert s["optional_str"] == list()

    with danger():
        s["optional_str"] = list()
    assert s["optional_str"] == list()
