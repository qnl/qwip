import pytest
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from attr import attrib

import qwip.processing as qproc

from qwip.flatdict import FlatDict
from qwip.settings.settings import qdefine
from qwip.processing.process import (
    get_process_type, Process, ProcessSettings
)

logger.enable('qwip')

@qdefine
class ExampleProcess(Process):
    a: dict[str, float]
    b: NDArray = np.zeros(5)
    c: int = 1
    d: float = attrib(
        metadata=dict(serialize=False),
        factory=float
    )

    def run(self, data, /, **kwargs):
        return data

class TestGetProcessType:
    def test_from_class(self):
        ptype = get_process_type(ExampleProcess)

        assert ptype is ExampleProcess

    def test_from_processing(self):
        ptype = get_process_type('classification.IQRotation')
        
        assert ptype is qproc.classification.IQRotation

    def test_full_path(self):
        ptype = get_process_type('qwip.processing.classification.GMM')

        assert ptype is qproc.classification.GMM

@pytest.fixture
def iq_rotation_settings():
    return FlatDict(
        name='rotate',
        process_type='classification.IQRotation',
        parameters=dict(angles={})
    )

@pytest.fixture
def example_settings():
    return FlatDict(name='ex1', process_type=ExampleProcess)

class TestProcessSettings:
    def test_initialize(self, iq_rotation_settings):
        psetting = ProcessSettings(**iq_rotation_settings)
        
        assert psetting.name == 'rotate'
        assert psetting.process_type is qproc.classification.IQRotation

    def test_validate_process(self, iq_rotation_settings):
        iq_rotation_settings['process_type'] = 'some_module.class'
        with pytest.raises(ModuleNotFoundError):
            ProcessSettings(**iq_rotation_settings)

        iq_rotation_settings['process_type'] = 'qwip.settings.settings.Settings'
        with pytest.raises(TypeError):
            ProcessSettings(**iq_rotation_settings)

    def test_get_process(self, iq_rotation_settings):
        iq_rotation_settings = ProcessSettings(**iq_rotation_settings)

        iq = iq_rotation_settings.get_process()

        assert type(iq) is qproc.classification.IQRotation

class TestProcess:
    def test_run_process(self):
        psetting = ProcessSettings(name='base', process_type='process.Process')

        base = psetting.get_process()
        
        with pytest.raises(NotImplementedError):
            base('data')

    def test_update_process_settings(self):
        psetting = ProcessSettings(name='example', process_type=ExampleProcess)

        ex = psetting.get_process(a={'R0': 2, 'R1': 3})

        assert ex.settings.parameters == {name: getattr(ex, name) for name in 'a'}

        ex.update_settings()

        for name in 'abc':
            assert np.array_equal(ex.settings.parameters[name], getattr(ex, name))