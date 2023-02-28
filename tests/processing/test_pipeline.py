from importlib import import_module

import pytest
from loguru import logger

from qwip.processing.pipeline import Pipeline
from qwip.processing.process import Process, ProcessSettings

logger.enable("qwip")


class TagProcess(Process):
    def run(self, *data):
        return (self.settings.name, *data)


class TestPipeline:
    @pytest.fixture
    def psettings(self):
        return [
            ProcessSettings(name="a", process_type=TagProcess),
            ProcessSettings(name="b", process_type=TagProcess, inputs=("a",)),
            ProcessSettings(name="c", process_type=TagProcess, inputs=("b",)),
            ProcessSettings(name="d", process_type=TagProcess, inputs=("a",)),
            ProcessSettings(name="e", process_type=TagProcess, inputs=("c", "d")),
            ProcessSettings(name="f", process_type=TagProcess, inputs=("c",)),
        ]

    def test_empty_pipeline(self):
        pipeln = Pipeline.from_process_settings([])

        assert bool(pipeln.processes) == False

    def test_create_from_process_settings(self, psettings):
        pipeln = Pipeline.from_process_settings(psettings)

        for ps in psettings:
            assert pipeln.processes[ps.name].settings is ps

    def test_run(self, psettings):
        pipeln = Pipeline.from_process_settings(psettings)

        assert pipeln.run("", "a") == ("a", "")
        assert pipeln.run("", "b") == ("b", ("a", ""))
        assert pipeln.run("", "e") == ("e", ("c", ("b", ("a", ""))), ("d", ("a", "")))
        assert pipeln.run("", "f") == ("f", ("c", ("b", ("a", ""))))
