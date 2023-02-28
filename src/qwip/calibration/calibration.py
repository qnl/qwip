from pathlib import Path
from typing import Any, Callable, Optional

import pendulum
from attr import field
from cattr.converters import GenConverter
from pendulum import DateTime
from qtrl.amiable_sequencer import AM_Sequence

import qwip
from qwip.flatdict import FlatDict
from qwip.processing.pipeline import Pipeline
from qwip.processing.process import ProcessSettings
from qwip.settings.settings import qdefine


@qdefine
class Parameter:
    name: str
    value: Any
    timestamp: DateTime = field(factory=pendulum.now)
    metadata: FlatDict = field(factory=FlatDict)


@qdefine
class CalibrationLogger:
    parameter: Parameter
    logpath: Path = field()
    extension: str = ".yaml"
    fmt: str = "YYYY-MM-DDTHH-mm-ss"

    @logpath.default
    def _from_qsettings(self):
        return (
            qwip.qsettings["logging/directory"] / f"calibration/{self.parameter.name}"
        )

    def get_filename(self, timestamp: DateTime = None) -> str:
        if timestamp is None:
            timestamp = pendulum.now()

        formatted_ts = timestamp.format(self.fmt)
        return formatted_ts + "_" + self.parameter.name + self.extension

    def update(self, results: FlatDict, filename: str = None):
        if filename is None:
            filename = self.parameter.name + self.extension

        fullpath = self.logpath / filename
        if fullpath.exists():
            with open(fullpath, "r") as f:
                current = FlatDict(qwip.yaml.load(f))
        else:
            current = FlatDict

        current.update(results)

        with open(fullpath, "w") as f:
            qwip.yaml.dump(current, f)

    def log(self, results: FlatDict):
        fullpath = self.logpath / self.get_filename()

        with open(fullpath, "w") as f:
            qwip.yaml.dump(results, f)

    def load(self, filename=None):
        if filename is None:
            filename = self.logpath / (self.parameter.name + self.extension)

        with open(filename, "r") as f:
            results = FlatDict(qwip.yaml.load(f))

        return results


@qdefine
class Calibration:
    @qdefine
    class Result:
        timestamp: DateTime = field(factory=pendulum.now)

        def unstructure(self, converter=qwip.converter):
            return converter.unstructure(self)

        @classmethod
        def structure(cls, data, converter=qwip.converter):
            return converter.structure(data, cls)

    processing: Pipeline
    results: FlatDict[str, Result] = field(factory=FlatDict)
    # logger: CalibrationLogger = field()

    def get_sequence(self) -> AM_Sequence:
        raise NotImplementedError

    def analysis(self, data, **kwargs):
        raise NotImplementedError

    # @logger.default
    # def _create_default_logger(self):
    #     return CalibrationLogger(parameter=self.parameter)
