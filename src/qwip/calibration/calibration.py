from typing import Any, Optional, Callable
from pathlib import Path
from cattr.converters import GenConverter

import pendulum
from attr import attrib
from pendulum import DateTime

from qtrl.amiable_sequencer import AM_Sequence

import qwip
from qwip.flatdict import FlatDict
from qwip.settings.settings import qattrs
from qwip.processing.process import ProcessSettings
from qwip.processing.pipeline import Pipeline

@qattrs
class Parameter:
    name: str
    value: Any
    timestamp: DateTime = attrib(factory=pendulum.now)
    metadata: FlatDict = attrib(factory=FlatDict)

@qattrs
class CalibrationLogger:
    parameter: Parameter
    logpath: Path = attrib()
    extension: str = '.yaml'
    fmt: str = 'YYYY-MM-DDTHH-mm-ss'

    @logpath.default
    def _from_qsettings(self):
        return qwip.qsettings['logging/directory'] / f'calibration/{self.parameter}'

    def get_filename(self, timestamp: DateTime = None) -> str:
        if timestamp is None:
            timestamp = pendulum.now()

        formatted_ts = timestamp.format(self.fmt)
        return formatted_ts + '_' + self.parameter + self.extension

    def update(self, results: FlatDict, filename: str = None):
        if filename is None:
            filename = self.parameter + self.extension

        fullpath = self.logpath / filename
        if fullpath.exists():
            with open(fullpath, 'r') as f:
                current = FlatDict(qwip.yaml.load(f))
        else:
            current = FlatDict

        current.update(results)

        with open(fullpath, 'w') as f:
            qwip.yaml.dump(current, f)

    def log(self, results: FlatDict):    
        fullpath = self.logpath / self.get_filename()

        with open(fullpath, 'w') as f:
            qwip.yaml.dump(results, f)

    def load(self, filename=None):
        if filename is None:
            filename = self.logpath / (self.parameter + self.extension)
        
        with open(filename, 'r') as f:
            results = FlatDict(qwip.yaml.load(f))

        return results

@qattrs
class Calibration:
    @qattrs
    class Result:
        timestamp: DateTime = attrib(factory=pendulum.now)

        def unstructure(self, converter=qwip.converter):
            return converter.unstructure(self)
        
        @classmethod
        def structure(cls, data, converter=qwip.converter):
            return converter.structure(data, cls)

    processing: Pipeline
    results: FlatDict[str, Result] = attrib(factory=FlatDict)
    # logger: CalibrationLogger = attrib()

    def get_sequence(self) -> AM_Sequence:
        raise NotImplementedError

    def analysis(self, data, **kwargs):
        raise NotImplementedError

    # @logger.default
    # def _create_default_logger(self):
    #     return CalibrationLogger(parameter=self.parameter)