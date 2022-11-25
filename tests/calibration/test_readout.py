import time

import pytest

from loguru import logger

from qwip.processing.process import ProcessSettings
from qwip.calibration.readout import ReadoutCalibration

class TestReadoutCalibration:
    
    def test_init(self):
        x = ReadoutCalibration(
            processing=[
                ProcessSettings(name='heterodyne', process_type='utils.FormatLegacyHeterodyne'),
                ProcessSettings(name='collect', process_type='utils.CollectData', inputs=('heterodyne',)),
            ]
        )
        print(x)