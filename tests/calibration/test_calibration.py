import pytest
import pendulum

import qwip
from qwip.calibration.calibration import Calibration, CalibrationLogger

class TestCalibrationLogger:
    @pytest.fixture
    def param_logger(self):
        return CalibrationLogger(parameter=dict(name='param', value='value'))

    @pytest.fixture
    def now(self):
        return pendulum.datetime(2006, 1, 2, 15, 4, 5)
    
    def test_default_logpath(self, param_logger):
        expected = qwip.qsettings['logging/directory'] / 'calibration/param'
        assert param_logger.logpath == expected

    def test_timestamp(self, param_logger, now):
        with pendulum.test(now):
            fname = param_logger.get_filename(now)
        
        assert fname == '2006-01-02T15-04-05_param.yaml'

    def test_log(self, param_logger, now, tmp_path):
        param_logger.logpath = tmp_path
        
        x = Calibration.Result()
        # print(x.unstructure())
        param_logger.log(results={'test': x.unstructure()})
        print(x)