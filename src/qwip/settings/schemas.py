from attr import attrib
from attr.validators import instance_of, in_, deep_iterable, deep_mapping
from enum import Enum
from typing import Type, List, Dict, Any

from pendulum import Date, DateTime, instance

from qwip.settings.settings import (
    Settings, qattrs
)

from qwip.settings.parameters import Parameters

@qattrs
class DataSettings(Settings):
    datadir : str = attrib(validator=instance_of(str))
    datetime_fmt : str = attrib(validator=instance_of(str))

@qattrs
class QWiPSettings(Settings):
    num : int = attrib(default=0, validator=instance_of(int))
    data : DataSettings = attrib(validator=instance_of(DataSettings))