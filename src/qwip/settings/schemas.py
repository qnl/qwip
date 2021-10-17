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

@qattrs
class QNLSettings(Settings):
    @qattrs
    class Researcher(Settings):
        class Position(Enum):
            PI = "Principal Investigator"
            GRAD = "Graduate Student"
            POSTDOC = "Postdoctoral Scholar"
            SCIENTIST = "Research Scientist"
        
        name: str = attrib()
        position: Position = attrib()
        tenure: int = attrib()
    
    @qattrs
    class Fridge(Settings):
        name: str = attrib()
        install_date: Date = attrib()

    @qattrs
    class Science(Settings):
        @qattrs
        class ResearchArea(Settings):
            field: str = attrib()
            subfield: str = attrib()
        
        research_area: ResearchArea = attrib()
        num_publications: int = attrib()

    def positive(self, attribute, value):
        if value < 0:
            raise ValueError()

    principal: Researcher = attrib()
    num_members: int = attrib(validator=positive)
    members: Dict[str, Researcher] = attrib()
    fridges: List[Fridge] = attrib()
    science: Science = attrib()