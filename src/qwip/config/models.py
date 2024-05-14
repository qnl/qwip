from typing import Self

import pendulum
import sqlalchemy as sa
from attrs import field
from sqlalchemy import Column, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection

import qwip
from qwip.attrs import qdefine
from qwip.database.database import VersionControlled
from qwip.database.dolt import DoltTable
from qwip.database.metadata import QWIP_DB_METADATA, QWIP_DB_REGISTRY
from qwip.database.utils import JSONTypes, PendulumDateTime, utc_timestamp
from qwip.sequencer.waveform import Waveform


@qdefine(slots=False)
class Folder(VersionControlled):
    name: str
    parent: Self | None = field(repr=lambda f: f.path() if f else repr(f), default=None)

    def path(self) -> str:
        if self.parent is None:
            return f"/{self.name}/"

        return f"{self.parent.path()}{self.name}/"


@qdefine(slots=False)
class Parameter(VersionControlled):
    name: str
    folder: Folder | None = field(
        repr=lambda f: f.path() if f else repr(f), default=None
    )
    timestamp: pendulum.DateTime | None = field(
        repr=lambda dt: (
            dt.in_tz("local").isoformat()
            if isinstance(dt, pendulum.DateTime)
            else repr(dt)
        ),
        default=None,
    )
    value: JSONTypes | None = None

    def path(self) -> str:
        if self.folder is None:
            return f"/{self.name}"

        return f"{self.folder.path()}{self.name}"


folder_table = DoltTable(
    "folders",
    QWIP_DB_METADATA,
    Column("folder_id", sa.Integer, primary_key=True, autoincrement=True),
    Column("name", sa.String(255), nullable=False),
    Column(
        "parent_id",
        sa.Integer,
        ForeignKey(
            "folders.folder_id",
            name="fk_folders_folders",
            onupdate="CASCADE",
            ondelete="CASCADE",
            use_alter=True,
        ),
    ),
    UniqueConstraint("name", "folder_id", name="uq_folders_name_folder_id"),
)

parameter_table = DoltTable(
    "parameters",
    QWIP_DB_METADATA,
    Column("parameter_id", sa.Integer, primary_key=True, autoincrement=True),
    Column("name", sa.String(255), nullable=False),
    Column(
        "folder_id",
        sa.Integer,
        ForeignKey(
            "folders.folder_id",
            name="fk_parameters_folders",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
    ),
    Column(
        "last_modified",
        PendulumDateTime,
        default=utc_timestamp(),
        onupdate=utc_timestamp(),
    ),
    Column("value", sa.JSON),
    UniqueConstraint("name", "folder_id", name="uq_parameters_name_folder_id"),
)


QWIP_DB_REGISTRY.map_imperatively(
    Folder,
    folder_table,
    properties=dict(
        subfolders=relationship(
            Folder,
            cascade="all, delete-orphan",
            back_populates="parent",
            collection_class=attribute_mapped_collection("name"),
        ),
        parent=relationship(
            Folder, back_populates="subfolders", remote_side=[folder_table.c.folder_id]
        ),
        parameters=relationship(
            Parameter,
            cascade="all, delete-orphan",
            back_populates="folder",
            collection_class=attribute_mapped_collection("name"),
        ),
    ),
)

QWIP_DB_REGISTRY.map_imperatively(
    Parameter,
    parameter_table,
    properties=dict(
        folder=relationship(
            Folder,
            back_populates="parameters",
        ),
        timestamp=parameter_table.c.last_modified,
    ),
)

## =============== Waveforms =============== ##

waveform_table = DoltTable(
    "waveforms",
    QWIP_DB_METADATA,
    Column("waveform_id", sa.Integer, primary_key=True, autoincrement=True),
    Column("classname", sa.String(255), nullable=False),
    Column("properties", sa.JSON, nullable=False, default=dict),
    Column("key", sa.String(255)),
    Column(
        "parent_id",
        sa.Integer,
        ForeignKey(
            "waveforms.waveform_id",
            name="fk_waveforms_waveforms",
            onupdate="CASCADE",
            ondelete="CASCADE",
            use_alter=True,
        ),
    ),
)


@qdefine(slots=False)
class WaveformModel(VersionControlled):
    classname: str
    properties: dict[str] = field(factory=dict)
    key: str | None = None
    parent: Self | None = field(default=None, repr=False)
    children: dict[str, Self] = field(factory=dict)

    @classmethod
    def from_unstructured_wave(cls, wave_dict, parent=None, key=None):
        properties = {}
        children = {}
        for k, val in wave_dict.items():
            if isinstance(val, dict) and "__class__" in val:
                children[k] = val
            elif k != "__class__":
                properties[k] = val

        wave_model = cls(
            classname=wave_dict["__class__"],
            properties=properties,
            key=key,
            parent=parent,
        )

        for key, child_dict in children.items():
            cls.from_unstructured_wave(child_dict, parent=wave_model, key=key)

        return wave_model

    @classmethod
    def from_waveform(cls, wave):
        wave_dict = qwip.converter.unstructure(wave)
        return cls.from_unstructured_wave(wave_dict)

    def to_unstructured_waveform(self):
        unstructured = dict(__class__=self.classname, **self.properties)

        for k, wave_model in self.children.items():
            unstructured[k] = wave_model.to_unstructured_waveform()

        return unstructured

    def to_waveform(self):
        return qwip.converter.structure(self.to_unstructured_waveform(), Waveform)


QWIP_DB_REGISTRY.map_imperatively(
    WaveformModel,
    waveform_table,
    properties=dict(
        children=relationship(
            WaveformModel,
            cascade="all, delete-orphan",
            back_populates="parent",
            collection_class=attribute_mapped_collection("key"),
        ),
        parent=relationship(
            WaveformModel,
            back_populates="children",
            remote_side=[waveform_table.c.waveform_id],
        ),
    ),
)


@qdefine(slots=False)
class WaveformLocationModel(VersionControlled):
    location: str
    waveform: WaveformModel
    timeline: "TimelineModel" = field(repr=False)


@qdefine(slots=False)
class ConstraintModel(VersionControlled):
    name: str
    location: str
    timeline: "TimelineModel" = field(repr=False)


@qdefine(slots=False)
class TimelineModel(VersionControlled):
    name: str
    width: str | None = None
    locations: list[WaveformLocationModel] = field(factory=list)
    constraints: dict[str, ConstraintModel] = field(factory=dict)

    @classmethod
    def from_timeline(cls, se, name):
        se_model = cls(name=name, width=qwip.converter.unstructure(se.width))

        for loc, wave in se.get_location_pairs():
            wave_model = WaveformModel.from_waveform(wave)
            pair = WaveformLocationModel(
                location=str(loc), waveform=wave_model, timeline=se_model
            )

        for name, expr in se.constraints.items():
            constraint = ConstraintModel(
                name=name, location=str(expr), timeline=se_model
            )

        return se_model

    def to_timeline(self):
        from qwip.sequencer.timeline import Timeline
        from qwip.sequencer.utils import Location

        constraints = {
            n: Location.from_string(c.location) for n, c in self.constraints.items()
        }

        pairs = [
            (Location.from_string(waveloc.location), waveloc.waveform.to_waveform())
            for waveloc in self.locations
        ]

        return Timeline.fromtuples(pairs, width=self.width, constraints=constraints)


waveform_location_table = DoltTable(
    "waveform_locations",
    QWIP_DB_METADATA,
    Column("location", sa.String(255), nullable=False),
    Column(
        "waveform_id",
        sa.Integer,
        ForeignKey(
            "waveforms.waveform_id",
            name="fk_waveform_locations_waveforms",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    ),
    Column(
        "timeline_id",
        sa.Integer,
        ForeignKey(
            "timelines.timeline_id",
            name="fk_waveform_locations_timelines",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    ),
)

constraint_table = DoltTable(
    "constraints",
    QWIP_DB_METADATA,
    Column("constraint_id", sa.Integer, primary_key=True, autoincrement=True),
    Column("name", sa.String(255)),
    Column("location", sa.String(255)),
    Column(
        "timeline_id",
        sa.Integer,
        ForeignKey(
            "timelines.timeline_id",
            name="fk_constraints_timelines",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
    ),
)

timeline_table = DoltTable(
    "timelines",
    QWIP_DB_METADATA,
    Column("timeline_id", sa.Integer, primary_key=True, autoincrement=True),
    Column("name", sa.String(255)),
    Column("width", sa.String(255)),
    UniqueConstraint("name", name="uq_timelines_name"),
)


QWIP_DB_REGISTRY.map_imperatively(
    WaveformLocationModel,
    waveform_location_table,
    properties=dict(
        waveform=relationship(
            WaveformModel,
            cascade="all",
        ),
        timeline=relationship(
            TimelineModel,
            back_populates="locations",
        ),
    ),
)

QWIP_DB_REGISTRY.map_imperatively(
    ConstraintModel,
    constraint_table,
    properties=dict(timeline=relationship(TimelineModel, back_populates="constraints")),
)

QWIP_DB_REGISTRY.map_imperatively(
    TimelineModel,
    timeline_table,
    properties=dict(
        locations=relationship(
            WaveformLocationModel,
            back_populates="timeline",
            cascade="all, delete-orphan",
        ),
        constraints=relationship(
            ConstraintModel,
            back_populates="timeline",
            collection_class=attribute_mapped_collection("name"),
            cascade="all, delete-orphan",
        ),
    ),
)

config_tables = [
    folder_table,
    parameter_table,
    waveform_table,
    waveform_location_table,
    constraint_table,
    timeline_table,
]

for table in config_tables:
    if isinstance(table, DoltTable):
        table.create_system_tables()


__all__ = [
    "VersionControlled",
    "Folder",
    "Parameter",
    "WaveformModel",
    "WaveformLocationModel",
    "ConstraintModel",
    "TimelineModel",
    "folder_table",
    "parameter_table",
    "waveform_table",
    "waveform_location_table",
    "constraint_table",
    "timeline_table",
]
