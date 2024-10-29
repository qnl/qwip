from typing import Self

import pendulum
import sqlalchemy as sa
import sympy as sym
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
from qwip.sequencer.waveform import Operation


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

operation_table = DoltTable(
    "operations",
    QWIP_DB_METADATA,
    Column("operation_id", sa.Integer, primary_key=True, autoincrement=True),
    Column("classname", sa.String(255), nullable=False),
    Column("properties", sa.JSON, nullable=False, default=dict),
    Column("key", sa.String(255)),
    Column(
        "parent_id",
        sa.Integer,
        ForeignKey(
            "operations.operation_id",
            name="fk_operations_operations",
            onupdate="CASCADE",
            ondelete="CASCADE",
            use_alter=True,
        ),
    ),
)


@qdefine(slots=False)
class OperationModel(VersionControlled):
    classname: str
    properties: dict[str] = field(factory=dict)
    key: str | None = None
    parent: Self | None = field(default=None, repr=False)
    children: dict[str, Self] = field(factory=dict)

    @classmethod
    def from_unstructured_op(cls, op_dict, parent=None, key=None):
        properties = {}
        children = {}
        for k, val in op_dict.items():
            if isinstance(val, dict) and "__class__" in val:
                children[k] = val
            elif k != "__class__":
                properties[k] = val

        op_model = cls(
            classname=op_dict["__class__"],
            properties=properties,
            key=key,
            parent=parent,
        )

        for key, child_dict in children.items():
            cls.from_unstructured_op(child_dict, parent=op_model, key=key)

        return op_model

    @classmethod
    def from_operation(cls, op):
        op_dict = qwip.converter.unstructure(op)
        return cls.from_unstructured_op(op_dict)

    def to_unstructured_operation(self):
        unstructured = dict(__class__=self.classname, **self.properties)

        for k, op_model in self.children.items():
            unstructured[k] = op_model.to_unstructured_operation()

        return unstructured

    def to_operation(self):
        return qwip.converter.structure(self.to_unstructured_operation(), Operation)


QWIP_DB_REGISTRY.map_imperatively(
    OperationModel,
    operation_table,
    properties=dict(
        children=relationship(
            OperationModel,
            cascade="all, delete-orphan",
            back_populates="parent",
            collection_class=attribute_mapped_collection("key"),
        ),
        parent=relationship(
            OperationModel,
            back_populates="children",
            remote_side=[operation_table.c.operation_id],
        ),
    ),
)


@qdefine(slots=False)
class OperationLocationModel(VersionControlled):
    location: str
    operation: OperationModel
    timeline: "TimelineModel" = field(repr=False)


@qdefine(slots=False)
class ConstraintModel(VersionControlled):
    expression: str
    timeline: "TimelineModel" = field(repr=False)


@qdefine(slots=False)
class TimelineModel(VersionControlled):
    name: str
    width: str | None = None
    locations: list[OperationLocationModel] = field(factory=list)
    constraints: list[ConstraintModel] = field(factory=dict)

    @classmethod
    def from_timeline(cls, tmln, name):
        tmln_model = cls(name=name, width=qwip.converter.unstructure(tmln.width))

        for loc, op in tmln:
            op_model = OperationModel.from_operation(op)
            pair = OperationLocationModel(
                location=qwip.converter.unstructure(loc),
                operation=op_model,
                timeline=tmln_model,
            )

        for expr in tmln.constraints:
            constraint = ConstraintModel(
                expression=qwip.converter.unstructure(expr), timeline=tmln_model
            )

        return tmln_model

    def to_timeline(self):
        from qwip.sequencer.timeline import Timeline

        constraints = {c.expression for c in self.constraints}
        pairs = [
            (oploc.location, oploc.operation.to_operation()) for oploc in self.locations
        ]

        return Timeline.fromtuples(pairs, width=self.width, constraints=constraints)


operation_location_table = DoltTable(
    "operation_locations",
    QWIP_DB_METADATA,
    Column("location", sa.String(255), nullable=False),
    Column(
        "operation_id",
        sa.Integer,
        ForeignKey(
            "operations.operation_id",
            name="fk_operation_locations_operations",
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
            name="fk_operation_locations_timelines",
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
    Column("expression", sa.String(255)),
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
    OperationLocationModel,
    operation_location_table,
    properties=dict(
        operation=relationship(
            OperationModel,
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
            OperationLocationModel,
            back_populates="timeline",
            cascade="all, delete-orphan",
        ),
        constraints=relationship(
            ConstraintModel,
            back_populates="timeline",
            collection_class=list,
            cascade="all, delete-orphan",
        ),
    ),
)

config_tables = [
    folder_table,
    parameter_table,
    operation_table,
    operation_location_table,
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
    "OperationModel",
    "OperationLocationModel",
    "ConstraintModel",
    "TimelineModel",
    "folder_table",
    "parameter_table",
    "operation_table",
    "operation_location_table",
    "constraint_table",
    "timeline_table",
]
