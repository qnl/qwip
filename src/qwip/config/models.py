import attrs
import pendulum
import sqlalchemy as sa
from attrs import field
from sqlalchemy import Column, ForeignKey, Table, UniqueConstraint, func
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.sql.expression import FunctionElement
from sqlalchemy.types import DateTime
from typing_extensions import Self

import qwip
from qwip.attrs import qdefine
from qwip.config.dolt import DoltTable
from qwip.config.metadata import QWIP_DB_METADATA, QWIP_DB_REGISTRY
from qwip.sequencer.waveform import REGISTERED_WAVEFORMS

JSONTypes = dict | list | bool | float | int | str | None


class utc_timestamp(FunctionElement):
    """Gets the current timestamp in UTC.

    Based on SQLAlchemy documentation [example](https://docs.sqlalchemy.org/en/20/core/compiler.html#utc-timestamp-function).
    """

    type = DateTime()
    inherit_cache = True


@compiles(utc_timestamp, "mysql")
def mysql_utc_timestamp(element, compiler, **kwargs):
    return "UTC_TIMESTAMP()"


@compiles(utc_timestamp, "sqlite")
def sqlite_utc_timestamp(element, compiler, **kwargs):
    return "DATETIME('now')"


@qdefine(slots=False)
class VersionControlled:
    @property
    def table(self) -> Table:
        return self.__table__

    def history(self, connection) -> list[str, Self]:
        table = self.table
        history_cols = table._dolt_history.columns

        cond = sa.and_(
            *(
                getattr(history_cols, col.name) == getattr(self, col.name)
                for col in table.primary_key.columns
            )
        )

        mapper = self.__mapper__

        aliases = dict()
        for f in attrs.fields(type(self)):
            columns = getattr(mapper.get_property(f.name), "columns", None)
            if columns and f.init:
                aliases[f.name] = columns[0].name

        # Everything before here can maybe be cached in the future.

        stmt = sa.select(table._dolt_history).where(cond)
        results = connection.execute(stmt)

        obj_results = []
        for row in results:
            obj = type(self)(
                **{
                    attribute: row._mapping.get(column)
                    for attribute, column in aliases.items()
                }
            )
            commit_hash = row._mapping.get("commit_hash")

            obj_results.append((commit_hash, obj))

        return obj_results


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
        repr=lambda dt: dt.in_tz("local").isoformat()
        if isinstance(dt, pendulum.DateTime)
        else repr(dt),
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
        sa.DateTime,
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
        from qwip.sequencer.waveform import Waveform

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
    sequence_element: "SequenceElementModel" = field(repr=False)


@qdefine(slots=False)
class ConstraintModel(VersionControlled):
    name: str
    location: str
    sequence_element: "SequenceElementModel" = field(repr=False)


@qdefine(slots=False)
class SequenceElementModel(VersionControlled):
    name: str
    width: str | None = None
    locations: list[WaveformLocationModel] = field(factory=list)
    constraints: dict[str, ConstraintModel] = field(factory=dict)

    @classmethod
    def from_sequence_element(cls, se, name):
        se_model = cls(name=name, width=str(se.width))

        for loc, wave in se.get_location_pairs():
            wave_model = WaveformModel.from_waveform(wave)
            pair = WaveformLocationModel(
                location=str(loc), waveform=wave_model, sequence_element=se_model
            )

        for name, expr in se.constraints.items():
            constraint = ConstraintModel(
                name=name, location=str(expr), sequence_element=se_model
            )

        return se_model

    def to_sequence_element(self):
        from qwip.sequencer.elements import SequenceElement
        from qwip.sequencer.utils import Location

        constraints = {
            n: Location.from_string(c.location) for n, c in self.constraints.items()
        }

        pairs = [
            (Location.from_string(waveloc.location), waveloc.waveform.to_waveform())
            for waveloc in self.locations
        ]

        return SequenceElement.fromtuples(
            pairs, width=self.width, constraints=constraints
        )


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
        "sequence_element_id",
        sa.Integer,
        ForeignKey(
            "sequence_elements.sequence_element_id",
            name="fk_waveform_locations_sequence_elements",
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
        "sequence_element_id",
        sa.Integer,
        ForeignKey(
            "sequence_elements.sequence_element_id",
            name="fk_constraints_sequence_elements",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
    ),
)

sequence_element_table = DoltTable(
    "sequence_elements",
    QWIP_DB_METADATA,
    Column("sequence_element_id", sa.Integer, primary_key=True, autoincrement=True),
    Column("name", sa.String(255)),
    Column("width", sa.String(255)),
    UniqueConstraint("name", name="uq_sequence_elements_name"),
)


QWIP_DB_REGISTRY.map_imperatively(
    WaveformLocationModel,
    waveform_location_table,
    properties=dict(
        waveform=relationship(
            WaveformModel,
            cascade="all",
        ),
        sequence_element=relationship(
            SequenceElementModel,
            back_populates="locations",
        ),
    ),
)

QWIP_DB_REGISTRY.map_imperatively(
    ConstraintModel,
    constraint_table,
    properties=dict(
        sequence_element=relationship(
            SequenceElementModel, back_populates="constraints"
        )
    ),
)

QWIP_DB_REGISTRY.map_imperatively(
    SequenceElementModel,
    sequence_element_table,
    properties=dict(
        locations=relationship(
            WaveformLocationModel,
            back_populates="sequence_element",
            cascade="all, delete-orphan",
        ),
        constraints=relationship(
            ConstraintModel,
            back_populates="sequence_element",
            collection_class=attribute_mapped_collection("name"),
            cascade="all, delete-orphan",
        ),
    ),
)

user_tables = list(QWIP_DB_METADATA.tables.values())

for table in user_tables:
    if isinstance(table, DoltTable):
        table.create_system_tables()


__all__ = [
    "VersionControlled",
    "Folder",
    "Parameter",
    "WaveformModel",
    "WaveformLocationModel",
    "ConstraintModel",
    "SequenceElementModel",
    "folder_table",
    "parameter_table",
    "waveform_table",
    "waveform_location_table",
    "constraint_table",
    "sequence_element_table",
]
