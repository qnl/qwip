import sqlalchemy as sa
from sqlalchemy import (
    Table,
    Column,
    ForeignKey,
    UniqueConstraint,
    func
)
from sqlalchemy.orm import (
    relationship,
)
from sqlalchemy.orm.collections import attribute_mapped_collection

import pendulum
from attrs import field
from typing_extensions import Self

from qwip.settings.settings import qdefine
from qwip.config.metadata import QWIP_DB_METADATA, QWIP_DB_REGISTRY
from qwip.config.dolt import (
    DoltTable
)

JSONTypes = dict | list | bool | float | int | str | None


from qwip.config.metadata import QWIP_DB_METADATA, QWIP_DB_REGISTRY
import attrs

@qdefine(slots=False)
class VersionControlled:
    @property
    def table(self) -> Table:
        return self.__table__

    def history(self, connection) -> list[str, Self]:
        table = self.table
        history_cols = table._dolt_history.columns

        cond = sa.and_(
            *(getattr(history_cols, col.name) == getattr(self, col.name) for col in table.primary_key.columns)
        )
    
        mapper = self.__mapper__

        aliases = dict()
        for f in attrs.fields(type(self)):
            columns = getattr(mapper.get_property(f.name), 'columns', None)
            if columns and f.init:
                aliases[f.name] = columns[0].name

        # Everything before here can maybe be cached in the future.

        stmt = sa.select(table._dolt_history).where(cond)
        results = connection.execute(stmt)

        obj_results = []
        for row in results:
            obj = type(self)(**{
                attribute: row._mapping.get(column) for attribute, column in aliases.items()
            })
            commit_hash = row._mapping.get('commit_hash')

            obj_results.append((commit_hash, obj))

        return obj_results

@qdefine(slots=False)
class Folder(VersionControlled):
    name: str
    parent: Self | None = field(
        repr=lambda f: f.path() if f else repr(f),
        default=None
    )

    def path(self) -> str:
        if self.parent is None:
            return f'/{self.name}/'

        return f'{self.parent.path()}{self.name}/'

@qdefine(slots=False)
class Parameter(VersionControlled):
    name: str
    folder: Folder | None = field(
        repr=lambda f: f.path() if f else repr(f),
        default=None
    )
    timestamp: pendulum.DateTime | None = field(
        repr=lambda dt: dt.in_tz('local').isoformat() if isinstance(dt, pendulum.DateTime) else repr(dt),
        default=None
    )
    value: JSONTypes | None = None

    def path(self) -> str:
        if self.folder is None:
            return f'/{self.name}'

        return f'{self.folder.path()}{self.name}'

folder_table = DoltTable(
    'folders',
    QWIP_DB_METADATA,
    Column('folder_id', sa.Integer, primary_key=True, autoincrement=True),
    Column('name', sa.String(255), nullable=False),
    Column(
        'parent_id',
        sa.Integer,
        ForeignKey(
            'folders.folder_id',
            name='fk_folders_folders',
            onupdate='CASCADE',
            ondelete='CASCADE'
        )
    ),
    UniqueConstraint('name', 'folder_id', name='uq_folders_name_folder_id')
)

parameter_table = DoltTable(
    'parameters',
    QWIP_DB_METADATA,
    Column('parameter_id', sa.Integer, primary_key=True, autoincrement=True),
    Column('name', sa.String(255), nullable=False),
    Column(
        'folder_id',
        sa.Integer,
        ForeignKey(
            'folders.folder_id',
            name='fk_parameters_folders',
            onupdate='CASCADE',
            ondelete='CASCADE'
        ),
    ),
    Column('last_modified',
        sa.DateTime,
        default=func.utc_timestamp(),
        onupdate=func.utc_timestamp()
    ),
    Column('value', sa.JSON),
    UniqueConstraint('name', 'folder_id', name='uq_parameters_name_folder_id')
)

parameter_table.create_system_tables()
folder_table.create_system_tables()

QWIP_DB_REGISTRY.map_imperatively(
    Folder,
    folder_table,
    properties=dict(
        subfolders=relationship(
            Folder,
            cascade='all, delete-orphan',
            back_populates='parent',
            collection_class=attribute_mapped_collection('name')
        ),
        parent=relationship(
            Folder,
            back_populates='subfolders',
            remote_side=[folder_table.c.folder_id]
        ),
        parameters=relationship(
            Parameter,
            cascade='all, delete-orphan',
            back_populates='folder',
            collection_class=attribute_mapped_collection('name')
        ),
    ),
)

QWIP_DB_REGISTRY.map_imperatively(
    Parameter,
    parameter_table,
    properties=dict(
        folder=relationship(
            Folder,
            back_populates='parameters',
        ),
        timestamp=parameter_table.c.last_modified
    ),
)