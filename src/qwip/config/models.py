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
class Parameter(VersionControlled):
    name: str
    parent: Self | None = None
    timestamp: pendulum.DateTime | None = field(
        repr=lambda dt: dt.in_tz('local').isoformat() if isinstance(dt, pendulum.DateTime) else repr(dt),
        default=None
    )
    value: JSONTypes = None


parameter_table = DoltTable(
    'parameters',
    QWIP_DB_METADATA,
    Column('parameter_id', sa.Integer, primary_key=True, autoincrement=True),
    Column('name', sa.String(50), nullable=False),
    Column(
        'parent_id',
        sa.Integer,
        ForeignKey(
            'parameters.parameter_id',
            name='fk_parameters_parameters',
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
    UniqueConstraint('name', 'parent_id', name='uq_parameters_name_parent_id')
)

parameter_table.create_system_tables()

QWIP_DB_REGISTRY.map_imperatively(
    Parameter,
    parameter_table,
    properties=dict(
        parameters=relationship(
            Parameter,
            cascade='all, delete-orphan',
            back_populates='parent',
            collection_class=attribute_mapped_collection('name')
        ),
        parent=relationship(
            Parameter,
            back_populates='parameters',
            remote_side=[parameter_table.c.parameter_id]
        ),
        timestamp=parameter_table.c.last_modified
    ),
)