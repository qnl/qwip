from typing import Any

import pendulum
import sqlalchemy as sa

from sqlalchemy import (
    Column,
    Table,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column
)

from qwip.config.metadata import QWIP_DB_METADATA, QWIP_DB_REGISTRY

def dolt_procedure(
    connection,
    func: str,
    *args,
) -> Any:
    arg_text = ', '.join(f"'{a}'" for a in args)
    stmt = sa.text(f'CALL {func.upper()}({arg_text})')

    result = connection.execute(stmt).scalar_one()
    return result

def dolt_add(
    connection,
    tables: list[str] | None = None,
) -> None:
    if not tables:
        tables = ['-A']

    return dolt_procedure(connection, 'DOLT_ADD', *tables)

def dolt_branch(
    connection,
    branch: str,
    other_branch: str | None = None,
    action: str = 'create',
    force: bool = False,
) -> None:
    """
    """
    args = []

    match action.lower():
        case 'create':
            pass
        case 'copy':
            args.append('-c')
        case 'move':
            args.append('-m')
        case 'delete':
            args.append('-d')
        case invalid:
            raise ValueError(f'{invalid} is not a valid action for dolt_branch.')
    
    if force:
        args.append('-f')

    args.append(branch)

    if other_branch:
        args.append(other_branch)

    return dolt_procedure(connection, 'DOLT_BRANCH', *args)

def dolt_checkout(
    connection,
    target: str,
    new_branch: bool = False
) -> None:
    args = ('-b', target) if new_branch else (target,)

    return dolt_procedure(connection, 'DOLT_CHECKOUT', *args)

def dolt_commit(
    connection,
    message: str,
    add: str | None = None,
    date: pendulum.DateTime | None = None,
    author: str | None = None,
    allow_empty: bool = False
) -> None:
    args = ['-m', message]

    if add and add.lower() == 'modified':
        args.append('-a')
    elif add and add.lower() == 'all':
        args.append('-A')
    
    if date:
        args += ['--date', str(date)]
    
    if author:
        args += ['--author', author]

    if allow_empty:
        args.append(['--allow-empty'])

    return dolt_procedure(connection, 'DOLT_COMMIT', *args)

def dolt_reset(
    connection,
    branch_or_commit: str | None = None,
    hard: bool = False,
) -> None:
    args = ['--hard'] if hard else []

    if branch_or_commit:
        args.append(branch_or_commit)

    return dolt_procedure(connection, 'DOLT_RESET', *args)

@QWIP_DB_REGISTRY.mapped_as_dataclass
class DoltLog:
    __tablename__ = 'dolt_log'
    commit_hash: Mapped[str] = mapped_column(sa.Text, primary_key=True, system=True)
    committer: Mapped[str] = mapped_column(sa.Text, system=True)
    email: Mapped[str] = mapped_column(sa.Text, system=True)
    date: Mapped[pendulum.DateTime] = mapped_column(sa.DateTime, system=True) 
    message: Mapped[str] = mapped_column(sa.Text, system=True)

@QWIP_DB_REGISTRY.mapped_as_dataclass
class DoltCommit:
    __tablename__ = 'dolt_commits'
    commit_hash: Mapped[str] = mapped_column(sa.Text, primary_key=True, system=True)
    committer: Mapped[str] = mapped_column(sa.Text, system=True)
    email: Mapped[str] = mapped_column(sa.Text, system=True)
    date: Mapped[pendulum.DateTime] = mapped_column(sa.DateTime, system=True) 
    message: Mapped[str] = mapped_column(sa.Text, system=True)

@QWIP_DB_REGISTRY.mapped_as_dataclass
class DoltDiff:
    __tablename__ = 'dolt_diff'
    commit_hash: Mapped[str] = mapped_column(sa.Text, primary_key=True, system=True)
    table_name: Mapped[str] = mapped_column(sa.Text, primary_key=True, system=True)
    committer: Mapped[str] = mapped_column(sa.Text, system=True)
    email: Mapped[str] = mapped_column(sa.Text, system=True)
    date: Mapped[pendulum.DateTime] = mapped_column(sa.DateTime, system=True) 
    message: Mapped[str] = mapped_column(sa.Text, system=True)
    data_change: Mapped[bool] = mapped_column(system=True)
    schema_change: Mapped[bool] = mapped_column(system=True)

@QWIP_DB_REGISTRY.mapped_as_dataclass
class DoltBranch:
    __tablename__ = 'dolt_branches'
    name: Mapped[str] = mapped_column(sa.Text, primary_key=True, system=True)
    hash: Mapped[str] = mapped_column(sa.Text, system=True)
    latest_committer: Mapped[str] = mapped_column(sa.Text, system=True)
    latest_committer_email: Mapped[str] = mapped_column(sa.Text, system=True)
    latest_commit_date: Mapped[pendulum.DateTime] = mapped_column(sa.DateTime, system=True)
    latest_commit_message: Mapped[str] = mapped_column(sa.Text, system=True)


@QWIP_DB_REGISTRY.mapped_as_dataclass
class DoltStatus:
    __tablename__ = 'dolt_status'
    table_name: Mapped[str] = mapped_column(sa.Text, primary_key=True, system=True)
    staged: Mapped[bool] = mapped_column(sa.Boolean, system=True)
    status: Mapped[str] = mapped_column(sa.Text, system=True)

class DoltTable(Table):
    def add(self, connection) -> None:
        dolt_add(connection, tables=[self.name])

    def commit(self,
        connection,
        message: str,
        author: str | None = None,
        date: pendulum.DateTime | None = None,
        allow_empty: bool = False
    ) -> None:
        dolt_commit(connection, message, author=author, date=date, allow_empty=allow_empty)

    def create_system_tables(self, metadata=None):
        self._dolt_history = self.create_history_table(metadata=metadata)
        self._dolt_diff = self.create_diff_table(metadata=metadata)
        self._dolt_blame = self.create_blame_table(metadata=metadata)

    def create_history_table(self, metadata=None):
        metadata = metadata or self.metadata
        name = f'dolt_history_{self.name}'

        if name in metadata.tables:
            raise ValueError(f'Table {name} already exists in metadata object: {metadata.tables}.')

        history_table = self.to_metadata(
            metadata,
            name=name
        )
        history_table.append_column(Column('commit_hash', sa.Text, primary_key=True, system=True))
        history_table.append_column(Column('committer', sa.Text, system=True))
        history_table.append_column(Column('commit_date', sa.DateTime, system=True))

        return history_table

    def create_diff_table(self, metadata=None):
        metadata = metadata or self.metadata
        name = f'dolt_diff_{self.name}'

        if name in metadata.tables:
            raise ValueError(f'Table {name} already exists in metadata object: {metadata.tables}.')

        to_columns = [
            Column(f'to_{col.name}', col.type, primary_key=col.primary_key, system=True) for col in self.columns
        ]

        from_columns = [
            Column(f'from_{col.name}', col.type, system=True) for col in self.columns
        ]

        return Table(
            name,
            metadata,
            Column('from_commit', sa.Text, primary_key=True, system=True),
            Column('from_commit_date', sa.DateTime, system=True),
            Column('to_commit', sa.Text, primary_key=True, system=True),
            Column('to_commit_date', sa.DateTime, system=True),
            Column('diff_type', sa.Text, system=True),
            *to_columns,
            *from_columns
        )

    def create_blame_table(self, metadata=None):
        metadata = metadata or self.metadata
        name = f'dolt_blame_{self.name}'

        if name in metadata.tables:
            raise ValueError(f'Table {name} already exists in metadata object: {metadata.tables}.')

        pk_columns = [
            Column(col.name, col.type, primary_key=True, system=True) for col in self.columns if col.primary_key
        ]

        return Table(
            name,
            metadata,
            Column('commit', sa.Text, primary_key=True, system=True),
            Column('commit_date', sa.DateTime, system=True),
            Column('committer', sa.Text, system=True),
            Column('email', sa.Text, system=True),
            Column('message', sa.Text, system=True),
            *pk_columns
        )