import functools
import re
import time
from enum import Enum

import sqlalchemy as sa
import typer
from packaging.version import Version
from rich import print
from rich.console import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

import qwip
from qwip.config import ConfigDB
from qwip.config.models import config_tables
from qwip.data.models import datastore_tables
from qwip.database.database import DoltDB
from qwip.database.metadata import QWIP_DB_METADATA

app = typer.Typer(no_args_is_help=True)


class DBType(str, Enum):
    config = "config"
    datastore = "datastore"


def add_progress(
    maybe_func=None, *, description: str = "Processing...", sleep: int = 0
):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description=description)
                if sleep:
                    time.sleep(sleep)
                return func(*args, **kwargs)

        return wrapped

    return decorator if maybe_func is None else decorator(maybe_func)


@add_progress(description="Testing database connection...", sleep=1)
def test_connection(db: DoltDB, validate: bool = True):
    try:
        db.connect(test=True, validate=validate)
        return True
    except sa.exc.OperationalError as e:
        print(e)
        return False


@add_progress(description="Creating database...", sleep=1)
def create_database(db: DoltDB, name: str):
    stmt = sa.text(f"CREATE DATABASE {name}")
    try:
        with db.session.begin():
            db.session.execute(stmt)
        print("Successfully created database!")
        return True

    except (sa.exc.ProgrammingError, sa.exc.OperationalError) as e:
        if db_exc := e.orig:
            code = db_exc.args[0]
            if code == 1007:
                print(f"Database {name} already exists!")
                return True

        print(e)
        return False


@add_progress(description="Adding tables to database...", sleep=1)
def create_tables(db: DoltDB, database: str, db_type: DBType):
    table = Table(title=database)
    table.add_column("Table")
    table.add_column("Columns")

    match db_type:
        case DBType.config:
            tables = config_tables
        case DBType.datastore:
            tables = datastore_tables

    to_add = {t.name: t for t in tables}
    QWIP_DB_METADATA.create_all(db.engine, tables=to_add.values())

    with db.session.begin():
        in_db = db.session.scalars(sa.text("SHOW TABLES")).all()

    if missing := set(to_add) - set(in_db):
        for name in missing:
            table.add_row(name, str(len(to_add[name].columns)))

        print("ERROR: Missing tables!")
        print(table)
        raise typer.Exit()

    for db_table in to_add.values():
        table.add_row(db_table.name, str(len(db_table.columns)))

    print("Successfully created tables!")
    print(table)

    if status := db.status(status="new table"):
        db.add([s.table for s in status])
        new_tables = ", ".join((s.table for s in status))

        commit = db.commit(f"Created tables: {new_tables}")
        branch = db.current_branch()

        print(Text(f"[{branch.name} {commit.short_hash}] {commit.message}"))
        print(f"{len(status)} tables added")


@add_progress(description="Getting all databases..", sleep=1)
def get_databases(db: DoltDB):
    stmt = sa.text("SHOW DATABASES")
    with db.session.begin():
        dbs = db.session.scalars(stmt).all()

    table = Table()
    table.add_column("Databases")

    for db_name in dbs:
        if db_name in ("information_schema", "mysql"):
            continue
        table.add_row(db_name)

    print(table)


@app.command()
def create(
    ctx: typer.Context,
    db_type: DBType = DBType.config,
    hostname: str = typer.Option(..., prompt=True),
    username: str = typer.Option(..., prompt=True),
    password: str = typer.Option(..., prompt=True, hide_input=True),
    database: str = typer.Option(
        ..., prompt="Select a name for the new database", confirmation_prompt=True
    ),
):
    db = DoltDB.from_parameters(username=username, host=hostname, password=password)

    if not test_connection(db):
        raise typer.Exit()

    DB_NAME_REGEX = r"[a-zA-Z0-9_]+"

    if not re.fullmatch(DB_NAME_REGEX, database):
        print(
            Text(
                f'{database} is not a valid database name. Must match r"{DB_NAME_REGEX}"'
            )
        )

    if not create_database(db, database):
        raise typer.Exit()

    db.disconnect()
    db = DoltDB.from_parameters(
        username=username, host=hostname, password=password, database=database
    )
    db.connect()

    create_tables(db, database, db_type)


@app.command()
def show(
    ctx: typer.Context,
    hostname: str = typer.Option(..., prompt=True),
    username: str = typer.Option(..., prompt=True),
    password: str = typer.Option(..., prompt=True, hide_input=True),
):
    db = DoltDB.from_parameters(username=username, host=hostname, password=password)

    if not test_connection(db):
        raise typer.Exit()

    get_databases(db)


def migrate(db: DoltDB, current: str, target: str = qwip.__version__):
    if Version(current) > Version(target):
        print("Downgrading database versions is currently unsupported!")
        typer.Exit()

    print(f"Upgrading database version from {current} to {target}...")

    if Version(target) == Version("24.5.0"):
        updated = migrate_24_5_0(db)
    elif Version(target) == Version("24.10.0"):
        updated = migrate_24_10_0(db)
    else:
        updated = False

    if updated:
        print(f"Upgraded database to version {target}!")
        db.commit(f"Upgraded databse to version {target}.", add="all")
    else:
        print("No changes applied.")


def migrate_24_5_0(db: DoltDB):
    for table in config_tables:
        if table.name == "constraints":
            break

    with db.session.begin():
        stmt = sa.text("SHOW COLUMNS FROM constraints")
        col_names = [name for (name, *rest) in db.session.execute(stmt).all()]

        stmt = sa.text("SELECT COUNT(*) FROM constraints")
        nrows = db.session.execute(stmt).scalar_one()

    if "expression" in col_names:
        print("Database has already been upgraded!")
        return False

    if nrows:
        print("Modifying non-empty constraints table is not supported!")
        return False

    with db.session.begin():
        stmt = sa.text(
            "ALTER TABLE `constraints` "
            "DROP COLUMN `name`, "
            "RENAME COLUMN `location` TO `expression`"
        )
        db.session.execute(stmt)

    return True


def migrate_24_10_0(db: DoltDB):
    current_tables = db.tables()

    updated = False

    if "waveforms" in current_tables:
        rename_table = sa.text("RENAME TABLE `waveforms` to `operations`;")
        rename_col = sa.text(
            "ALTER TABLE `operations` RENAME COLUMN `waveform_id` to `operation_id`;"
        )

        rename_fk = sa.text(
            "ALTER TABLE `operations` DROP FOREIGN KEY `fk_waveforms_waveforms`,"
            "ADD CONSTRAINT `fk_operations_operations` FOREIGN KEY (operation_id) "
            "REFERENCES operations(operation_id) "
            "ON DELETE CASCADE ON UPDATE CASCADE;"
        )

        with db.session.begin():
            db.session.execute(rename_table)
            db.session.execute(rename_col)
            db.session.execute(rename_fk)

        updated = True

    if "waveform_locations" in current_tables:
        rename_table = sa.text(
            "RENAME TABLE `waveform_locations` to `operation_locations`;"
        )
        rename_col = sa.text(
            "ALTER TABLE `operation_locations` RENAME COLUMN `waveform_id` to "
            "`operation_id`;"
        )

        rename_tmln_fk = sa.text(
            "ALTER TABLE `operation_locations` DROP FOREIGN KEY "
            "`fk_waveform_locations_timelines`, "
            "ADD CONSTRAINT `fk_operation_locations_timelines` FOREIGN KEY "
            "(timeline_id) REFERENCES timelines(timeline_id) "
            "ON DELETE CASCADE ON UPDATE CASCADE;"
        )
        rename_wf_fk = sa.text(
            "ALTER TABLE `operation_locations` DROP FOREIGN KEY "
            "`fk_waveform_locations_waveforms`, "
            "ADD CONSTRAINT `fk_operation_locations_operations` FOREIGN KEY "
            "(operation_id) REFERENCES operations(operation_id) "
            "ON DELETE CASCADE ON UPDATE CASCADE;"
        )

        with db.session.begin():
            db.session.execute(rename_table)
            db.session.execute(rename_col)
            db.session.execute(rename_tmln_fk)
            db.session.execute(rename_wf_fk)

        updated = True

    return updated


@app.command()
def upgrade(
    ctx: typer.Context,
    hostname: str = typer.Option(..., prompt=True),
    username: str = typer.Option(..., prompt=True),
    password: str = typer.Option(..., prompt=True, hide_input=True),
    database: str = typer.Option(..., prompt=True),
):
    db = ConfigDB.from_parameters(
        host=hostname, username=username, password=password, database=database
    )

    if not test_connection(db, validate=False):
        raise typer.Exit()

    try:
        current_version = db.config.version
    except AttributeError:
        current_version = qwip.__version__

    migrate(db, current_version, target="24.10.0")


if __name__ == "__main__":
    app()
