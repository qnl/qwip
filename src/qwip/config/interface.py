import functools
import itertools as it
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, get_args

import attrs
import pandas as pd
import pendulum
import sqlalchemy as sa
from attrs import field
from loguru import logger
from sqlalchemy.engine import URL, make_url
from typing_extensions import Self

import qwip
from qwip.attrs import qdefine, qfrozen
from qwip.config.models import (
    Folder,
    JSONTypes,
    Parameter,
    TimelineModel,
    config_tables,
)
from qwip.database.database import SHORT_HASH_LEN, Database, DoltDB, session_context
from qwip.database.metadata import QWIP_DB_METADATA
from qwip.flatdict import FlatDict, FlatMapping
from qwip.sequencer.timeline import Timeline
from qwip.typing import issubtype

try:
    from IPython.display import JSON, display
except ImportError:
    logger.info("Unable to import IPython.")


@qfrozen
class ReadOnlyParameter:
    name: str
    value: JSONTypes | None = None
    folder: str = "/"
    timestamp: pendulum.DateTime | None = field(
        repr=lambda dt: dt.in_tz("local").isoformat() if dt else repr(dt), default=None
    )
    parameter_id: int | None = field(repr=False)

    @classmethod
    def from_orm(cls, model: Parameter) -> Self:
        return cls(
            name=model.name,
            value=model.value,
            folder=model.folder.path() if model.folder else "/",
            timestamp=model.timestamp,
            parameter_id=model.parameter_id,
        )


def get_folder_list(name: str) -> list[Path | str]:
    path = Path(name)

    folders = []
    while path.parent != path:
        folders.append(path.name)
        path = path.parent

    folders.append(path)

    return folders[::-1]


@qdefine(repr=False)
class ConfigFolder(FlatMapping):
    __orig_class__: type = field(init=False, metadata=dict(validate=False, db=False))
    session: sa.orm.Session = field(metadata=dict(db=False))
    folder: Folder | None = field(default=None, metadata=dict(db=False))

    @property
    @session_context
    def folder_id(self):
        return self.folder.folder_id if self.folder else None

    @property
    def subfolder_class(self):
        try:
            self_cls = object.__getattribute__(self, "__orig_class__")
            match get_args(self_cls):
                case (kt, vt):
                    if issubtype(vt, ConfigFolder):
                        return vt
                case _:
                    ...
            return vt

        except AttributeError:
            ...

        return ConfigFolder

    @classmethod
    def from_folder_id(cls, session: sa.orm.Session, folder_id: int) -> Self:
        folder = session.scalars(
            sa.select(Folder).where(Folder.folder_id == folder_id)
        ).one()

        return cls(session=session, folder=folder)

    @classmethod
    def from_name(cls, session: sa.orm.Session, name: str) -> Self:
        if name == "/":
            return cls(session=session)

        breadcrumbs = name.strip(cls._delim).split(cls._delim)
        stmt = sa.select(Folder).where(Folder.name == breadcrumbs[0])

        if name.startswith(cls._delim):
            stmt = stmt.where(Folder.parent_id == None)

        folders = session.scalars(stmt).all()

        if not len(folders):
            raise KeyError(f"Folder {name} does not exist.")

        for name in breadcrumbs[1:]:
            subfolders = [sub for f in folders if (sub := f.subfolders.get(name))]

            if not len(subfolders):
                raise KeyError(f"Folder {name} does not exist.")

            folders = subfolders

        if len(folders) > 1:
            raise KeyError(f"Found multiple folders with name {name}")

        return cls(session=session, folder=folders[0])

    def __dictrepr__(self):
        return (
            "{"
            + ", ".join(
                [
                    f"{repr(k)}: {getattr(v, '__dictrepr__', v.__repr__)()}"
                    for k, v in self.items()
                ]
            )
            + "}"
        )

    @session_context
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(path={self.path()}, contents={self.__dictrepr__()})"
        )

    def _ipython_display_(self):
        with self.session.begin():
            root = f"{type(self).__name__}(path={self.path()})"
            json = JSON(self.todict(), root=root)

        display(json)

    def __rich_repr__(self):
        yield "path", self.path()
        yield "contents", self.todict()

    @staticmethod
    def _get_folders_from_db(session, name=None, parent_id=None):
        stmt = sa.select(Folder)

        if parent_id is not ...:
            stmt = stmt.where(Folder.parent_id == parent_id)

        if name is not None:
            stmt = stmt.where(Folder.name == name)

        return session.scalars(stmt)

    @staticmethod
    def _get_parameters_from_db(session, name=None, folder_id=None):
        stmt = sa.select(Parameter)

        if folder_id is not ...:
            stmt = stmt.where(Parameter.folder_id == folder_id)

        if name is not None:
            stmt = stmt.where(Parameter.name == name)

        return session.scalars(stmt)

    def _get_mapping_type(self, key: str) -> type:
        return FlatDict

    @session_context
    def __proxy_setitem__(self, name, value):
        # First we check if we're trying to write to an actual attribute.
        if name not in self:
            raise KeyError(
                f"'{name}' does not exist. Use create_parameter to or create_folder to "
                f"create a new parameter or folder."
            )

        existing_value = self[name]
        value = qwip.converter.unstructure(value)
        if isinstance(existing_value, ConfigFolder):
            try:
                existing_value.update(**value)
            except ValueError as e:
                raise KeyError(f"'{name}' is a folder.") from e
        else:
            param = (
                type(self)
                ._get_parameters_from_db(self.session, name, self.folder_id)
                .one()
            )
            param.value = value

        self.session.flush()

    @session_context
    def create_parameter(self, name, value=None):
        splitname = name.rsplit(self._delim, maxsplit=1)

        if len(splitname) == 1:
            folder_name = None
            param_name = splitname[0]
        else:
            folder_name, param_name = splitname

        folderproxy = self
        if folder_name:
            folderproxy = self.get(folder_name)
            if folderproxy is None:
                raise AttributeError(f"Folder {folder_name} does not exist in {self}.")

        if param_name in folderproxy:
            raise AttributeError(f"Parameter {param_name} already exists in {self}.")

        param = Parameter(
            name=param_name,
            value=qwip.converter.unstructure(value),
            folder=folderproxy.folder,
        )

        self.session.add(param)
        self.session.flush()
        return param.value

    @session_context
    def create_folder(self, name, parents=True, exist_ok=False):
        folder_list = get_folder_list(name.replace(self._delim, "/"))

        if folder_list[0] == Path("/") and self.folder is not None:
            self_path = self.path()
            self_folder_list = get_folder_list(self_path.replace(self._delim, "/"))

            if folder_list[: len(self_folder_list)] != self_folder_list:
                raise ValueError(
                    f"Folder path {name} is not a subfolder of {self_path}."
                )

            folder_list = folder_list[len(self_folder_list) :]
        else:
            folder_list = folder_list[1:]

        folders_to_add = []
        folder = self.folder
        for i, sub in enumerate(folder_list):
            if folder is not None and sub in folder.subfolders:
                folder = folder.subfolders[sub]
            elif folder is None and sub in self:
                folder = self[sub].folder
            elif i == len(folder_list) - 1:
                folder = Folder(name=sub, parent=folder)
                folders_to_add.append(folder)
                break
            elif parents:
                folder = Folder(name=sub, parent=folder)
                folders_to_add.append(folder)
            else:
                raise FileNotFoundError(
                    f"Path '{name}' does not exist. Use parents=True to create all parent folders."
                )
        else:
            if not exist_ok:
                raise FileExistsError(
                    f"Cannot create folder {name} that exists already."
                )

        self.session.add_all(folders_to_add)
        self.session.flush()

        return type(self)(session=self.session, folder=folder)

    @session_context
    def path(self) -> str:
        if self.folder is None:
            return self._delim

        return self.folder.path().replace("/", self._delim)

    @session_context
    def folder_name(self) -> str:
        if self.folder is None:
            return None

        return self.folder.name

    @session_context
    def parent(self) -> Self | None:
        if self.folder is None:
            return None
        elif self.folder.parent is None:
            return ConfigFolder.from_name(self.session, "/")
        else:
            return ConfigFolder.from_folder_id(
                self.session, self.folder.parent.folder_id
            )

    @session_context
    def rename(self, name: str) -> str:
        """Renames a folder.

        The root folder `"/"` cannot be renamed.

        Args:
            name: The new name for the folder.

        Returns:
            The new path for the folder.

        Raises:
            ValueError: If the folder being renamed is the root folder or if the new
                path already exists.
        """
        if self.folder is None:
            raise ValueError(f"Cannot rename root directory: {self.path()}")

        parent = self.parent()
        if name in parent:
            existing_path = parent[name].path()
            raise ValueError(
                f"Cannot rename {self.path()} to {name} because {existing_path} already"
                f" exists."
            )

        self.folder.name = name
        return self.path()

    @session_context
    def __proxy_getitem__(self, name):
        # Look for subfolder with name first
        if self.folder:
            subfolder = self.folder.subfolders.get(name)
        else:
            subfolder = (
                type(self)._get_folders_from_db(self.session, name, None).one_or_none()
            )

        if subfolder:
            subfolder_cls = object.__getattribute__(self, "subfolder_class")
            return subfolder_cls(session=self.session, folder=subfolder)

        # Then look for parameter with name
        if self.folder:
            parameter = self.folder.parameters.get(name)
        else:
            parameter = (
                type(self)
                ._get_parameters_from_db(self.session, name, None)
                .one_or_none()
            )

        if parameter:
            return parameter.value

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    @session_context
    def _get_parameter(self, name):
        subkeys = self.rsplit(self.strip(name), maxsplit=1)

        last_folder = self.folder if len(subkeys) == 1 else self[subkeys[0]].folder

        if last_folder:
            db_param = last_folder.parameters.get(subkeys[-1])
        else:
            db_param = (
                type(self)
                ._get_parameters_from_db(self.session, subkeys[-1], None)
                .one_or_none()
            )

        return db_param

    @session_context
    def get_parameter(self, name):
        db_param = self._get_parameter(name)

        if db_param is None:
            return db_param

        return db_param

    @session_context
    def get_parameter_history(self, name):
        db_param = self._get_parameter(name)

        return db_param.history(self.session)

    @session_context
    def __proxy_delitem__(self, name):
        obj = self.__proxy_getitem__(name)

        if isinstance(obj, ConfigFolder):
            self.session.delete(obj.folder)
            return

        if self.folder:
            parameter = self.folder.parameters.get(name)
        else:
            parameter = (
                type(self)
                ._get_parameters_from_db(self.session, name, None)
                .one_or_none()
            )

        self.session.delete(parameter)

    @session_context
    def search(self, name: str = None, sort=True):
        return self.search_folders(name, sort) + self.search_parameters(name, sort)

    @session_context
    def search_folders(self, name: str = None, sort=True):
        """Searches for all subfolders.

        See https://www.mysqltutorial.org/mysql-adjacency-list-tree/

        Args:
            name:
        """
        folder_path = (
            sa.select(Folder.folder_id, Folder.name, Folder.name.label("path"))
            .where(Folder.parent_id == self.folder_id)
            .cte(name="folder_path", recursive=True)
        )

        fp = sa.orm.aliased(folder_path, name="fp")
        f = sa.orm.aliased(Folder, name="f")

        subquery = folder_path.union_all(
            sa.select(
                f.folder_id, f.name, sa.func.concat(fp.c.path, "/", f.name)
            ).select_from(sa.join(fp, f, fp.c.folder_id == f.parent_id))
        )

        stmt = sa.select(subquery)

        if name:
            stmt = stmt.where(subquery.c.name == name)
        if sort:
            stmt = stmt.order_by(subquery.c.path)

        return [
            type(self).from_folder_id(session=self.session, folder_id=fid)
            for fid in self.session.scalars(stmt)
        ]

    @session_context
    def search_parameters(self, name: str = None, sort=True):
        """Searches for all parameters.

        See https://www.mysqltutorial.org/mysql-adjacency-list-tree/

        Args:
            name:
        """
        folder_path = (
            sa.select(Folder.folder_id, Folder.name, Folder.name.label("path"))
            .where(Folder.parent_id == self.folder_id)
            .cte(name="folder_path", recursive=True)
        )

        fp = sa.orm.aliased(folder_path, name="fp")
        f = sa.orm.aliased(Folder, name="f")

        subquery = folder_path.union_all(
            sa.select(
                f.folder_id, f.name, sa.func.concat(fp.c.path, "/", f.name)
            ).select_from(sa.join(fp, f, fp.c.folder_id == f.parent_id))
        )

        # This first statement gets all Parameters in nested subfolders
        stmt = sa.select(Parameter).select_from(
            sa.join(subquery, Parameter, subquery.c.folder_id == Parameter.folder_id)
        )
        # This second statement gets all Parameters in this folder
        non_nested = sa.select(Parameter).where(Parameter.folder_id == self.folder_id)

        if name:
            stmt = stmt.where(Parameter.name == name)
            non_nested = non_nested.where(Parameter.name == name)
        if sort:
            stmt = stmt.order_by(subquery.c.path)

        all_parameters = it.chain(
            self.session.scalars(stmt), self.session.scalars(non_nested)
        )

        return all_parameters

    def create_all(self, **kwargs):
        if not kwargs:
            return

        generic = getattr(self, "__orig_class__", None)

        match get_args(generic):
            case (_, value_class):
                ...
            case _:
                value_class = None

        for key, value in kwargs.items():
            if key not in self:
                if issubtype(value_class, ConfigFolder):
                    self.create_folder(key)
                    self[key].create_all(**value)
                else:
                    self.create_parameter(key, value)
            else:
                self[key] = value

    @session_context
    def __iter__(self):
        if self.folder:
            subfolders = self.folder.subfolders
            parameters = self.folder.parameters
        else:
            subfolders = (
                f.name for f in type(self)._get_folders_from_db(self.session).all()
            )
            parameters = (
                p.name for p in type(self)._get_parameters_from_db(self.session).all()
            )

        return it.chain(subfolders, parameters)

    @session_context
    def __len__(self):
        if self.folder:
            return len(self.folder.subfolders) + len(self.folder.parameters)
        else:
            num_root_folders = self.session.scalar(
                sa.select(sa.func.count())
                .select_from(Folder)
                .where(Folder.parent_id == None)
            )

            num_parameters = self.session.scalar(
                sa.select(sa.func.count())
                .select_from(Parameter)
                .where(Parameter.folder_id == None)
            )

            return num_root_folders + num_parameters


def unstructure_ConfigFolder(settings: ConfigFolder) -> dict:
    return {k: qwip.converter.unstructure(v) for k, v in settings.items()}


qwip.converter.register_unstructure_hook(ConfigFolder, unstructure_ConfigFolder)


@qdefine(repr=False)
class ValidatedConfigFolder(ConfigFolder):
    def __getattribute__(self, name):
        try:
            field = getattr(attrs.fields(type(self)), name)
            if field.metadata.get("db", True):
                return self.__proxy_getitem__(name)
        except AttributeError:
            ...

        return super().__getattribute__(name)

    def __proxy_setitem__(self, name, value):
        # First we check if we're trying to write to an actual attribute.
        if name not in self:
            raise KeyError(
                f"'{name}' does not exist. Use create_parameter or create_folder to "
                f"create a new parameter or folder."
            )

        if issubtype(getattr(type(self).fields(), name).type, ConfigFolder):
            try:
                self[name].update(value)
                return
            except ValueError:
                pass

        setattr(self, name, value)

    def __proxy_getitem__(self, name):
        value = super().__proxy_getitem__(name)

        if isinstance(value, ConfigFolder):
            return type(self).get_key_type(name)(
                session=value.session, folder=value.folder
            )

        field = getattr(type(self).fields(), name, None)
        if field and field.converter:
            value = field.converter(value)

        return value

    @classmethod
    def get_key_type(cls, key):
        subkeys = key.split(cls._delim)

        fieldtype = cls

        for subkey in subkeys:
            if issubtype(fieldtype, ValidatedConfigFolder):
                fieldtype = getattr(fieldtype.fields(), subkey, ...)

                fieldtype = getattr(fieldtype, "type", ...)

            elif issubtype(fieldtype, Mapping):
                match get_args(fieldtype):
                    case (kt, vt):
                        fieldtype = vt
                    case _:
                        raise KeyError(f"'{key}' has no specified type.")

            if fieldtype is ...:
                raise KeyError(f"'{key}' has no specified type.")

        return fieldtype

    @classmethod
    @functools.cache
    def fields(cls):
        return attrs.fields(cls)

    def create_all(self, **dict_keys):
        for field in type(self).fields():
            name = field.name
            if not field.metadata.get("db", True):
                continue

            subfolder_keys = dict_keys.get(name, {})

            if issubtype(field.type, ConfigFolder):
                if name not in self:
                    self.create_folder(name)

                self[name].create_all(**subfolder_keys)
            else:
                if name in self:
                    if name in dict_keys:
                        self[name] = dict_keys[name]
                    continue

                match field.default:
                    case attrs.NOTHING:
                        default = None
                    case attrs.Factory(factory=f, takes_self=takes_self):
                        default = f(self) if takes_self else f()
                    case default:
                        ...

                self.create_parameter(name, value=dict_keys.get(name, default))


def set_in_db(inst, attr, value):
    super(ValidatedConfigFolder, inst).__proxy_setitem__(attr.name, value)
    return value


configschema = functools.partial(
    qdefine,
    init=False,
    repr=False,
    on_setattr=[attrs.setters.convert, attrs.setters.validate, set_in_db],
)


@qdefine
class PulsesFolder:
    session: sa.orm.Session

    def _get_timeline_model(self, name: str) -> TimelineModel:
        se_model = self.session.scalar(
            sa.select(TimelineModel).where(TimelineModel.name == name)
        )

        return se_model

    @session_context
    def keys(self) -> tuple[str]:
        keys = self.session.scalars(sa.select(TimelineModel.name))
        return tuple(k for k in keys)

    @session_context
    def add(self, name: str, tmln: Timeline):
        if name in self.keys():
            raise ValueError(
                f"Timeline '{name}' already exists. Use `update` to modify "
                f"an existing Timeline in the database."
            )
        tmln_model = TimelineModel.from_timeline(tmln, name)
        self.session.add(tmln_model)
        self.session.flush()

    @session_context
    def get(self, name: str) -> Timeline | None:
        tmln_model = self._get_timeline_model(name)

        if tmln_model:
            return tmln_model.to_timeline()

        return None

    @session_context
    def update(self, name: str, new_tmln: Timeline) -> None:
        tmln_model = self._get_timeline_model(name)

        if tmln_model is None:
            self.add(name, new_tmln)
            return

        old_tmln = tmln_model.to_timeline()

        if new_tmln == old_tmln:
            return

        self.session.delete(tmln_model)
        self.add(name, new_tmln)

    @session_context
    def delete(self, name: str) -> None:
        tmln_model = self._get_timeline_model(name)

        if tmln_model is None:
            raise KeyError(f"Timeline '{name}' not found.")

        self.session.delete(tmln_model)
        self.session.flush()

    @session_context
    def search(self, key: str) -> dict[str, Timeline]:
        """Searches for pulse timelines by name."""

        results = self.session.scalars(
            sa.select(TimelineModel).where(TimelineModel.name.icontains(key))
        )

        return {t.name: t.to_timeline() for t in results}

    def __getitem__(self, key: str) -> Timeline:
        tmln = self.get(key)

        if tmln:
            return tmln

        raise KeyError(f"'{key}' does not exist in the Timeline table.")

    def __contains__(self, key: str) -> bool:
        return key in self.keys()


@qdefine
class OfflineConfigDB(Database):
    """An interface to a local configuration database backend.

    The OfflineConfigDB can be used with a SQLite database backend, with the caveat that
    no database version control features are available. This is primarily used for
    testing and development purposes, but can also be used as a backup in the event
    that the dolt database server is down.

    Attributes:
        url: The database connection url.
        engine: The SQLAlchemy engine that maintains the database connection.
        session: The database session associated with the engine.
    """

    schema: type[ConfigFolder] = ConfigFolder

    config: ConfigFolder | None = field(init=False, default=None)
    pulses: PulsesFolder | None = field(init=False, default=None)

    def init_config(self):
        self.config = self.schema.from_name(self.session, "/")

        try:
            if (c := self.config["version"]) != (q := qwip.qsettings["version"]):
                logger.warning(
                    f"Config database was last used with QWiP version {c} which "
                    f"differs from current QWiP version {q}! Call "
                    f"ConfigDB.update_db_version() to update database to current "
                    f"version, or revert QWiP to {c}."
                )
            if (c := self.config["qwip_commit"]) != (q := qwip.qsettings["src/commit"]):
                logger.warning(
                    f"Config database was last used with QWiP source commit "
                    f"{c[:SHORT_HASH_LEN]} which differs from current source commit "
                    f"{q[:SHORT_HASH_LEN]}! Call ConfigDB.update_db_version() to "
                    f" update database to current version or revert source to "
                    f"{c[:SHORT_HASH_LEN]}."
                )
        except KeyError:
            ...

    def init_pulses(self):
        self.pulses = PulsesFolder(session=self.session)

    def connect(self, test: bool = True, timeout: int = 2):
        engine = super().connect(test=test, timeout=timeout)

        reflected_tables = self.tables()
        expected_tables = set(t.name for t in config_tables)

        if reflected_tables != expected_tables:
            version = qwip.qsettings.version
            raise ValueError(
                f"Database {self.url} has tables {reflected_tables} that do not match "
                f"the expected tables {expected_tables} for version {version}. Use the"
                f"'database-setup.py' script to upgrade or downgrade the database."
            )

        self.init_config()
        self.init_pulses()

        return engine

    def update_db_version(self):
        self.config["version"] = qwip.qsettings["version"]
        self.config["qwip_commit"] = (
            qwip.qsettings["src/commit"] or self.config["qwip_commit"]
        )

    def __getitem__(self, name):
        if self.config is None:
            raise ValueError("No settings folder object.")

        return self.config[name]

    @session_context
    def add_pulse(
        self,
        name: str,
        targets: tuple[str],
        pulse_key: str = None,
        tmln: Timeline | None = None,
        include_var: Callable[[str], bool] = lambda v: True,
    ) -> ConfigFolder:
        """Adds a pulse to the config database.

        A pulse is stored as a timeline along with some metadata in the configuration
        table. To faciliate tracking of calibration parameters, the pulse timeline can
        act as a pulse "prototype" with string parameters whose concrete values are
        referenced in the configuration database.

        Args:
            name: The name of the pulse to add.
            targets: The targets on which this pulse acts.
            pulse_key: The name of the timeline that this pulse refers to.
                If `None`, the pulse key is assumed to be the same as `name`.
            tmln: The pulse prototype to add to the database. If `None`, the pulse key
                must refer to an existing timeline in the database.

        Returns:
            The pulse configuration.
        """
        if name in self.config["pulses"]:
            raise KeyError(f"Pulse '{name}' already exists.")

        if extra := set(targets) - set(self.config["targets"]):
            raise ValueError(f"Targets {extra} are not registered in '/targets/'.")

        pulse_key = pulse_key or name

        if tmln is None:
            tmln = self.pulses[pulse_key]
        else:
            self.pulses.add(pulse_key, tmln)

        parameters = {
            name: dict(
                variables={v: v for v in tmln.variables() if include_var(v)},
                targets=targets,
                pulse_key=pulse_key,
            )
        }
        self.config["pulses"].create_all(**parameters)

        return self.config["pulses"][name]

    def load_pulse(
        self,
        name: str,
        variables: dict[str, str | float | int] = {},
        rename_func: Callable[
            [str, "PulsesSchema"], str
        ] = lambda v, pm: f"{pm.folder_name()}.{v}",
    ) -> Timeline:
        """Loads a Timeline from the database.

        This method loads the pulse prototype specified by the pulse metadata and
        replaces all variables specified in the pulse metadata. These variables
        can be overridden by passing in a variables dictionary.

        Additionally all variables that remain unchanged are automatically renamed
        to avoid collisions. This can be customized by specifying a `rename_func`.

        Args:
            name: The name of the pulse to load
            variables: A dictionary mapping variables to values. These will override
                any variables in the pulse metadata. Values can be renamed variables
                in addition to concrete values.
            rename_func: A callable that takes in a variable `str` and the pulse
                metadata `PulsesSchema` and returns the renamed variable `str`.

        Returns:
            A `Timeline` representing the pulse.

        Raises:
            `ValueError`: If any variables are specified but not present in the loaded
                `Timeline` from the database.

        """
        pulse_metadata = self.config["pulses"][name]
        tmln = self.pulses[pulse_metadata["pulse_key"]]

        to_replace = pulse_metadata["variables"].todict() | variables

        if extra := set(to_replace) - tmln.variables():
            raise ValueError(
                f"Found extra variables {extra} when loading pulse '{name}'"
            )

        def replace(v):
            if v not in to_replace:
                return v
            elif v == to_replace[v]:
                return rename_func(v, pulse_metadata)
            else:
                return to_replace[v]

        tmln.rename_variables(replace)

        return tmln

    def pulse_parameters(self, names: str | list[str] = r".*") -> pd.DataFrame:
        """Returns a dataframe of pulse parameters in table form.

        Args:
            names: A list of pulse names to include in the table. If a list of names is
                passed in, only these pulses will be included. If a names is a string,
                it will be used as a regex to filter out pulses whose names don't match.

        Returns:
            A dataframe indexed by pulse names, with all possible pulse parameters as
            columns. NaN values indicate that the given parameter does not exist for the
            pulse.
        """
        parameters = {}

        match names:
            case str():
                exclude = lambda s: not bool(re.match(names, s))
            case list():
                exclude = lambda s: s not in names
            case _:
                raise ValueError(
                    f"'names' must be a str or a list of str, got {names} that is {type(names)}"
                )

        for name, pulse in self.config.pulses.items():
            if exclude(name):
                continue

            parameters[(name, "pulse_key")] = pulse.pulse_key
            parameters |= {
                (name, param): value for param, value in pulse.variables.items()
            }

        return pd.Series(parameters.values(), index=parameters.keys()).unstack()


@qdefine
class ConfigDB(OfflineConfigDB, DoltDB):
    """An interface to a configuration database backend using Dolt.

    Attributes:
        url: The database connection url.
        engine: The SQLAlchemy engine that maintains the database connection.
        session: The database session associated with the engine.
    """

    url: URL = field(
        converter=lambda s: make_url(s) if isinstance(s, str) else s,
    )

    @url.validator
    def _validate_url(self, attribute, value):
        if value.get_backend_name() != "mysql":
            raise ValueError("Dolt database must use mysql driver")

        if value.database is None:
            raise ValueError("Database name must be provided for ConfigDB.")


__all__ = [
    "ReadOnlyParameter",
    "ConfigDB",
    "OfflineConfigDB",
    "ConfigFolder",
    "ValidatedConfigFolder",
    "configschema",
    "PulsesFolder",
]
