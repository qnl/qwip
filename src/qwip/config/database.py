import sqlalchemy as sa
from sqlalchemy.orm import Session
from sqlalchemy.engine import make_url, URL

import re
import pendulum
import functools
import itertools as it
from pathlib import Path
import attrs
from attrs import field
from typing_extensions import Self

import qwip
from qwip.settings.settings import Settings, qdefine, qfrozen
from qwip.flatdict import FlatDict, FlatMapping
from qwip.config.dolt import (
    DoltBranch,
    DoltCommit,
    DoltLog,
    dolt_add,
    dolt_branch,
    dolt_checkout,
    dolt_commit,
)
from qwip.config.models import Folder, Parameter, JSONTypes

SHORT_HASH_LEN = 8

@qfrozen
class Commit:
    hash: str = field(repr=lambda h: h[:SHORT_HASH_LEN])
    committer: str
    email: str
    date: pendulum.DateTime = field(repr=lambda dt: dt.in_tz('local').isoformat())
    message: str

    @classmethod
    def from_orm(cls, model: DoltCommit):
        return cls(
            hash=model.commit_hash,
            committer=model.committer,
            email=model.email,
            date=model.date,
            message=model.message
        )

@qfrozen
class Branch:
    name: str
    latest: Commit

    @classmethod
    def from_orm(cls, model: DoltBranch) -> Self:
        commit = Commit(
            hash=model.hash,
            committer=model.latest_committer,
            email=model.latest_committer_email,
            date=model.latest_commit_date,
            message=model.latest_commit_message
        )

        return cls(
            name=model.name,
            latest=commit
        )

@qfrozen
class ReadOnlyParameter:
    name: str
    value: JSONTypes | None = None
    folder: str = '/'
    timestamp: pendulum.DateTime | None = field(
        repr=lambda dt: dt.in_tz('local').isoformat() if dt else repr(dt),
        default=None
    )
    parameter_id: int = field(repr=False)

    @classmethod
    def from_orm(cls, model: Parameter) -> Self:
        return cls(
            name=model.name,
            value=model.value,
            folder=model.folder.path() if model.folder else '/',
            timestamp=model.timestamp,
            parameter_id=model.parameter_id
        )

import functools

def session_context(func):
    @functools.wraps(func)
    def decorated(inst, *args, **kwargs):
        if inst.session.in_transaction():
            return func(inst, *args, **kwargs)
        else:
            with inst.session.begin():
                return func(inst, *args, **kwargs)

    return decorated

@qdefine
class ConfigDB:
    database: str | None = None
    username: str 
    password: str
    host: str
    port: int = 3306

    engine: sa.engine.Engine | None = None
    session: sa.orm.Session | None = None

    def connect(self):
        url = URL.create(
            drivername='mysql',
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database
        )
        engine = sa.create_engine(url)
        self.engine = engine
        self.session = Session(engine, autobegin=False, expire_on_commit=False)
        
        return engine
    
    @session_context
    def current_branch(self) -> Branch:
        # with self.session.begin():
        name = self.session.scalars(sa.func.active_branch()).one()
        stmt = sa.select(DoltBranch).where(DoltBranch.name == name)

        result = self.session.scalars(stmt).one()
        branch = Branch.from_orm(result)

        return branch

    @session_context
    def get_branch(self, name: str) -> Branch | None:
        stmt = sa.select(DoltBranch).where(DoltBranch.name == name)

        result = self.session.scalars(stmt).one_or_none()

        if result is None:
            return result

        return Branch.from_orm(result)

    @session_context
    def add(
        self,
        tables: list[str] | None = None
    ) -> None:
        dolt_add(self.session, tables=tables)

    @session_context
    def branch(
        self,
        branch: str | None = None,
        other_branch: str | None = None,
        action: str = 'create',
        force: bool = False,
    ) -> None:
        dolt_branch(self.session, branch, other_branch, action, force)

    @session_context
    def checkout(self, name: str, new_branch: bool = False) -> Branch:
        dolt_checkout(self.session, name, new_branch=new_branch)

        return self.current_branch()

    @session_context
    def log(
        self,
        committer: str | None = None,
        date: pendulum.DateTime | None = None,
        date_filter: str = 'after'
    ) -> list[Commit]:
        stmt = sa.select(DoltLog).order_by(DoltLog.date)

        if committer:
            stmt = stmt.where(DoltLog.committer == committer)

        results = self.session.scalars(stmt)

        commits = [Commit.from_orm(commit) for commit in results]
        return commits

    @session_context
    def commit(
        self,
        message: str,
        add: str | None = None,
        date: pendulum.DateTime | None = None,
        author: str | None = None,
        allow_empty: bool = False
    ) -> Commit:
        dolt_commit(self.session, message, add, date, author, allow_empty)

        result = self.session.execute(
            sa.select(DoltLog).where(
                DoltLog.commit_hash == sa.func.hashof('HEAD')
            )
        ).scalar_one()

        return Commit.from_orm(result)

    @session_context
    def get_commit(
        self,
        commit_hash: str
    ) -> Commit | None:
        result = self.session.execute(
            sa.select(DoltLog).where(
                DoltLog.commit_hash == commit_hash
            )
        ).scalar_one()

        return Commit.from_orm(result)

    @classmethod
    def from_url(cls, db_url: str) -> Self:
        url = make_url(db_url)

        return cls(
            database=url.database,
            username=url.username,
            password=url.password,
            host=url.host,
            port=url.port
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
class SettingsFolder(FlatMapping):
    session: sa.orm.Session
    folder: Folder | None = field(default=None)

    @property
    def folder_id(self):
        return self.folder.folder_id if self.folder else None

    @classmethod
    def from_folder_id(cls, session: sa.orm.Session, folder_id: int) -> Self:
        folder = session.scalars(
            sa.select(Folder).where(Folder.folder_id == folder_id)
        ).one()

        return cls(session=session, folder=folder)

    @classmethod
    def from_name(cls, session: sa.orm.Session, name: str) -> Self:
        if name == '/':
            return cls(session=session)

        breadcrumbs = name.strip(cls._delim).split(cls._delim)
        stmt = sa.select(Folder).where(Folder.name == breadcrumbs[0])

        if name.startswith(cls._delim):
            stmt = stmt.where(Folder.parent_id == None)

        folders = session.scalars(stmt).all()

        if not len(folders):
            raise KeyError(f'Folder {name} does not exist.')

        for name in breadcrumbs[1:]:
            subfolders = [sub for f in folders if (sub := f.subfolders.get(name))]

            if not len(subfolders):
                raise KeyError(f'Folder {name} does not exist.')
            
            folders = subfolders
        
        if len(folders) > 1:
            raise KeyError(f'Found multiple folders with name {name}')
        
        return cls(session=session, folder=folders[0])

    def __dictrepr__(self):
        return (
            "{" +
            ", ".join([f"{repr(k)}: {getattr(v, '__dictrepr__', v.__repr__)()}" for k, v in self.items()]) +
            "}"
        )

    @session_context
    def __repr__(self) -> str:
        return f'{type(self).__name__}(path={self.path()}, contents={self.__dictrepr__()})'

    def __rich_repr__(self):
        yield 'path', self.path()
        yield 'contents', self.todict()

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
        if isinstance(existing_value, SettingsFolder):
            try:
                existing_value.update(**value)
            except Exception as e:
                raise KeyError(f"'{name}' is a folder.") from e
        else:
            param = type(self)._get_parameters_from_db(
                self.session,
                name,
                self.folder_id
            ).one()
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
                raise AttributeError(f'Folder {folder_name} does not exist in {self}.')

        if param_name in folderproxy:
            raise AttributeError(f'Parameter {param_name} already exists in {self}.')

        param = Parameter(
            name=param_name,
            value=qwip.converter.unstructure(value),
            folder=folderproxy.folder
        )

        self.session.add(param)
        self.session.flush()
        return param.value

    @session_context
    def create_folder(self, name, parents=True, exist_ok=False):
        folder_list = get_folder_list(name.replace(self._delim, '/'))

        if folder_list[0] == Path('/') and self.folder is not None:
            self_path = self.path()
            self_folder_list = get_folder_list(self_path.replace(self._delim, '/'))

            if folder_list[:len(self_folder_list)] != self_folder_list:
                raise ValueError(
                    f'Folder path {name} is not a subfolder of {self_path}.'
                )
            
            folder_list = folder_list[len(self_folder_list):]
        else:
            folder_list = folder_list[1:]

        folders_to_add = []
        folder = self.folder
        for i, sub in enumerate(folder_list):
            if folder is not None and sub in folder.subfolders:
                folder = folder[sub]
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
                    f'Cannot create folder {name} that exists already.'
                )

        self.session.add_all(folders_to_add)
        self.session.flush()
        
        return type(self)(session=self.session, folder=folder)

    @session_context
    def path(self) -> str:
        if self.folder is None:
            return self._delim

        return self.folder.path().replace('/', self._delim)

    @session_context
    def __proxy_getitem__(self, name):
        # Look for subfolder with name first
        if self.folder:
            subfolder = self.folder.subfolders.get(name)
        else:
            subfolder = type(self)._get_folders_from_db(
                self.session, name, None
            ).one_or_none()
        
        if subfolder:
            return type(self)(session=self.session, folder=subfolder)

        # Then look for parameter with name
        if self.folder:
            parameter = self.folder.parameters.get(name)
        else:
            parameter = type(self)._get_parameters_from_db(self.session, name, None).one_or_none()
        
        if parameter:
            return parameter.value

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @session_context
    def _get_parameter(self, name):
        subkeys = self.rsplit(self.strip(name), maxsplit=1)

        last_folder = self.folder if len(subkeys) == 1 else self[subkeys[0]].folder
        
        if last_folder:
            db_param = last_folder.parameters.get(subkeys[-1])
        else:
            db_param = type(self)._get_parameters_from_db(
                self.session,
                subkeys[-1],
                None
            ).one_or_none()

        return db_param

    @session_context
    def get_parameter(self, name):
        db_param = self._get_parameter(name)

        return ReadOnlyParameter.from_orm(model=db_param)

    @session_context
    def get_parameter_history(self, name):
        db_param = self._get_parameter(name)

        history = db_param.history(self.session)

        return [(cmt, ReadOnlyParameter.from_orm(model=p)) for cmt, p in history]

    @session_context
    def __proxy_delitem__(self, name):
        obj = self.__proxy_getitem__(name)

        if isinstance(obj, SettingsFolder):
            self.session.delete(obj.folder)
            return 

        if self.folder:
            parameter = self.folder.parameters.get(name)
        else:
            parameter = type(self)._get_parameters_from_db(self.session, name, None).one_or_none()
       
        self.session.delete(parameter)

    @session_context
    def search(self, name: str = None, sort = True):
        return self.search_folders(name, sort) + self.search_parameters(name, sort)
    
    @session_context
    def search_folders(self, name: str = None, sort=True):
        """Searches for all subfolders.

        See https://www.mysqltutorial.org/mysql-adjacency-list-tree/

        Args:
            name: 
        """
        folder_path = (
            sa.select(Folder.folder_id, Folder.name, Folder.name.label('path'))
            .where(Folder.parent_id == self.folder_id)
            .cte(name='folder_path', recursive=True)
        )

        fp = sa.orm.aliased(folder_path, name='fp')
        f = sa.orm.aliased(Folder, name='f')

        subquery = folder_path.union_all(
            sa.select(f.folder_id, f.name, sa.func.concat(fp.c.path, '/', f.name))
            .select_from(
                sa.join(fp, f, fp.c.folder_id == f.parent_id)
            )
        )

        stmt = sa.select(subquery)
        
        if name: stmt = stmt.where(subquery.c.name == name)
        if sort: stmt = stmt.order_by(subquery.c.path)

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
            sa.select(Folder.folder_id, Folder.name, Folder.name.label('path'))
            .where(Folder.parent_id == self.folder_id)
            .cte(name='folder_path', recursive=True)
        )

        fp = sa.orm.aliased(folder_path, name='fp')
        f = sa.orm.aliased(Folder, name='f')

        subquery = folder_path.union_all(
            sa.select(f.folder_id, f.name, sa.func.concat(fp.c.path, '/', f.name))
            .select_from(
                sa.join(fp, f, fp.c.folder_id == f.parent_id)
            )
        )

        # This first statement gets all Parameters in nested subfolders
        stmt = (
            sa.select(Parameter)
            .select_from(
                sa.join(subquery, Parameter, subquery.c.folder_id == Parameter.folder_id)
            )
        )
        # This second statement gets all Parameters in this folder
        non_nested = sa.select(Parameter).where(Parameter.folder_id == self.folder_id)
        
        if name: 
            stmt = stmt.where(Parameter.name == name)
            non_nested = non_nested.where(Parameter.name == name)
        if sort: 
            stmt = stmt.order_by(subquery.c.path)

        all_parameters = it.chain(self.session.scalars(stmt), self.session.scalars(non_nested))

        return [ReadOnlyParameter.from_orm(p) for p in all_parameters]

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
            num_root_folders =  self.session.scalar(
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


def unstructure_SettingsFolder(settings: SettingsFolder) -> dict:
    return {k: qwip.converter.unstructure(v) for k, v in settings.items()}

qwip.converter.register_unstructure_hook(
    SettingsFolder,
    unstructure_SettingsFolder
)

@qdefine
class ValidatedSettingsFolder(SettingsFolder):
    _schema: type | None = field()

    @_schema.validator
    def _schema_validator(self, attr, value):
        if not attrs.has(value):
            raise ValueError(f"'_schema' must be an attrs clas, got {value}")

    # def __proxy_setitem__(self, key, val):
    @classmethod
    def from_settings_folder(cls, settings, schema) -> Self:
        return cls(
            session=settings.session,
            folder=settings.folder,
            schema=schema
        )

    def __proxy_setitem__(self, name, value):
        # First we check if we're trying to write to an actual attribute.
        if name not in self:
            raise KeyError(
                f"'{name}' does not exist. Use create_parameter to or create_folder to "
                f"create a new parameter or folder."
            )

        field = getattr(attrs.fields(self._schema), name)

        convert = field.converter
        validate = field.validator

        new_value = convert(value)
        validate(self, field, new_value)

        super().__proxy_setitem__(name, new_value)

    def __proxy_getitem__(self, name):
        value = super().__proxy_getitem__(name)

        if isinstance(value, SettingsFolder):
            schema = getattr(attrs.fields(self._schema), name).type
            value = type(self).from_settings_folder(value, schema=schema)
        
        return value