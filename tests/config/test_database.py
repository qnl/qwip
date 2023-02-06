import pytest
import sqlalchemy as sa

import attrs
import pendulum

from qwip.settings.settings import qdefine, Settings
from qwip.config.database import (
    ConfigDB,
    Commit,
    Branch,
    SettingsFolder
)
from qwip.config.models import(
    Parameter,
    Folder,
)
from qwip.config.metadata import QWIP_DB_METADATA

from .fixtures import (
    configdb,
    models,
    session,
    reset_models
)


class TestConfigDB:
    def test_current_branch(self, configdb):
        branch = configdb.current_branch()
        assert branch.name == 'main'

    def test_create_delete_branch(self, configdb):
        configdb.branch('new')
        branch = configdb.get_branch('new')
        assert branch.name == 'new'

        configdb.branch('new', action='delete')
        assert configdb.get_branch('new') is None

    def test_checkout_branch(self, configdb):
        import time

        configdb.branch('new')
        new = configdb.get_branch('new')
        current = configdb.checkout('new')

        assert new == current
        main = configdb.get_branch('main')
        current = configdb.checkout('main')

        assert main == current
        configdb.branch('new', action='delete')

class TestFolder:
    def test_select_insert(self, session, reset_models):
        hardware = Folder(name='hardware')
        lo = Folder(name='local_oscillators', parent=hardware)
        dc = Folder(name='dc_sources', parent=hardware)

        session.add(hardware)
        session.flush()

        results = session.scalars(sa.select(Folder)).all()

        assert len(results) == 3
        assert results == [hardware, lo, dc]

    def test_select_none(self, session, reset_models):
        folders = session.scalars(sa.select(Folder)).one_or_none()
        assert folders is None

    def test_select_condition(self, session, reset_models):
        hardware = Folder(name='hardware')
        lo = Folder(name='local_oscillators', parent=hardware)
        dc = Folder(name='dc_sources', parent=hardware)
        yoko = Folder(name='yokos', parent=dc)
        qubits = Folder(name='qubits')

        session.add_all([hardware, qubits])
        session.flush()

        assert session.scalars(sa.select(Folder)).all() == [hardware, qubits, lo, dc, yoko]

        assert session.scalars(
            sa.select(Folder)
            .where(Folder.parent_id == None)
        ).all() == [hardware, qubits]
        assert session.scalars(
            sa.select(Folder)
            .where(Folder.parent_id == hardware.folder_id)
        ).all() == [lo, dc]
        
class TestParameter:
    def test_select(self, session, reset_models):
        params = session.scalars(sa.select(Parameter)).one_or_none()
        assert params is None

class TestSettingsFolder:
    def test_init(self, session, reset_models):
        settings = SettingsFolder(session=session)
        assert list(settings.keys()) == []

        a = Parameter(name='a')
        b = Parameter(name='b')
        session.add_all([a, b])
        session.flush()

        settings = SettingsFolder(session=session)
        assert list(settings.keys()) == ['a', 'b']
    