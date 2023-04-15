import attrs
import pendulum
import pytest
import sqlalchemy as sa

from qwip.config.database import Branch, Commit, ConfigDB, ConfigFolder, DoltDB
from qwip.config.metadata import QWIP_DB_METADATA
from qwip.config.models import Folder, Parameter
from qwip.settings.settings import Settings, qdefine


class TestDoltDB:
    def test_current_branch(self, doltdb):
        branch = doltdb.current_branch()
        assert branch.name == "main"

    def test_create_delete_branch(self, doltdb):
        doltdb.branch("new")
        branch = doltdb.get_branch("new")
        assert branch.name == "new"

        doltdb.branch("new", action="delete")
        assert doltdb.get_branch("new") is None

    def test_checkout_branch(self, doltdb):
        doltdb.branch("new")
        new = doltdb.get_branch("new")
        current = doltdb.checkout("new")

        assert new == current
        main = doltdb.get_branch("main")
        current = doltdb.checkout("main")

        assert main == current
        doltdb.branch("new", action="delete")


class TestConfigFolder:
    def test_init(self, session, reset_models):
        settings = ConfigFolder(session=session)
        assert list(settings.keys()) == []

        a = Parameter(name="a")
        b = Parameter(name="b")
        session.add_all([a, b])
        session.flush()

        settings = ConfigFolder(session=session)
        assert list(settings.keys()) == ["a", "b"]
