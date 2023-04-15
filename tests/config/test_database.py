import attrs
import pendulum
import pytest
import sqlalchemy as sa

from qwip.config.database import Branch, Commit, ConfigDB, ConfigFolder, DoltDB
from qwip.config.metadata import QWIP_DB_METADATA
from qwip.config.models import Folder, Parameter
from qwip.settings.settings import Settings, qdefine


class TestDoltDB:
    def test_current_branch(self, database):
        branch = database.current_branch()
        assert branch.name == "main"

    def test_create_delete_branch(self, database):
        database.branch("new")
        branch = database.get_branch("new")
        assert branch.name == "new"

        database.branch("new", action="delete", force=True)
        assert database.get_branch("new") is None

    def test_checkout_branch(self, database):
        database.branch("new")
        new = database.get_branch("new")
        current = database.checkout("new")

        assert new == current
        main = database.get_branch("main")
        current = database.checkout("main")

        assert main == current
        database.branch("new", action="delete", force=True)


# class TestConfigFolder:
#     def test_init(self, session, reset_models):
#         settings = ConfigFolder(session=session)
#         assert list(settings.keys()) == []

#         a = Parameter(name="a")
#         b = Parameter(name="b")
#         session.add_all([a, b])
#         session.flush()

#         settings = ConfigFolder(session=session)
#         assert list(settings.keys()) == ["a", "b"]
