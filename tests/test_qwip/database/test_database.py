import warnings

import pytest
import sqlalchemy as sa

from qwip.database.database import Status
from qwip.testing import ignore_order


class TestDatabase:
    def test_tables(self, database, models):
        assert database.tables() == {
            "folders",
            "parameters",
            "waveforms",
            "waveform_locations",
            "constraints",
            "sequence_elements",
            "datasets",
            "assets",
        }


@pytest.mark.usefixtures("skip_dolt")
class TestDoltDB:
    def test_current_branch(self, database):
        branch = database.current_branch()
        assert branch.name == "main"

    def test_create_delete_branch(self, database):
        database.branch("new")
        branch = database.get_branch("new")
        assert branch.name == "new"

        try:
            database.branch("new", action="delete")
        except sa.exc.OperationalError:
            warnings.warn("Forcing branch deletion")
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

        try:
            database.branch("new", action="delete")
        except sa.exc.OperationalError:
            warnings.warn("Forcing branch deletion")
            database.branch("new", action="delete", force=True)

    def test_status(self, database, models):
        assert database.status() == ignore_order(
            [
                Status(table=t, staged=False, status="new table")
                for t in database.tables()
            ]
        )

    def test_author(self, database):
        assert database.author == "pytest <pytest@qnl>"
