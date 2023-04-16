import warnings

import attrs
import pendulum
import pytest
import sqlalchemy as sa

from qwip.config.database import Branch, Commit, ConfigDB, ConfigFolder, ReadOnlyParameter
from qwip.config.metadata import QWIP_DB_METADATA
from qwip.config.models import Folder, Parameter
from qwip.settings.settings import Settings, qdefine


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

@pytest.fixture(scope="function")
def config(session, models):
    return ConfigFolder(session=session)

class TestConfigFolder:
    def test_init(self, config):
        assert list(config.keys()) == []
        assert config.folder_id  is None

    def test_create(self, config):
        with pytest.raises(KeyError):
            config["folder1"] = {}

        config.create_folder("folder1")
        config.create_folder("/folder2/")
        config.create_parameter("p1", 10)
        config.create_parameter("folder1/p2", True)

        assert config.todict() == dict(folder1=dict(p2=True), p1=10, folder2=dict())

    def test_get(self, config):
        config.create_parameter("Q0", "qubit")
        config.create_parameter("Q1", 1)
        
        assert config["Q0"] == config.Q0 == "qubit"

        with pytest.raises(KeyError):
            config["Q3"]

        assert config.get("Q3") is None
        assert getattr(config, "Q3", None) is None

    def test_delete(self, config):
        config.create_folder("qubits")
        config.create_parameter("qubits/Q0", 0)
        config.create_parameter("qubits/Q1", 1)
        config.create_parameter("is_transmon", True)

        assert config.todict() == dict(qubits=dict(Q0=0, Q1=1), is_transmon=True)

        del config["is_transmon"]
        assert config.todict() == dict(qubits=dict(Q0=0, Q1=1))
        
        delattr(config, "qubits")
        assert config.todict() == dict()

    def test_folder_id(self, config):
        assert config.folder_id is None

        config.create_folder("folder")
        assert config["folder"].folder_id is not None

    def test_subfolder_class(self, config):
        assert config.subfolder_class == ConfigFolder

    def test_path(self, config):
        assert config.path() == "/"

        config.create_folder("folder")
        assert config["folder"].path() == "/folder/"

        config.create_folder("nested/folder/")
        assert config["nested/folder"].path() == "/nested/folder/"

    def test_folder_name(self, config):
        assert config.folder_name() is None

        config.create_folder("dir_a")
        assert config["dir_a"].folder_name() == "dir_a"

    def test_get_parameter(self, config):
        config.create_parameter("tunable", False)
        now = pendulum.now()

        param = config.get_parameter("tunable")
        assert type(param) is ReadOnlyParameter
        assert param.name == "tunable"
        assert (now - param.timestamp).seconds < 1