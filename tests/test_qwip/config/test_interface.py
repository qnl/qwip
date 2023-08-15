import pendulum
import pytest

from qwip.config.interface import ConfigFolder
from qwip.config.models import Parameter
from qwip.config.schema import ConfigSchema


@pytest.fixture(scope="function")
def config(session, models):
    return ConfigFolder(session=session)


class TestConfigFolder:
    def test_init(self, config):
        assert list(config.keys()) == []
        assert config.folder_id is None

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

    def test_parent(self, config):
        config.create_folder("child_dir")
        config["child_dir"].create_folder("grandchild_dir")
        assert config.parent() is None

        parent = config["child_dir"].parent()
        assert isinstance(parent, ConfigFolder)
        assert parent.path() == "/"

        parent = config["child_dir/grandchild_dir"].parent()
        assert isinstance(parent, ConfigFolder)
        assert parent.path() == "/child_dir/"

    def test_rename(self, config):
        with pytest.raises(ValueError):
            config.rename("root")

        config.create_folder("test_rename_1")
        config.create_folder("test_rename_2")

        with pytest.raises(ValueError):
            config["test_rename_1"].rename("test_rename_2")

        folder_id = config["test_rename_1"].folder_id
        assert config["test_rename_1"].rename("test_rename_3") == "/test_rename_3/"
        assert "test_rename_1" not in config
        assert config["test_rename_3"].folder_id == folder_id

    def test_get_parameter(self, config):
        config.create_parameter("tunable", False)
        now = pendulum.now()

        param = config.get_parameter("tunable")
        assert type(param) is Parameter
        assert param.name == "tunable"
        assert (now - param.timestamp).seconds < 1

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (ConfigFolder, dict(a=1, b=2, c=dict(d=3, e=4))),
            (ConfigFolder[str, int], dict(a=1, b=2, c=3)),
            (ConfigFolder[str, ConfigFolder], dict(a=dict(a1=1, b1=2))),
        ],
    )
    def test_create_all(self, session, models, cls, kwargs):
        config = cls(session=session)

        config.create_all(**kwargs)
        assert config.todict() == kwargs


class TestConfigSchema:
    def test_create_all(self, session, models):
        config = ConfigSchema(session=session)

        qubit_names = [f"Q{q}" for q in range(4)]
        resonator_names = [f"R{r}" for r in range(4)]
        config.create_all(
            hardware=dict(
                local_oscillators=dict(
                    qubit=dict(power=5, frequency=5.8e9),
                    readout=dict(power=0, frequency=6.5e9),
                    twpa=dict(power=8, frequency=7.8e9),
                ),
                num_dac_channels=16,
            ),
            readout=dict(
                default=dict(
                    classification={r: {} for r in resonator_names},
                    drives={r: f"{r}_basic" for r in resonator_names},
                    length=2048 / 1.8e9,
                )
            ),
            subsystems={q: {} for q in qubit_names} | {r: {} for r in resonator_names},
            targets=qubit_names + resonator_names,
        )

        assert list(config.keys()) == [
            "hardware",
            "readout",
            "subsystems",
            "pulses",
            "compilation",
            "extra",
            "version",
            "qwip_commit",
            "sample_id",
            "cooldown_id",
            "targets",
        ]
