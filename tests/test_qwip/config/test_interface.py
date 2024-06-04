import pendulum
import pytest

from qwip.config.interface import ConfigFolder
from qwip.config.models import Parameter
from qwip.config.schema import ConfigSchema
from qwip.sequencer import *


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

        assert set(config.keys()) == {
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
        }


class TestConfigDB:
    @pytest.fixture(scope="class")
    def populated_config_db(self, config_db):
        config_db.config.create_all(
            targets=[f"Q{i}" for i in range(4)] + [f"R{i}" for i in range(4)]
        )

        yield config_db

    @pytest.fixture
    def ro_tmln(self):
        drive = ModulatedWaveform(
            envelope=SquareWaveform(width="drive_width"),
            modulation=CWWaveform(
                amplitude="amplitude",
                frequency="frequency",
                channel="drive",
                hardware_modulation=True,
            ),
        )

        demod = ModulatedWaveform(
            envelope=SquareWaveform(width="demod_width"),
            modulation=CWWaveform(
                amplitude=1,
                frequency="frequency",
                channel="demod",
                hardware_modulation=True,
            ),
        )

        ro_tmln = Timeline().add(drive).add(demod, "demod_delay")
        return ro_tmln

    def test_add_pulse(self, populated_config_db, ro_tmln):
        db = populated_config_db

        db.add_pulse(
            name="Q0_RO_basic",
            targets=["R0"],
            pulse_key="RO_basic",
            tmln=ro_tmln,
            channel_map=dict(drive="Q0.qdrv", demod="Q0.rdlo"),
        )

        db.add_pulse(
            name="Q1_RO_basic",
            targets=["R1"],
            pulse_key="RO_basic",
            channel_map=dict(drive="Q1.qdrv", demod="Q1.rdlo"),
        )

        for i in range(2):
            pulse = db.config.pulses[f"Q{i}_RO_basic"]
            assert pulse.pulse_key == "RO_basic"
            assert pulse.channels.todict() == dict(
                drive=f"Q{i}.qdrv", demod=f"Q{i}.rdlo"
            )
            assert set(pulse.variables.keys()) == ro_tmln.variables()

        assert db.pulses["RO_basic"] == ro_tmln

    def test_load_pulse(self, populated_config_db, ro_tmln):
        db = populated_config_db

        db.add_pulse(
            name="Q2_RO",
            targets=["R0"],
            pulse_key="RO_square",
            tmln=ro_tmln,
            channel_map=dict(drive="Q2.qdrv", demod="Q2.rdlo"),
        )

        db.config.pulses["Q2_RO"].variables.update(
            amplitude=0.01,
            frequency=6.543e9,
            drive_width=1e-6,
            demod_width=1.2e-6,
            demod_delay=200e-9,
        )

        q2_ro = db.load_pulse("Q2_RO")

        assert q2_ro.variables() == set()
        assert q2_ro.channels == {"Q2.qdrv", "Q2.rdlo"}
