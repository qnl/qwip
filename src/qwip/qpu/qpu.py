"""The QPU module.

The QPU (Quantum Processing Unit) handles all the automation necessary for treating
a superconduting quantum device as a quantum circuit processor, including circuit
compilation/transpilation, data acquisition, and measurement processing. This is the
main user interface for interacting with experimental devices.
"""
import re

from attrs import field
from loguru import logger

import qwip
from qwip._cattr import make_attrs_unstructure_fn
from qwip.config.database import ConfigDB, ConfigFolder, SequenceElementFolder, OfflineConfigDB
from qwip.config.schema import Target
from qwip.processing.data_processor import (
    DATA_PROCESSORS,
    DataProcessor,
    MeasurementResult,
    ReadoutPipeline,
)
from qwip.processing.processors import FormatLegacyIQ, GMMClassification, IQRotation
from qwip.qpu.backend import QuantumBackend
from qwip.qpu.systems import REGISTERED_QSYSTEMS, QuantumSystem, ReadoutResonator
from qwip.sequencer.compilation import (
    ChannelGroup,
    ChannelInfo,
    CompiledSequence,
    WaveformSequencer,
)
from qwip.sequencer.elements import SequenceElement
from qwip.sequencer.phase_tracker import ModulationFrequency
from qwip.settings import Settings, qdefine

REGISTERED_QSYSTEMS: dict[str, "QuantumSystem"] = dict()


def register_qsystem(cls) -> type:
    if not issubclass(cls, QuantumSystem):
        raise TypeError(f"Registered quantum model must subclass {QuantumSystem}")

    REGISTERED_QSYSTEMS[cls.__name__] = cls

    return cls


@qdefine
class QuantumSystem:
    name: str


@register_qsystem
@qdefine
class Transmon(QuantumSystem):
    frequency: float
    anharmonicity: float | None = None
    local_oscillator: str | None = None

    @property
    def frequency_EF(self) -> float:
        alpha = self.anharmonicity
        return None if alpha is None else self.frequency + alpha

    @property
    def mod_keys(self) -> tuple[str, ...]:
        return ("GE", "EF")

    def mod_frequency(self, local_oscillators, key="GE"):
        lo_freq = local_oscillators.get(self.local_oscillator)

        if not lo_freq:
            raise KeyError(
                f"Specified LO '{self.local_oscillator}' is not present in {local_oscillators}."
            )

        match key:
            case "GE":
                return self.frequency - lo_freq
            case "EF":
                return self.frequency_EF - lo_freq if self.frequency_EF else None
            case _:
                raise KeyError(
                    f"'{key}' is not a valid frequency key for {type(self).__name__}."
                )


@register_qsystem
@qdefine
class ReadoutResonator(QuantumSystem):
    frequency: float
    kappa: float | None = None
    chi: float | None = None
    local_oscillator: str | None = None

    def mod_frequency(self, local_oscillators) -> float:
        lo_freq = local_oscillators.get(self.local_oscillator)

        if not lo_freq:
            raise KeyError(
                f"Specified LO '{self.local_oscillator}' is not present in {local_oscillators}."
            )

        return self.frequency - lo_freq


def make_quantum_system_unstructure_fn(cls):
    unstructure_attrs = make_attrs_unstructure_fn(cls)

    def unstructure_fn(obj):
        return {**unstructure_attrs(obj), "__class__": type(obj).__name__}

    return unstructure_fn


qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, QuantumSystem), make_quantum_system_unstructure_fn
)


@qdefine
class QPU:
    db: OfflineConfigDB = field(repr=lambda db: db.database)
    subsystems: dict[Target, QuantumSystem] = field(
        repr=lambda sys: repr([s for s in sys])
    )
    sequencer: WaveformSequencer
    pipeline: ReadoutPipeline
    backend: QuantumBackend | None = None

    @property
    def config(self) -> ConfigFolder:
        return self.db.config

    @property
    def pulses(self) -> SequenceElementFolder:
        return self.db.pulses

    def set_backend(self, backend: QuantumBackend):
        self.backend = backend

    @classmethod
    def load(cls, db, readout_config: str = "default"):
        if not db.session or not db.config:
            db.connect()

        subsystems = cls.load_subsystems(db.config)
        sequencer = cls.load_sequencer(db.config)
        pipeline = cls.load_pipeline(db.config, readout_config=readout_config)

        qpu = cls(
            db=db,
            sequencer=sequencer,
            pipeline=pipeline,
            subsystems=subsystems,
        )
        qpu.update_modulations()

        ro_qubits = []
        for sys in qpu.subsystems.values():
            match sys:
                case ReadoutResonator(name=n):
                    r = int(re.match(r"R(\d+)", n)[1])
                    ro_qubits.append(r)
        qpu.sequencer.readout_qubits = ro_qubits

        return qpu

    @classmethod
    def load_sequencer(
        cls, config: ConfigFolder, modulations: dict[str, ModulationFrequency] = {}
    ) -> WaveformSequencer:
        compilation = config["compilation"]
        channel_groups = []

        for key, ch_group_config in compilation["channel_groups"].items():
            channels = tuple(
                ChannelInfo(**compilation["channels"][ch_name])
                for ch_name in ch_group_config["channels"]
            )

            channel_groups.append(
                ChannelGroup(
                    name=key,
                    channels=channels,
                    sample_rate=ch_group_config["sample_rate"],
                )
            )

        if not modulations:
            modulations = dict()

        sequencer = WaveformSequencer.from_channel_groups(
            channel_groups, modulations=modulations
        )

        return sequencer

    def save_sequencer(self):
        with self.db.session.begin():
            for group in self.sequencer.channels.values():
                self.config["compilation/channel_groups"][group.name].update(
                    name=group.name, sample_rate=group.sample_rate
                )

                ch_names = []
                for ch in group.channels:
                    ch_names.append(ch.name)

                    self.config["compilation/channels"][ch.name].update(
                        name=ch.name,
                        index=ch.index,
                        group=ch.group,
                        subchannel=ch.subchannel,
                        delay=ch.delay,
                    )

                self.config["compilation/channel_groups"][group.name][
                    "channels"
                ] = ch_names

    def update_modulations(self):
        modulation_keys = {}
        local_oscillators = {
            key: LO_info["frequency"]
            for key, LO_info in self.config["hardware/local_oscillators"].items()
        }

        for name, system in self.subsystems.items():
            match system:
                case QuantumSystem(mod_keys=_, mod_frequency=_):
                    for key in system.mod_keys:
                        mod_freq = system.mod_frequency(local_oscillators, key=key)

                        if mod_freq:
                            modulation_keys[f"mod_{name}_{key}"] = ModulationFrequency(
                                mod_freq
                            )
                case QuantumSystem(mod_frequency=_):
                    modulation_keys[f"mod_{name}"] = ModulationFrequency(
                        system.mod_frequency(local_oscillators)
                    )
                case _:
                    logger.info(
                        f"Skipping modulation frequency for system {system.name}"
                    )
                    continue

        self.sequencer.modulations.update(**modulation_keys)
        return modulation_keys

    @classmethod
    def load_pipeline(
        cls, config: ConfigFolder, readout_config: str
    ) -> ReadoutPipeline:
        processors = []

        ro_config = config["readout"][readout_config]

        for processor_cls in DATA_PROCESSORS.data_processors():
            if issubclass(processor_cls, GMMClassification | IQRotation):
                continue
            else:
                processors.append(processor_cls())

        for k, classification in ro_config["classification"].items():
            processors.append(
                GMMClassification(
                    measurement_key=k,
                    means=classification["means"].astype(float),
                    covariances=classification["covariances"].astype(float),
                    num_states=classification["num_states"],
                )
            )

            if angle := classification["rotation"]:
                processors.append(IQRotation(measurement_key=k, angle=angle))

        return ReadoutPipeline(name=readout_config, processors=processors)

    def save_pipeline(self):
        with self.db.session.begin():
            ro_config = self.config["readout"][self.pipeline.name]

            for processor in self.pipeline.processors:
                match processor:
                    case GMMClassification(
                        measurement_key=k, means=m, covariances=c, num_states=s
                    ):
                        ro_config["classification"][k] = dict(
                            means=m, covariances=c, num_states=s
                        )
                    case IQRotation(angle) if angle:
                        ro_config[f"classification/{k}/rotation"] = angle

    @classmethod
    def load_subsystems(cls, config: ConfigFolder) -> dict[Target, QuantumSystem]:
        subsystems = {}

        for target, system_info in config.subsystems.items():
            system_cls = REGISTERED_QSYSTEMS.get(system_info["system_class"])

            if not system_cls:
                raise TypeError(f"'{system_cls}' is not a registered model.")

            subsystems[target] = system_cls(name=target, **system_info["parameters"])

        return subsystems

    def save_subsystems(self):
        ## Starting a session here ensures that either all the models get saved or
        ## none do.
        with self.db.session.begin():
            for system in self.subsystems.values():
                system_data = qwip.converter.unstructure(system)

                match system_data:
                    case {"name": name, "__class__": cls, **parameters}:
                        self.config.subsystems[name]["system_class"] = cls

                        for k, v in parameters.items():
                            self.config[f"subsystems/{name}/parameters/{k}"] = v
                    case _:
                        raise ValueError(
                            f"Model data is missing parameters: {system_data}"
                        )
