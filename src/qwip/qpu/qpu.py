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
from qwip.config.database import ConfigDB, ConfigFolder, SequenceElementFolder
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
from qwip.sequencer.sequence import Sequence
from qwip.settings.settings import qdefine


@qdefine
class QPU:
    """Quantum Processing Unit.

    The QPU is meant to be the main interface for users to take measurements on a
    device. It orchestrates the execution of various subcomponents to take a
    circuit or sequence, execute it on hardware, and process the results.

    !!! note

        While the QPU wraps a lot of functionality into a simple user-facing
        interface, it is not meant to hold the actual logic for all these routines.
        Rather, these are delegated to submodules so that behavior can be modified
        by passing the QPU different submodules where the implementation details
        are contained.

    Attributes:
        db: A config database object. This is where all the configuration settings
            representing a quantum device are stored and referenced.
        subsystems: A mapping of targets to QuantumSystems, which together form a
            model of the quantum device.
        sequencer: A sequencer instance that controls the compilation from Sequences
            to waveform data and programs that are uploaded to hardware.
        pipeline: A data processing pipeline.

    """

    db: ConfigDB = field(repr=lambda db: db.database)
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

    def run(
        self,
        program: Sequence | CompiledSequence,
        processor: type[DataProcessor]
        | dict[str, type[DataProcessor]] = FormatLegacyIQ,
        repetitions: int = 512,
        readout: dict | SequenceElement = {},
        compilation: dict = {},
        backend: dict = {},
    ):
        match readout:
            case dict():
                ro_se = self.get_readout_sequence(**readout)
            case SequenceElement():
                ro_se = readout
            case _:
                raise ValueError(
                f"Readout must be a sequence element or a dictionary of parameters. "
                f"Got {readout}"
            )

        ## Update all frequencies before compilation
        self.update_modulations()
        self.backend.update_parameters(self)

        match program:
            case Sequence():
                cseq = self.sequencer.compile(program, readout=ro_se, **compilation)
                seq = cseq.sequence
            case CompiledSequence():
                cseq = program
                seq = cseq.sequence

            case _:
                raise NotImplementedError(
                    f"Only 'Sequence' and 'CompiledSequence' programs are currently "
                    f"supported. Got {program}"
                )

        self.backend.upload(cseq)
        meas = self.backend.acquire(cseq, repetitions=repetitions)

        return self.process_results(meas, processor)

    def get_readout_sequence(
        self,
        readout: str = "default",
        length: float | None = None,
        length_variable: str = "width",
    ) -> SequenceElement:
        readout_config = self.config.readout[readout]

        ro_se = SequenceElement()
        length = length or readout_config.length

        for r in self.sequencer.readout_qubits:
            pulse_name = readout_config.drives[f"R{r}"]

            ro_se += self.db.load_pulse(pulse_name, {length_variable: length})

        return ro_se

    def process_results(
        self,
        raw_data: dict,
        processor: type[DataProcessor]
        | dict[str, type[DataProcessor]] = FormatLegacyIQ,
    ) -> dict[str, MeasurementResult]:
        if not isinstance(processor, dict):
            processor = {f"R{r}": processor for r in self.sequencer.readout_qubits}

        return self.pipeline.process_results(raw_data, processor)
