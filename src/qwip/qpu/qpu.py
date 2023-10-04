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
from qwip.attrs import qdefine
from qwip.backends.backend import QuantumBackend
from qwip.config.interface import ConfigFolder, OfflineConfigDB, SequenceElementFolder
from qwip.config.schema import Target
from qwip.data.datastore import OfflineDatastore
from qwip.processing.data_processor import (
    DATA_PROCESSORS,
    DataProcessor,
    MeasurementResult,
    ReadoutPipeline,
)
from qwip.processing.processors import GMMClassification, IQRotation
from qwip.qpu.systems import REGISTERED_QSYSTEMS, QuantumSystem, ReadoutResonator
from qwip.sequencer import Sequence, SequenceElement
from qwip.sequencer.compilation import (
    REGISTERED_COMPILERS,
    ChannelInfo,
    DeviceInfo,
    QuantumExecutable,
    QWiPCompiler,
    TriggerInfo,
)
from qwip.sequencer.phase_tracker import ModulationFrequency


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
        compiler: A compiler instance that controls the compilation from Sequences
            to executable formats that can be uploaded to hardware.
        pipeline: A data processing pipeline.

    """

    db: OfflineConfigDB = field(repr=lambda db: db.database)
    subsystems: dict[Target, QuantumSystem] = field(
        repr=lambda sys: repr([s for s in sys])
    )
    compiler: QWiPCompiler
    pipeline: ReadoutPipeline
    backend: QuantumBackend | None = None
    datastore: OfflineDatastore | None = field(
        repr=lambda db: db.database, default=None
    )

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
        compiler = cls.load_compiler(db.config)
        pipeline = cls.load_pipeline(db.config, readout_config=readout_config)

        qpu = cls(
            db=db,
            compiler=compiler,
            pipeline=pipeline,
            subsystems=subsystems,
        )
        qpu.update_modulations()

        return qpu

    @classmethod
    def load_compiler(
        cls, config: ConfigFolder, modulations: dict[str, ModulationFrequency] = {}
    ) -> QWiPCompiler:
        compilation = config["compilation"]
        devices = []

        for key, dev_config in compilation["devices"].items():
            channels = tuple(
                ChannelInfo(**compilation["channels"][ch_name])
                for ch_name in dev_config["channels"]
            )

            devices.append(
                DeviceInfo(
                    name=key,
                    channels=channels,
                    sample_rate=dev_config["sample_rate"],
                    trigger=None
                    if dev_config["trigger"]["device"] is None
                    else TriggerInfo(**dev_config["trigger"]),
                )
            )

        if not modulations:
            modulations = dict()

        compiler_cls = compilation.get("compiler", dict(__class__="QWiPCompiler")).get(
            "__class__", "QWiPCompiler"
        )
        try:
            compiler_cls = REGISTERED_COMPILERS[compiler_cls]
        except KeyError:
            raise KeyError(f"'{compiler_cls}' is not a registered compiler.")

        compiler = compiler_cls.from_devices(devices, modulations=modulations)

        return compiler

    def save_compiler(self):
        with self.db.session.begin():
            for device in self.compiler.channels.values():
                self.config["compilation/devices"].create_all(**{device.name: {}})

                for ch in device.channels:
                    self.config["compilation/channels"].create_all(**{ch.name: {}})

                dev_info = qwip.converter.unstructure(device)
                dev_info["channels"] = [ch["name"] for ch in dev_info["channels"]]

                self.config["compilation/devices"][device.name].update(**dev_info)

                for ch in device.channels:
                    self.config["compilation/channels"][ch.name].update(
                        **qwip.converter.unstructure(ch)
                    )

    def update_modulations(self):
        modulation_keys = {}
        local_oscillators = {
            key: LO_info["frequency"]
            for key, LO_info in self.config["hardware/local_oscillators"].items()
        }

        for system in self.subsystems.values():
            match system:
                case QuantumSystem(get_modulations=_):
                    modulation_keys |= qwip.converter.structure(
                        system.get_modulations(local_oscillators),
                        dict[str, ModulationFrequency],
                    )
                case _:
                    logger.info(
                        f"Skipping modulation frequency for system {system.name}"
                    )
                    continue

        self.compiler.modulations.update(**modulation_keys)
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
            self.config.subsystems.create_all(**{k: {} for k in self.subsystems})
            for system in self.subsystems.values():
                system_data = qwip.converter.unstructure(system)

                match system_data:
                    case {"name": name, "__class__": cls, **parameters}:
                        self.config.subsystems[name]["system_class"] = cls

                        config_params = self.config[f"subsystems/{name}/parameters"]

                        for k, v in parameters.items():
                            try:
                                config_params[k] = v
                            except KeyError:
                                config_params.create_parameter(name=k, value=v)
                    case _:
                        raise ValueError(
                            f"Model data is missing parameters: {system_data}"
                        )

    def run(
        self,
        program: Sequence | QuantumExecutable | None,
        processor: type[DataProcessor] | dict[str, type[DataProcessor]] | None = None,
        repetitions: int = 512,
        compilation: dict = {},
        backend: dict = {},
        data: dict = {},
    ) -> dict[str, MeasurementResult]:
        """Uploads a sequence and acquires data from a backend.

        Args:
            program: A `Sequence` or a `QuantumExecutable` object. If given a `Sequence`,
                it will be compiled with any specified compilation parameters. If an
                executable, it should match the backend being used. Otherwise, if `None`,
                the previously uploaded sequence is run.
            processor: The data processor to use. See `qpu.process_results`.
            repetitions: The number of shots to take for each sequence element.
            compilation: The compilation arguments, which are passed to
                `self.compiler.compile`.

        Returns:
            A dictionary mapping measurement keys to the acquired and processed data.
        """

        ## Update all frequencies before compilation
        self.update_modulations()
        self.backend.update_parameters(self)

        match program:
            case Sequence():
                exe = self.compiler.compile(program, **compilation)
            case QuantumExecutable():
                exe = program
            case None:
                exe = self.backend.uploaded
            case _:
                raise NotImplementedError(
                    f"Only 'Sequence' and 'CompiledSequence' programs are currently "
                    f"supported. Got {program}"
                )

        if program is not None:
            self.backend.upload(exe)

        seq = exe.seq if exe else None
        raw_data = self.backend.acquire(exe, repetitions=repetitions, **backend)
        processed = self.process_results(raw_data, processor, seq=seq)

        if self.datastore:
            data = dict(config_db=self.db, seq=seq) | data
            self.datastore.save(self.pipeline.grouped_data(), **data)

        return processed

    def process_results(
        self,
        raw_data: dict,
        processor: type[DataProcessor] | dict[str, type[DataProcessor]],
        **kwargs,
    ) -> dict[str, MeasurementResult]:
        """Runs the readout pipeline and returns the processed data.

        Args:
            raw_data: A mapping from measurement keys to the raw data.
            processor: The final data processor. If it is a single data processor it
                will be used for all measurement keys. If it is a dictionary, each
                measurement key can specify a different data processor.

        Returns:
            A mapping from measurement keys to `MeasurementResult` instances.
        """
        if not isinstance(processor, dict):
            processor = {key: processor for key in raw_data.keys()}

        return self.pipeline.process_results(raw_data, processor, **kwargs)


__all__ = ["QPU"]
