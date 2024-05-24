"""The QPU module.

The QPU (Quantum Processing Unit) handles all the automation necessary for treating
a superconduting quantum device as a quantum circuit processor, including circuit
compilation/transpilation, data acquisition, and measurement processing. This is the
main user interface for interacting with experimental devices.
"""

from collections import defaultdict

import attrs
import numpy as np
import pandas as pd
from attrs import field
from loguru import logger

import qwip
from qwip.attrs import qdefine, qfrozen
from qwip.backends.backend import QuantumBackend
from qwip.config.interface import ConfigFolder, OfflineConfigDB, PulsesFolder
from qwip.config.schema import Target
from qwip.data.datastore import OfflineDatastore
from qwip.processing.data_processor import (
    DATA_PROCESSORS,
    DataProcessor,
    MeasurementResult,
    ReadoutPipeline,
)
from qwip.processing.processors import BatchReindex, GMMClassification, IQRotation
from qwip.qpu.systems import REGISTERED_QSYSTEMS, QuantumSystem
from qwip.sequencer import Sequence
from qwip.sequencer.compilation import (
    REGISTERED_COMPILERS,
    BatchedExecutable,
    ChannelInfo,
    DeviceInfo,
    QuantumExecutable,
    QWiPCompiler,
)
from qwip.sequencer.phase_tracker import Frame
from qwip.utils import deprecated


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
    def pulses(self) -> PulsesFolder:
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
        qpu.update_frames()

        return qpu

    @classmethod
    def load_compiler(
        cls, config: ConfigFolder, frames: dict[str, Frame] = {}
    ) -> QWiPCompiler:
        compilation = config["compilation"]
        devices = []

        for dev_config in compilation["devices"].values():
            channels = tuple(
                ChannelInfo(**compilation["channels"][ch_name])
                for ch_name in dev_config["channels"]
            )

            unstruct = dev_config.todict()
            unstruct["channels"] = channels
            if unstruct["trigger"]["device"] is None:
                unstruct["trigger"] = None

            devices.append(qwip.converter.structure(unstruct, DeviceInfo))

        if not frames:
            frames = dict()

        compiler_cls = compilation.get("compiler", dict(__class__="QWiPCompiler")).get(
            "__class__", "QWiPCompiler"
        )
        try:
            compiler_cls = REGISTERED_COMPILERS[compiler_cls]
        except KeyError:
            raise KeyError(f"'{compiler_cls}' is not a registered compiler.")

        compiler = compiler_cls.from_devices(devices, frames=frames)

        return compiler

    def save_compiler(self):
        with self.db.session.begin():
            for device in self.compiler.devices.values():
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

    def update_frames(self):
        frames = {}
        local_oscillators = {
            key: LO_info["frequency"]
            for key, LO_info in self.config["hardware/local_oscillators"].items()
        }

        for system in self.subsystems.values():
            match system:
                case QuantumSystem(get_frames=_):
                    frames |= qwip.converter.structure(
                        system.get_frames(local_oscillators),
                        dict[str, Frame],
                    )
                case _:
                    logger.info(f"Skipping frame for system {system.name}")
                    continue

        self.compiler.frames.update(**frames)
        return frames

    @deprecated(
        version="23.10.0",
        removed="23.11.0",
        message="Use `qpu.update_frames()` instead.",
    )
    def update_modulations(self):
        return self.update_frames()

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

        return ReadoutPipeline(
            name=readout_config, processors=processors, default_processor=BatchReindex
        )

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

            unstructured = dict(
                name=target,
                **system_info["parameters"],
            )

            subsystems[target] = qwip.converter.structure(unstructured, system_cls)

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

    def batch_program(
        self,
        program: Sequence | QuantumExecutable | None,
        repetitions: int,
        timelines_per_batch: int | None = None,
        repetitions_per_batch: int | None = None,
        compilation: dict = {},
    ) -> list[BatchedExecutable]:
        match program:
            case Sequence():
                seq = program
                flattened = seq.flatten()
                N = len(flattened)
                timelines_per_batch = timelines_per_batch or N

                exes = [
                    self.compiler.compile(
                        flattened[tmln_idx : tmln_idx + timelines_per_batch],
                        **compilation,
                    )
                    for tmln_idx in np.r_[:N:timelines_per_batch]
                ]

            case QuantumExecutable():
                if timelines_per_batch is not None:
                    logger.warning(
                        f"Cannot batch a pre-compiled executable by timeline. Argument "
                        f"`timelines_per_batch` = {timelines_per_batch} will be ignored."
                    )

                exes = [program]
            case None:
                exes = [self.backend.uploaded]
            case _:
                raise NotImplementedError(
                    f"Only 'Sequence' and 'CompiledSequence' programs are currently "
                    f"supported. Got {program}"
                )

        repetitions_per_batch = repetitions_per_batch or repetitions

        batched_exes = []
        for rep_idx in np.r_[:repetitions:repetitions_per_batch]:
            for batch_idx, exe in enumerate(exes):
                batch_reps = min(repetitions_per_batch, repetitions - rep_idx)

                skip_upload = (program is None) or (len(exes) == 1 and rep_idx > 0)

                batched_exes.append(
                    BatchedExecutable(
                        exe=exe,
                        repetitions=batch_reps,
                        timeline_index=batch_idx * timelines_per_batch,
                        repetition_index=rep_idx,
                        batch_index=(batch_idx, rep_idx // repetitions_per_batch),
                        upload=not skip_upload,
                    )
                )

        return batched_exes

    def run(
        self,
        program: Sequence | QuantumExecutable | None,
        processor: type[DataProcessor] | dict[str, type[DataProcessor]] | None = None,
        repetitions: int = 512,
        batch: dict = {},
        compilation: dict = {},
        backend: dict = {},
        data: dict = {},
        save: bool = True,
    ) -> dict[str, MeasurementResult]:
        """Uploads a sequence and acquires data from a backend.

        Args:
            program: A `Sequence` or a `QuantumExecutable` object. If given a `Sequence`,
                it will be compiled with any specified compilation parameters. If an
                executable, it should match the backend being used. Otherwise, if `None`,
                the previously uploaded sequence is run.
            processor: The data processor to use. See `qpu.process_results`.
            repetitions: The number of shots to take for each pulse timeline.
            compilation: The compilation arguments, which are passed to
                `self.compiler.compile`.
            backend: Any backend arguments which are passed to `self.backend.acquire`
            data: Data saving arguments which are passed to `self.datastore.save`
            save: Whether or not to save the data.

        Returns:
            A dictionary mapping measurement keys to the acquired and processed data.
        """

        ## Update all frequencies before compilation
        self.update_frames()
        self.backend.update_parameters(self)

        batched_exes = self.batch_program(
            program, repetitions=repetitions, **batch, compilation=compilation
        )

        if self.datastore and save:
            data = dict(config_db=self.db) | data
            dataset = self.datastore.save(**data)

        results = defaultdict(list)
        for batch in batched_exes:
            exe = batch.exe
            if batch.upload:
                self.backend.upload(exe)

            raw_data = self.backend.acquire(repetitions=batch.repetitions, **backend)

            processed = self.process_results(raw_data, processor, exe=exe, batch=batch)
            for k, res in processed.items():
                results[k].append(res)

            if self.datastore and save:
                if exe and exe.sequence is None:
                    data["executable"] = batch.exe
                elif exe:
                    data["sequence"] = exe.sequence

                asset_kwargs = dict()
                if len(batched_exes) > 1:
                    suffix = f"_t{batch.batch_index[0]}-r{batch.batch_index[1]}"
                    asset_kwargs["name_fmt"] = "{name}" + suffix

                with self.datastore.begin():
                    assets = self.datastore._make_assets(
                        self.pipeline.grouped_data(), **asset_kwargs
                    )
                    dataset.add(assets)
                    for asset in assets:
                        asset.save()

        if len(batched_exes) > 1:
            result = {}
            for k, rlist in results.items():
                data = pd.concat([r.d for r in rlist])
                result[k] = attrs.evolve(rlist[0], data=data)

            if self.datastore and save:
                with self.datastore.begin():
                    assets = self.datastore._make_assets([result])
                    dataset.add(assets)
                    for asset in assets:
                        asset.save()
        else:
            result = processed

        return result

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
