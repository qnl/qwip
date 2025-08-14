from collections import defaultdict
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Self,
    Tuple,
)

import numpy as np

from laboneq.simple import (
    pulse_library,
    CompiledExperiment,
    DeviceSetup,
    Experiment,
    ExperimentSignal,
    ModulationType,
    Oscillator,
    Session,
    SignalCalibration,
)

import laboneq.controller.devices as l1q_devices

from qwip.attrs import qdefine, qfrozen
from qwip.backends.backend import QuantumBackend
from qwip.instruments.zurich.device_options import (
    DEVICE_TYPES,
    DEVICE_SIGNAL_TYPES,
    DevicePurpose,
    get_device_nchannels,
)
from qwip.instruments.zurich.device_setup import create_setup

from qwip.sequencer.compilation import (
    ChannelInfo,
    DeviceInfo,
    Instruction,
    IntermediateProgram,
    PlayInstruction,
    ReadInstruction,
    QuantumExecutable,
    QWiPCompiler,
    QWiPExecutable,
    WaveformMemory,
    register_compiler,
)
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.timeline import Timeline
from qwip.sequencer.utils import _to_python_number
from qwip.sequencer.waveform import ModulatedWaveform

__all__ = [
    "LabOneQDeviceInfo",
    "LabOneQExecutable",
    "LabOneQCompiler",
    "LabOneQBackend",
]


LabOneQDevicePurpose = DevicePurpose

@qfrozen
class LabOneQDeviceInfo(DeviceInfo):
    """Device properties from LabOneQ to QWiP DeviceInfo"""

    kind: str
    purpose: LabOneQDevicePurpose

    @staticmethod
    def _create_channelinfo(iq, label, nchannels):
        if iq:
            channels = tuple(
                ChannelInfo(f"{label}{i}_IQ", i) for i in range(nchannels)
            )
            dtype = np.complex64
        else:
            channels = tuple(
                ChannelInfo(f"{label}{i}_" + (i%2 and "Q" or "I"), i) for i in range(nchannels)
            )
            dtype = np.float32
        return channels, dtype

    @classmethod
    def from_defaults(cls, kind: str, name: str, purpose: LabOneQDevicePurpose,
                      sample_rate: float=None):

        nchannels, iq = get_device_nchannels(kind, purpose)

        if purpose == LabOneQDevicePurpose.DRIVE:
            channels, dtype = cls._create_channelinfo(iq, "Q", nchannels)
            name = name+"_drive"
        elif purpose == LabOneQDevicePurpose.MEASURE:
            channels, dtype = cls._create_channelinfo(iq, "R", nchannels)
            name = name+"_measure"
        elif purpose == LabOneQDevicePurpose.ACQUIRE:
            channels = tuple (
                ChannelInfo(f"R{r}", r) for r in range(nchannels)
            )
            dtype = np.complex64
            name = name+"_acquire"

        if sample_rate is None:
            sample_rate = cls.default_sample_rate(kind, purpose)

        return cls.from_channels(
            kind=kind.lower(),
            purpose=purpose,
            name=name,
            channels=channels,
            sample_rate=sample_rate,
            dtype=dtype,
            trigger=None,    # LabOneQ programs work with timing info only
        )

    @staticmethod
    def default_sample_rate(kind: str, purpose: LabOneQDevicePurpose = None):
        _kind = kind.lower()
        if _kind == "shfqc":
            if purpose == LabOneQDevicePurpose.DRIVE:
                _kind = "shfsg"
            else:
                _kind = "shfqa"

        return getattr(l1q_devices, f"device_{_kind}").SAMPLE_FREQUENCY_HZ


@qdefine
class LabOneQExecutable(QWiPExecutable):
     experiment: Experiment
     binary:     CompiledExperiment|None
     modulation: Dict


_uid_counter = 0

@register_compiler
@qdefine
class LabOneQCompiler(QWiPCompiler):
    """LabOneQ-based compiler, targeting Zurich Instruments' devices."""

    @classmethod
    def from_devices(cls, devices: Iterable[DeviceInfo], **kwargs: Any) -> Self:
        return super().from_devices(devices, **kwargs)

    def compile(
        self,
        seq: Sequence,
        substitutions: dict = {},
    ) -> LabOneQExecutable:
        """Compiles a sequence to a LabOneQ executable format.

        Compiles an abstract sequence into a concrete set of timepoints lined up
        on a LabOneQ sequence.

        For LabOneQ, which has its own compiler, the pulse information on each
        timeline needs to be retained at a higher level in order to allow its wave
        optimizer to do its job. The standard compilation chain places all pulses
        of a timeline onto a single piece of wave memory. The approach here splits
        up the timeline to retain separate pulse instructions (and time stamps).

        Args:
            seq: The sequence to compile.
            substitutions: A dictionary mapping variables to substitutions that get
                passed to `Timeline.resolve`.

        Returns:
            A `LabOneQExecutable` instance.
        """

        global _uid_counter

        # find the channels used in this circuit to setup logical groups
        channels = set()
        for tmln in seq.flat:
            for loc, w in tmln:
                if isinstance(w, ModulatedWaveform):
                    for channel in [w.envelope.channel, w.modulation.channel]:
                        if channel:
                            channels.add(channel)

        _uid_counter += 1
        exp = Experiment(
            uid="QWiP_experiment_%d" % _uid_counter,
            signals=[
                ExperimentSignal(x) for x in channels
            ],
        )

        modulation = dict()
        exe = LabOneQExecutable.from_devices(
                  sequence=seq,
                  devices=self.devices.values(),
                  experiment=exp,
                  binary=None,
                  modulation=modulation)

        signal_map = dict()
        signal_time = defaultdict(int)

        readout_weighting_function = pulse_library.const(
            uid="readout_weighting_function", length=400e-9, amplitude=1.0
        )

        with exp.acquire_loop_rt(count=1, uid="shots"):

            with exp.section(uid="pulse"):
                for tmln in seq.flat:
                    exe.num_reads.append(0)
                    for loc, w in tmln:
                        # create a fresh timeline to compile to ensure that the start of the
                        # waveform is aligned with the start of the wave data
                        subt = Timeline()
                        subt.add(w, 0.0)                 # align at 0

                        instruction_cache = dict()
                        self.compile_timeline(exe, subt, substitutions, instruction_cache)

                        for (_, device), ir in instruction_cache.items():
                            program = exe.programs[device]
                            for inst in ir:
                                if isinstance(inst, PlayInstruction):
                                    wmem = program.waveforms[inst.waveform_index]
                                    for (ch, sub), samples in wmem.data.items():
                                        devinfo = self.devices[device]

                                        # reverse lookup the signal name for re-mapping through
                                        # LabOneQ's logical/physical signal groupings
                                        signal = devinfo.channels[ch].name

                                        # reverse-engineer the logical grouping
                                        if not signal in signal_map:
                                            sigtype = DEVICE_SIGNAL_TYPES[devinfo.purpose]
                                            signal_map[signal] = f"/logical_signal_groups/q{ch}/{sigtype}"

                                        # line up timestamp
                                        tdiff = loc - signal_time[signal]
                                        exp.delay(signal=signal, time=tdiff)

                                        # play the pulse
                                        pulse = pulse_library.PulseSampled(samples=samples)
                                        exp.play(signal=signal, pulse=pulse)

                                        # if this is a measurement pulse, follow up with an acquisition
                                        """if devinfo.purpose == DevicePurpose.MEASURE:
                                            exp.acquire(
                                                signal=signal,#[:2]+"_acquire",
                                                handle=signal[:2],
                                                length=w.envelope.width,
                                            )
                                        """

                                        # collect modulation for physical channels to assign later
                                        # in the backend
                                        freq = self.frames[w.modulation.frequency.offset].offset
                                        modulation[(f"q{ch}", sigtype)] = freq

                                        # adjust the current time on this channel (TODO: this assumes there
                                        # are no pulse overlaps)
                                        signal_time[signal] = loc + w.envelope.width

        # update all used channels to point to their respective logical groupings
        exp.set_signal_map(signal_map)

        return exe


@qdefine
class LabOneQBackend(QuantumBackend):
    """A hardware backend for interfacing with ZI through LabOneQ."""

    devices: Dict                 # devices' names and infos
    setup:   DeviceSetup          # instruments and their connections
    session: Session              # runtime access to instruments

    def __init__(self, devices: Dict[str, List[str]]) -> None:
        self.devices = {key.lower(): devices[key] for key in devices.keys()}
        self.setup   = create_setup(self.devices)
        self.session = Session(device_setup=self.setup)

    @classmethod
    def from_setup(cls, devices: Dict[str, List[str]]) -> Self:
        return cls(devices)

    def _device_infos_p(self, purpose: LabOneQDevicePurpose) -> Iterable[DeviceInfo]:
        selected = DEVICE_TYPES[purpose]

        devices = list()
        for kind, names in self.devices.items():
            if kind in selected:
                for name in names:
                    devices.append(
                        LabOneQDeviceInfo.from_defaults(kind, name, purpose)
                    )

        return devices

    def drive_devices(self) -> Iterable[DeviceInfo]:
        return self._device_infos_p(LabOneQDevicePurpose.DRIVE)

    def readout_devices(self) -> Iterable[DeviceInfo]:
        return self._device_infos_p(LabOneQDevicePurpose.READOUT)

    def demod_devices(self) -> Iterable[DeviceInfo]:
        return self._device_infos_p(LabOneQDevicePurpose.DEMOD)

    def device_infos(self) -> Tuple[Iterable[DeviceInfo]]:
        return (self.drive_devices(), self.readout_devices(), self.demod_devices())

    def upload(self, exe: LabOneQExecutable, **kwargs) -> None:
        for (qubit, purpose), freq in exe.modulation.items():
            signals = self.setup.logical_signal_groups[qubit].logical_signals
            signals[purpose].calibration = SignalCalibration(
                oscillator=Oscillator(frequency=freq, modulation_type=ModulationType.HARDWARE
            ))

        # only at this point can the actual compilation happen
        exe.binary = self.session.compile(exe.experiment)

        # TODO: not technically uploaded yet; need to figure out how
        # this relates wrt. the LabOneQ pipeliner
        self.uploaded = exe

    def acquire(self, **kwargs) -> dict:
        compiled_exp = self.uploaded.binary

        self.session.connect(do_emulation=True)
        output = self.session.run(compiled_exp);

        return output.acquired_results

    def update_parameters(self, qpu: "QPU", **kwargs):
        # parameters are added to phyiscal groups; upload will connect
        # physical to logical, followed by the actual compilation step

        # local oscillators for qubits and readout
        LO_config = qpu.config["hardware/local_oscillators"]
        LO_drive   = LO_config["qubit"]["frequency"]
        LO_measure = 6400000000 # LO_config["readout"]["frequency"]

        for group in self.setup.logical_signal_groups.values():
            signals = group.logical_signals
            for lo_freq, purpose in [(LO_drive, "drive"),
                                     (LO_measure, "measure"),
                                     (LO_measure, "acquire"),]:
                try:
                    signals[purpose].local_oscillator = Oscillator(
                        frequency=lo_freq, modulation_type=ModulationType.HARDWARE
                    )
                except KeyError:
                    pass     # grouping not used for `purpose`


