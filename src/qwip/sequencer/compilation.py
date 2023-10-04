import itertools as it
from abc import ABCMeta
from collections.abc import Callable, Collection, Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from attrs import evolve, field
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq, fftshift
from typing_extensions import Self

from qwip.attrs import qdefine, qfrozen
from qwip.sequencer.elements import SequenceElement
from qwip.sequencer.phase_tracker import ModulationFrequency, PhaseTracker, PhaseUpdater
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.utils import Location
from qwip.sequencer.waveform import Marker, ReadoutMarker, TriggeredWaveform, Waveform
from qwip.visualization.utils import all_legend_handles_labels

REGISTERED_COMPILERS: dict[str, "QWiPCompiler"] = dict()


def register_compiler(cls: type["QWiPCompiler"]) -> type["QWiPCompiler"]:
    if not issubclass(cls, QWiPCompiler):
        raise TypeError(f"Registered sequencer must subclass {QWiPCompiler}")

    REGISTERED_COMPILERS[cls.__name__] = cls

    return cls


def find_end_marker(locations, name="end") -> Location | None:
    for loc, waves in locations.items():
        if Marker(name=name) in waves:
            return loc

    return None


@qfrozen(kw_only=False)
class ChannelInfo:
    """Holds information mapping a logical channel name to a physical channel index.

    Attributes:
        name: The logical channel name.
        index: The physical channel index (0-indexed) corresponding to the hardware channel.
        device: The name of the device this channel belongs to.
        subchannel: The subchannel (used for markers) that this channel name refers to.
        read: True if the channel is an ADC channel.
        delay: A channel delay in ns to add to all waves on this channel.
    """

    name: str
    index: int
    device: str | None = None
    subchannel: int = 0  # Use nonzero for markers
    read: bool = False
    output: bool = True
    delay: float = 0


@qfrozen
class TriggerInfo:
    device: str
    index: int
    subchannel: int = 0


@qfrozen
class DeviceInfo:
    """Holds information about a device, which contains a group of logical channels.

    A device typically corresponds to a single hardware instrument or set of instruments
    that are addressed together. They should all share the same sample rate.

    Attributes:
        name: The name of the device.
        channels: A tuple of all channels that belong to this device.
        sample_rate: The sampling rate of the device in samples/second.
    """

    name: str
    channels: tuple[ChannelInfo, ...] = field(
        repr=lambda channels: repr(tuple(ch.name for ch in channels))
    )
    sample_rate: float
    trigger: TriggerInfo | None = None

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def num_subchannels(self) -> int:
        return len(set(ch.subchannel for ch in self))

    @property
    def max_channel_index(self) -> int:
        return max(ch.index for ch in self)

    @property
    def max_subchannel_index(self) -> int:
        return max(ch.subchannel for ch in self)

    def get_max_channel_delay(self) -> int:
        """Returns the largest channel delay specified within the device."""
        return max(ch.delay for ch in self)

    def channel_names(self) -> set[str]:
        return {ch.name for ch in self.channels}

    def channel_indices(self) -> set[int]:
        return {ch.index for ch in self.channels}

    def has_output(self) -> bool:
        for ch in self.channels:
            if ch.output:
                return True
        return False

    @classmethod
    def from_channels(
        self,
        channels: Iterable[ChannelInfo],
        sample_rate: float,
        name: str | None = None,
        **kwargs,
    ) -> Self:
        """Constructs a device from a list of channels.

        Args:
            channels: The channels that belong to the device. They must all have unique names,
                and should not have conflicting devices. If the device attribute for all the
                channels is None, then name must be provided.
            sample_rate: The sampling rate for the device.
            name: The name of the device. This can be inferred from the channels if
                they specify a device. Otherwise, name must be provided.

        Returns:
            The device.
        """
        device = name
        names = set()

        for ch in channels:
            if device and ch.device and ch.device != device:
                raise ValueError(
                    "Cannot create device from channels with different device names."
                )

            device = device or ch.device

            if ch.name in names:
                raise ValueError("Channels must all have unique names!")

            names.add(ch.name)

        channels = tuple(
            ch if ch.device else evolve(ch, device=device) for ch in channels
        )

        return DeviceInfo(
            name=device, channels=channels, sample_rate=sample_rate, **kwargs
        )

    def __iter__(self):
        yield from self.channels.__iter__()

    def __getitem__(self, channel_name: str):
        try:
            return next(ch for ch in self if ch.name == channel_name)
        except StopIteration as e:
            raise KeyError(f"'{channel_name}'") from e


@qdefine
class WaveformMemory:
    samples: int
    sample_rate: float
    data: dict[tuple[int, int], np.ndarray] = field(repr=lambda x: repr(x.keys()))

    @classmethod
    def from_channels(
        cls, samples: int, sample_rate: float, channels: Iterable["ChannelInfo"]
    ) -> Self:
        data = {(ch.index, ch.subchannel): np.zeros(samples) for ch in channels}
        return cls(samples=samples, sample_rate=sample_rate, data=data)

    def __getitem__(self, key) -> np.ndarray:
        match key:
            case int():
                return self.data[key, 0]
            case ChannelInfo(index=idx, subchannel=sc):
                return self.data[idx, sc]
            case _:
                return self.waveforms[key]

    def __contains__(self, key) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False


@qfrozen
class Instruction:
    ...


@qdefine
class Program:
    device: str


@qfrozen
class PlayInstruction(Instruction):
    waveform_index: int


@qfrozen
class DelayInstruction(Instruction):
    time: float


@qfrozen
class WaitTriggerInstruction(Instruction):
    device: str | None = None
    index: int = 0
    subchannel: int = 0


@qfrozen
class ReadInstruction(Instruction):
    samples: int
    sample_rate: float
    channel: tuple[int, ...]


@qfrozen
class ResetInstruction(Instruction):
    ...


@qdefine
class QuantumExecutable(metaclass=ABCMeta):
    """An abstract base class for hardware-specific executables."""

    sequence: Sequence | None = field(eq=id, default=None)

    @property
    def seq(self) -> Sequence | None:
        return self.sequence


@qdefine
class IntermediateProgram(Program):
    instructions: list[Instruction] = field(factory=list)
    markers: dict[tuple[int, int], list] = field(factory=dict)
    waveforms: list[WaveformMemory | None] = field(factory=dict)
    read_registers: set[int] = field(factory=set)

    def add_waveform(self, waveform: WaveformMemory) -> int:
        self.waveforms.append(waveform)

        return len(self.waveforms) - 1


@qdefine
class QWiPExecutable(QuantumExecutable):
    programs: dict[str, Program] = field(factory=dict)
    num_reads: list[int] = field(factory=list)
    reset_delay: float = 500e-6

    @classmethod
    def from_devices(cls, devices: Iterable[DeviceInfo], **kwargs) -> Self:
        programs = {cg.name: IntermediateProgram(device=cg.name) for cg in devices}

        return cls(programs=programs, **kwargs)

    @property
    def read_registers(self) -> set[int]:
        return {
            reg
            for reg in it.chain.from_iterable(
                prog.read_registers for prog in self.programs.values()
            )
        }


@qdefine
class HardwareCompiler:
    ...


@qdefine
class QWiPCompiler:
    """
    Attributes:
        channels: A mapping from device names to `DeviceInfo` instances that
            contain information about the channels.
        modulations: A mapping from modulation keys for phase tracking to concrete
            modulation frequencies.
        end_marker: A string specifying the marker name that is used to specify the
            end of a sequence element.
    """

    channels: dict[str, DeviceInfo] = field(factory=dict)
    modulations: dict[str, ModulationFrequency] = field(factory=dict)
    subcompilers: dict[str, HardwareCompiler] = field(factory=dict)

    @classmethod
    def from_devices(cls, devices: Iterable[DeviceInfo], **kwargs: Any) -> Self:
        """Contruct the waveform sequencer from a list of `DeviceInfo`.

        Args:
            devices: A list of `DeviceInfo` instances representing the
                measurement hardware.
            **kwargs: Remaining keyworad arguments are passed to the `__init__`
                function.

        Returns:
            A new WaveformSequencer instance.
        """
        channels = {dev.name: dev for dev in devices}

        return cls(channels=channels, **kwargs)

    def get_channel_info(self, name: str) -> ChannelInfo | None:
        """Returns the `ChannelInfo` with the given name.

        It is assumed that there are no repeated channel names between channel groups,
        so this method will short circuit on the first channel that matches the name.

        Args:
            name: The name of the channel to get.

        Returns:
            A `ChannelInfo` or `None`, if no channel matching the name exists.
        """
        for device in self.channels.values():
            try:
                return device[name]
            except KeyError:
                continue

        return None

    def compile_phases(self, locations: dict[Location, list[Waveform]]) -> PhaseTracker:
        """Returns a new phase tracker instance with all virtual phase updates.

        Every waveform that has an `update_phase_tracker` method will be called
        on the `PhaseTracker`.

        Args:
            locations: A dictionary mapping locations to waveforms.

        Returns:
            An updated phase tracker.
        """
        phase_tracker = PhaseTracker.from_modulations(self.modulations)

        for loc, waves in locations.items():
            loc = loc.offset

            for w in waves:
                if not isinstance(w, PhaseUpdater):
                    continue

                w.update_phase_tracker(loc, phase_tracker)

        return phase_tracker

    def compile_waveform(
        self,
        exe: QWiPExecutable,
        instructions: list[Instruction],
        start: int,
        end: int,
        wave: Waveform,
        device: DeviceInfo,
        location_kwargs: dict = {},
        pulse_kwargs: dict = {},
        instruction_cache: dict[tuple[int, str], list[Instruction]] = {},
    ) -> None:
        program = exe.programs[device.name]

        channels = [device[c] for c in wave.channels if c in device.channel_names()]
        read = np.any([ch.read for ch in channels])

        if read:
            instructions.append(
                ReadInstruction(
                    samples=end - start,
                    sample_rate=device.sample_rate,
                    channel=(reg := sorted(ch.index for ch in channels if ch.read)),
                )
            )
            program.read_registers.update(reg)

        match wave:
            case TriggeredWaveform():
                self.compile_sequence_element(
                    exe,
                    wave.target,
                    location_kwargs,
                    pulse_kwargs,
                    instruction_cache,
                )

                for ch in channels:
                    markers = program.markers.get((ch.index, ch.subchannel), [])
                    markers.append(start)
                    program.markers[(ch.index, ch.subchannel)] = markers

    def compile_single_timeline(
        self,
        exe: QWiPExecutable,
        locations: dict[Location, list[Waveform]],
        wmem: WaveformMemory,
        device: DeviceInfo,
        phase_tracker: PhaseTracker,
        location_kwargs: dict = {},
        pulse_kwargs: dict = {},
        instruction_cache: dict[tuple[int, str], list[Instruction]] = {},
    ) -> list[Instruction]:
        sample_rate = wmem.sample_rate
        samples = wmem.samples
        program = exe.programs[device.name]

        ts = np.arange(samples) / sample_rate

        instructions = []

        if device.has_output():
            wf_index = program.add_waveform(wmem)
            instructions.append(PlayInstruction(waveform_index=wf_index))

        for loc, waves in locations.items():
            for w in waves:
                if not (set(w.channels) & device.channel_names()):
                    continue

                width = w.width.resolve(**pulse_kwargs)
                start, end = loc.offset, loc.offset + width.offset

                s_idx = int(start * sample_rate)
                e_idx = samples if np.isinf(end) else int(end * sample_rate) + 1
                # Zero width
                if s_idx == e_idx - 1:
                    continue

                ts_wave = ts[s_idx:e_idx]

                w_t = w(
                    ts_wave,
                    t0=start + w.t0,
                    phase_tracker=phase_tracker,
                    modulations=self.modulations,
                    **pulse_kwargs,
                )

                if len(w_t.shape) == 1:
                    w_t = w_t[np.newaxis, :]

                for i, c in enumerate(w.channels):
                    try:
                        wmem[device[c]][s_idx:e_idx] += w_t[i]
                    except KeyError:
                        pass

                self.compile_waveform(
                    exe,
                    instructions,
                    s_idx,
                    e_idx,
                    w,
                    device,
                    location_kwargs,
                    pulse_kwargs,
                    instruction_cache,
                )

        return instructions

    def compile_sequence_element(
        self,
        exe: QWiPExecutable,
        se: SequenceElement,
        location_kwargs: dict = {},
        pulse_kwargs: dict = {},
        instruction_cache: dict[tuple[int, str], list[Instruction]] = {},
    ) -> None:
        """Compiles a single sequence elements.

        This method makes two passes through location waveform mapping. The
        first pass compiles all phase jumps and the second pass evaluates the
        pulse timepoints.

        Args:
            se: The sequence element to compile. The locations
                should be time ordered.
            waveform_array: A numpy array with shape `(channels, timepoints, subchannels)`
                that will hold the compiled timepoints
            device: The device that corresponds to this location
                map.
            pulse_kwargs: A mapping of variable names to resolved values to pass to
                all pulses.

        """
        markers = {}
        locations = se.resolve_locations(
            end_marker="end", markers=markers, **location_kwargs
        )

        # Compile phases
        phase_tracker = self.compile_phases(locations)
        t_end = markers["end"].offset

        for name, device in self.channels.items():
            program = exe.programs[name]
            sample_rate = device.sample_rate
            num_timepoints = int(t_end * sample_rate)

            # Skip compilation for device if no waveforms on channel
            if not (se.channels & device.channel_names()):
                continue

            if device.trigger:
                start = WaitTriggerInstruction(
                    device=device.trigger.device,
                    index=device.trigger.index,
                    subchannel=device.trigger.subchannel,
                )
            else:
                start = WaitTriggerInstruction()

            program.instructions.append(start)

            channels = (ch for ch in device.channels if ch.name in se.channels)

            if id(se) in instruction_cache:
                instructions = instruction_cache[id(se), device.name]
            else:
                wmem = WaveformMemory.from_channels(
                    num_timepoints, sample_rate, channels
                )
                instructions = self.compile_single_timeline(
                    exe,
                    locations,
                    wmem,
                    device,
                    phase_tracker,
                    location_kwargs,
                    pulse_kwargs,
                    instruction_cache,
                )
                instruction_cache[id(se), device.name] = instructions

            program.instructions.extend(instructions)
            exe.num_reads[-1] += sum(
                [isinstance(ins, ReadInstruction) for ins in instructions]
            )

    def compile(
        self,
        seq: Sequence,
        location_kwargs: dict = {},
        pulse_kwargs: dict = {},
    ) -> QWiPExecutable:
        """Compiles a sequence.

        This function takes an abstract sequence and compiles it into a concrete
        set of timepoints.

        Args:
            seq: The sequence to compile.
            location_kwargs: Any location constraints to add to the sequence
                before compilation.
            pulse_kwargs: A mapping of variables names to resolved pulse parameters.

        Returns:
            A `QWiPExecutable` instance.
        """
        exe = QWiPExecutable.from_devices(sequence=seq, devices=self.channels.values())

        instruction_cache = dict()

        for se in seq.flat:
            exe.num_reads.append(0)
            self.compile_sequence_element(
                exe, se, location_kwargs, pulse_kwargs, instruction_cache
            )

        for dev, program in exe.programs.items():
            if dev in self.subcompilers:
                exe.programs[dev] = self.subcompilers[dev].compile(
                    program, device=self.channels[dev]
                )

        return exe


# @qdefine
# class WaveformSequencer:
#     """A waveform sequencer.

#     The waveform sequencer is responsible for compiling sequences into concrete timepoints
#     that can then be uploaded to the measurement hardware (DAC/ADC).

#     Attributes:
#         channels: A mapping from device names to `DeviceInfo` instances that
#             contain information about the channels.
#         modulations: A mapping from modulation keys for phase tracking to concrete
#             modulation frequencies.
#         readout_qubits: The list of qubits that should be included in the hardware
#             demodulation weights that are uploaded to the ADC. This is a legacy
#             parameter necessary for the ZI UHFQA's.
#         end_marker: A string specifying the marker name that is used to specify the
#             end of a sequence element.
#     """

#     channels: dict[str, DeviceInfo] = field(factory=dict)
#     modulations: dict[str, ModulationFrequency] = field(factory=dict)
#     readout_qubits: list[int] = field(factory=list)
#     end_marker: str = "end"

#     @classmethod
#     def from_devices(cls, devices: Iterable[DeviceInfo], **kwargs: Any) -> Self:
#         """Contruct the waveform sequencer from a list of `DeviceInfo`.

#         Args:
#             devices: A list of `DeviceInfo` instances representing the
#                 measurement hardware.
#             **kwargs: Remaining keyworad arguments are passed to the `__init__`
#                 function.

#         Returns:
#             A new WaveformSequencer instance.
#         """
#         channels = {device.name: device for device in devices}

#         return cls(channels=channels, **kwargs)

#     def get_channel_info(self, name: str) -> ChannelInfo | None:
#         """Returns the `ChannelInfo` with the given name.

#         It is assumed that there are no repeated channel names between devices,
#         so this method will short circuit on the first channel that matches the name.

#         Args:
#             name: The name of the channel to get.

#         Returns:
#             A `ChannelInfo` or `None`, if no channel matching the name exists.
#         """
#         for device in self.channels.values():
#             try:
#                 return device[name]
#             except KeyError:
#                 continue

#         return None

#     def compile_phases(self, locations: dict[Location, list[Waveform]]) -> PhaseTracker:
#         """Returns a new phase tracker instance with all virtual phase updates.

#         Every waveform that has an `update_phase_tracker` method will be called
#         on the `PhaseTracker`.

#         Args:
#             locations: A dictionary mapping locations to waveforms.

#         Returns:
#             An updated phase tracker.
#         """
#         phase_tracker = PhaseTracker.from_modulations(self.modulations)

#         for loc, waves in locations.items():
#             loc = loc.offset

#             for w in waves:
#                 if not isinstance(w, PhaseUpdater):
#                     continue

#                 w.update_phase_tracker(loc, phase_tracker)

#         return phase_tracker

#     def compile_timepoints(
#         self,
#         locations: dict[Location, list[Waveform]],
#         waveform_array: NDArray[np.float32],
#         device: DeviceInfo,
#         phase_tracker: PhaseTracker,
#         pulse_kwargs: dict = {},
#     ) -> None:
#         """Compiles a single timeline of pulses into concrete timepoints.

#         Each element represents a DAC amplitude (normalized between -1 and 1)
#         on a specific channel/subchannel for a given sample timestep.

#         Args:
#             locations: A dictionary mapping locations to waveforms. The locations
#                 should be time ordered.
#             waveform_array: A numpy array with shape `(channels, timepoints, subchannels)`
#                 that will hold the compiled timepoints
#             device: The device that corresponds to this location
#                 map.
#             phase_tracker: A phase tracker instance that holds all phase jumps for
#                 this timeline of pulses.
#             pulse_kwargs: A mapping of variable names to resolved values to pass to
#                 all pulses.
#         """
#         sample_rate = device.sample_rate
#         num_timepoints = waveform_array.shape[1]

#         ts = np.arange(num_timepoints) / sample_rate
#         logger.debug(f"Time array shape: {ts.shape}.")

#         for loc, waves in locations.items():
#             for w in waves:
#                 width = w.width.resolve(**pulse_kwargs)
#                 start, end = loc.offset, loc.offset + width.offset

#                 s_idx = int(start * sample_rate)
#                 e_idx = num_timepoints if np.isinf(end) else int(end * sample_rate) + 1
#                 if s_idx == e_idx - 1:
#                     continue

#                 ts_wave = ts[s_idx:e_idx]

#                 w_t = w(
#                     ts_wave,
#                     t0=start + w.t0,
#                     phase_tracker=phase_tracker,
#                     modulations=self.modulations,
#                     **pulse_kwargs,
#                 )

#                 if len(w_t.shape) == 1:
#                     w_t = w_t[np.newaxis, :]

#                 for i, c in enumerate(w.channels):
#                     ch_idx = (device[c].index,)
#                     subchannel = device[c].subchannel
#                     waveform_array[ch_idx, s_idx:e_idx, subchannel] += w_t[i]

#         return waveform_array

#     def compile_sequence_element(
#         self,
#         locations: dict[Location, list[Waveform]],
#         waveform_array: np.ndarray,
#         device: DeviceInfo,
#         pulse_kwargs: dict = {},
#     ):
#         """Compiles a single sequence elements.

#         This method makes two passes through location waveform mapping. The
#         first pass compiles all phase jumps and the second pass evaluates the
#         pulse timepoints.

#         Args:
#             locations: A dictionary mapping locations to waveforms. The locations
#                 should be time ordered.
#             waveform_array: A numpy array with shape `(channels, timepoints, subchannels)`
#                 that will hold the compiled timepoints
#             device: The device that corresponds to this location map.
#             pulse_kwargs: A mapping of variable names to resolved values to pass to
#                 all pulses.

#         """
#         # Compile phases
#         phase_tracker = self.compile_phases(locations)

#         return self.compile_timepoints(
#             locations=locations,
#             waveform_array=waveform_array,
#             device=device,
#             phase_tracker=phase_tracker,
#             pulse_kwargs=pulse_kwargs,
#         )

#     def initialize_compiled_sequence(self, seq: Sequence, max_times: dict[str, float]):
#         """Creates a new `CompiledSequence` instance.

#         This function allocates the waveform arrays that will hold all the waveform
#         data for the given sequence.

#         Args:
#             seq: The sequence to be compiled.
#             max_times: A dictionary mapping device keys to the latest timepoint
#                 played on any channel accross all sequence elements. Times are specified
#                 in seconds.

#         Returns:
#             A new `CompiledSequence` instance with the waveform data arrays initialized
#             to all zeros.
#         """

#         waveform_arrs = dict()
#         for key, dev in self.channels.items():
#             sample_rate = dev.sample_rate
#             num_channels = dev.max_channel_index + 1
#             num_subchannels = dev.max_subchannel_index + 1

#             num_elements = np.prod(seq.shape) if key == "seq" else 1
#             num_timepoints = int(max_times[key].offset * sample_rate)

#             # We put num_subchannels as the first index and then transpose in an
#             # attempt to make memory layout more sensible.
#             arr_shape = (num_subchannels, num_channels, num_elements, num_timepoints)
#             logger.debug(f"Device {key} shape: {arr_shape}.")

#             waveform_arrs[key] = WaveformData(
#                 sample_rate=sample_rate,
#                 n_elements=num_elements,
#                 num_channels=num_channels,
#                 # This should return a view of the array
#                 array=np.zeros(arr_shape, dtype=np.float32).transpose(1, 2, 3, 0),
#             )
#             logger.debug(f"dtype: {waveform_arrs[key].array.dtype}")

#         return CompiledSequence(waveforms=waveform_arrs, sequence=seq)

#     def compile(
#         self,
#         seq: Sequence,
#         location_kwargs: dict = {},
#         pulse_kwargs: dict = {},
#         **triggered_elements,
#     ):
#         """Compiles a sequence.

#         This function takes an abstract sequence and compiles it into a concrete
#         set of timepoints.

#         Args:
#             seq: The sequence to compile.
#             location_kwargs: Any location constraints to add to the sequence
#                 before compilation.
#             pulse_kwargs: A mapping of variables names to resolved pulse parameters.

#         Returns:
#             A `CompiledSequence` instance.
#         """
#         # First resolve all locations in the main sequence
#         locations = [
#             se.resolve_locations(end_marker=self.end_marker, **location_kwargs)
#             for se in seq.flat
#         ]

#         # Then for any triggered sequence elements (most commonly readout)
#         triggered_locations = {
#             k: se.resolve_locations(end_marker=self.end_marker, **location_kwargs)
#             for k, se in triggered_elements.items()
#         }

#         # The above step was necessary to determine the number of timepoints
#         # in the waveform array
#         max_times = dict(seq=0) | {
#             k: find_end_marker(se_locs, self.end_marker)
#             for k, se_locs in triggered_locations.items()
#         }

#         max_times["seq"] = max(
#             find_end_marker(se_locs, self.end_marker) for se_locs in locations
#         )
#         logger.debug(max_times)

#         # Then use the collected information to initialize an empty CompiledSequence
#         # with the correct sizes for all waveform data arrays
#         cseq = self.initialize_compiled_sequence(seq, max_times)

#         # Now we move on to actually compiling timepoints and writing them to the
#         # waveform data arrays
#         for i, (se_locs, se) in enumerate(zip(locations, seq.flat)):
#             logger.debug(cseq.waveforms["seq"].sample_rate)
#             logger.debug(se.constraints | pulse_kwargs)
#             self.compile_sequence_element(
#                 se_locs,
#                 # (channel_idx, element_idx, timepoints, num_outports)
#                 cseq.waveforms["seq"].array[:, i, :],
#                 self.channels["seq"],
#                 se.constraints | pulse_kwargs,
#             )

#         # Then do the same for triggered sequence_elements
#         for trigger, se_locs in triggered_locations.items():
#             se = triggered_elements[trigger]
#             self.compile_sequence_element(
#                 se_locs,
#                 # (channel_idx, element_idx, timepoints, num_outports)
#                 cseq.waveforms[trigger].array[:, 0, :],
#                 self.channels[trigger],
#                 se.constraints | pulse_kwargs,
#             )

#         # Finally we pull out all the readout locations in the sequence
#         for i, se_locs in enumerate(locations):
#             sample_rate = cseq.waveforms["seq"].sample_rate
#             rloc = int(find_readout_marker(se_locs).offset * sample_rate)
#             cseq.get_readout_locations()[i] = rloc

#         # This is needed for backwards compatibility.
#         rinfo = _ReadoutInfo(
#             cseq._readout, self.readout_qubits, len(cseq.get_readout_locations())
#         )
#         cseq._readout._readout = rinfo

#         return cseq


register_compiler(QWiPCompiler)


@qdefine
class CompiledSequencePlotter:
    axsize: tuple[float, float] = (8, 1)

    def make_axes(
        self,
        n: int,
        axsize: tuple[float, float] | None = None,
        sharex: bool = True,
        sharey: bool = True,
        **props,
    ) -> Figure:
        """Creates a matplotlib figure and axes.

        Args:
            n: Number of axes.
            axsize: The size (width, height) in inc
        """
        if "figsize" not in props:
            axsize = axsize or self.axsize
            width, height = axsize

            if width == height == ...:
                width, height = (8, 1)
            elif width is ...:
                width = 8 / height
            elif height is ...:
                height = 1 / 8 * width

            props["figsize"] = (width, n * height)

        fig, _ = plt.subplots(n, 1, sharex=sharex, sharey=sharey, **props)

        return fig

    def plot(
        self,
        cseq,
        element: int,
        title: str,
        channels: list[tuple[int, ...]] | None = None,
        axes: Collection[Axes] | None = None,
        fig_props: dict = {},
    ) -> Figure:
        tdict = {
            name: np.arange(waveformdata.array.shape[2]) / waveformdata.sample_rate
            for name, waveformdata in cseq.waveforms.items()
        }

        mainseq = cseq.waveforms["seq"]
        ts = tdict["seq"]

        if channels is None:
            channels = [tuple(ch for ch in range(mainseq.array.shape[0]))]

        if axes is None:
            fig = self.make_axes(len(channels) + 1, **fig_props)
            axes = fig.axes

            if isinstance(axes, Axes):
                axes = np.array([axes])

        for ax_id, dev in enumerate(channels):
            ax = axes[ax_id]

            for ch in dev:
                for marker in range(mainseq.array.shape[3]):
                    pts = mainseq.array[ch, element, :, marker]
                    if not pts.any():
                        continue

                    ax.plot(ts, pts, label=f"CH{ch}", color=f"C{ch}")

        readoutseq = cseq.waveforms["readout"]

        treadout = mainseq.get_readout_locations()[element] / mainseq.sample_rate
        for ch in range(readoutseq.array.shape[0]):
            pts = readoutseq.array[ch, 0, :, 0]
            props = dict(color=f"C{ch + mainseq.shape[0]}", label=f"Readout CH{ch}")
            axes[-1].plot(tdict["readout"] + treadout, pts, **props)

        figwidth, _ = fig.get_size_inches()

        h, l = all_legend_handles_labels(axes)
        axes[0].legend(
            h,
            l,
            mode="expand",
            bbox_to_anchor=(0, 1.05, 1, 0.05),
            loc="lower left",
            ncols=min(figwidth // 2, len(l)),
            borderaxespad=0,
        )
        axes[0].set_ylim(-1, 1)

        axes[-1].set_xlabel("Time (s)")
        return fig


@qdefine
class InteractiveSequencePlotter:
    def plot(
        self,
        cseq,
        title: str = "Pulse Sequence Simulation",
        channels: list[tuple[int, ...]] | None = None,
        axes: Collection[Axes] | None = None,
        fig_props: dict = {},
    ) -> Figure:
        fig = go.Figure()

        ts_pulse = (
            np.arange(cseq.waveforms["seq"].array.shape[2])
            / cseq.waveforms["seq"].sample_rate
        )
        N_channels, N_elements, N_steps, N_subchannels = cseq.waveforms[
            "seq"
        ].array.shape

        active_elements = dict()
        num_traces = 0

        for i in range(N_elements):
            active_channels = np.where(
                np.any(
                    cseq.waveforms["seq"].array[:, i, :, 0].reshape(N_channels, -1),
                    axis=1,
                )
            )[0]

            if len(active_channels) != 0:
                active_elements[i] = active_channels

            for ch in active_channels:
                fig.add_trace(
                    go.Scatter(
                        x=ts_pulse,
                        y=cseq.waveforms["seq"].array[ch, i, :, 0],
                        visible=False,
                        name=f"CH {ch}",
                    )
                )
                num_traces += 1

        # Slider to filter Sequence Element
        steps_seq, start = [], 0
        last_element = 0

        for index, (element, targets) in enumerate(active_elements.items()):
            visible_seq = [False] * num_traces
            end = start + len(targets)

            visible_seq[start:end] = [True] * (end - start)
            steps_seq.append(
                dict(
                    label=f"{element}", method="update", args=[{"visible": visible_seq}]
                )
            )
            if index == len(active_elements.items()) - 1:
                for i in range(start, end):
                    fig.data[i].visible = True
                last_element = index

            start = end

        sliders = [
            dict(
                active=last_element,
                currentvalue={"prefix": "Sequence Element: "},
                pad={"t": 50, "b": 50},
                steps=steps_seq,
                borderwidth=2,
            )
        ]

        fig.update_layout(
            sliders=sliders,
            dragmode="pan",
            title={"text": title, "x": 0.5, "xanchor": "center"},
            yaxis_title="Amplitude",
            xaxis_title="Time",
            width=1000,
            height=600,
            autosize=False,
            margin=dict(t=50, b=0, l=0, r=0),
        )

        fig.update_yaxes(fixedrange=True)

        config = {"scrollZoom": True}
        fig.show(config=config)

        return fig


__all__ = [
    "ChannelInfo",
    "DeviceInfo",
    "CompiledSequencePlotter",
    "InteractiveSequencePlotter",
    "WaveformMemory",
    "Program",
    "register_compiler",
    "Instruction",
    "IntermediateProgram",
    "PlayInstruction",
    "DelayInstruction",
    "WaitTriggerInstruction",
    "ReadInstruction",
    "ResetInstruction",
]
