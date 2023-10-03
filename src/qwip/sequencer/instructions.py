from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
from attrs import field
from typing_extensions import Self

from qwip.attrs import qdefine, qfrozen
from qwip.sequencer.compilation import ChannelInfo, DeviceInfo, QuantumExecutable
from qwip.sequencer.elements import SequenceElement
from qwip.sequencer.phase_tracker import ModulationFrequency, PhaseTracker, PhaseUpdater
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.utils import Location
from qwip.sequencer.waveform import TriggeredWaveform, Waveform


@qdefine
class WaveformMemory:
    samples: int
    sample_rate: float
    data: dict[tuple[int, int], np.ndarray] = field(repr=lambda x: repr(x.keys()))

    @classmethod
    def from_channels(
        cls, samples: int, sample_rate: float, channels: Iterable[ChannelInfo]
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


@qdefine
class IntermediateProgram(Program):
    instructions: list[Instruction] = field(factory=list)
    markers: dict[tuple[int, int], list] = field(factory=dict)
    waveforms: list[WaveformMemory | None] = field(factory=dict)

    def add_waveform(self, waveform: WaveformMemory) -> int:
        self.waveforms.append(waveform)

        return len(self.waveforms) - 1


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
class QWiPExecutable(QuantumExecutable):
    programs: dict[str, Program] = field(factory=dict)
    num_reads: list[int] = field(factory=list)
    reset_delay: float = 500e-6

    @classmethod
    def from_devices(cls, devices: Iterable[DeviceInfo], **kwargs) -> Self:
        programs = {cg.name: IntermediateProgram(device=cg.name) for cg in devices}

        return cls(programs=programs, **kwargs)


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
                    channel=sorted(ch.index for ch in channels),
                )
            )

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
