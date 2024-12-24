import itertools as it
from abc import ABCMeta
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Collection, Iterable
from typing import Any, Self

import cattr
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import sympy as sym
from attrs import evolve, field
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq, fftshift

import qwip
from qwip._cattr import make_attrs_structure_fn, make_attrs_unstructure_fn
from qwip.attrs import qdefine, qfrozen
from qwip.sequencer.phase_tracker import Frame, PhaseTracker, PhaseUpdater
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.timeline import Timeline
from qwip.sequencer.utils import _to_python_number
from qwip.sequencer.waveform import Marker, ReadoutMarker, TriggeredWaveform, Waveform
from qwip.utils import deprecated
from qwip.visualization.utils import all_legend_handles_labels

REGISTERED_COMPILERS: dict[str, "QWiPCompiler"] = dict()


def register_compiler(cls: type["QWiPCompiler"]) -> type["QWiPCompiler"]:
    if not issubclass(cls, QWiPCompiler):
        raise TypeError(f"Registered sequencer must subclass {QWiPCompiler}")

    REGISTERED_COMPILERS[cls.__name__] = cls

    return cls


# def find_end_marker(locations, name="end") -> "Location | None":
#     for loc, waves in locations.items():
#         if Marker(name=name) in waves:
#             return loc

#     return None


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
    envelope_sample_rate: float | None = None
    dtype: type = np.float32
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
        cls,
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

        return cls(name=device, channels=channels, sample_rate=sample_rate, **kwargs)

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
        cls,
        samples: int,
        sample_rate: float,
        channels: Iterable["ChannelInfo"],
        dtype=np.float32,
    ) -> Self:
        data = {
            (ch.index, ch.subchannel): np.zeros(samples, dtype=dtype) for ch in channels
        }
        return cls(samples=samples, sample_rate=sample_rate, data=data)

    def __getitem__(self, key) -> np.ndarray:
        match key:
            case int():
                return self.data[key, 0]
            case ChannelInfo(index=idx, subchannel=sc):
                return self.data[idx, sc]
            case _:
                return self.data[key]

    def __contains__(self, key) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False


@qfrozen
class Instruction: ...


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
class ResetInstruction(Instruction): ...


@qdefine
class QuantumExecutable(metaclass=ABCMeta):
    """An abstract base class for hardware-specific executables."""

    sequence: Sequence | None = field(eq=id, default=None)
    timeline_index: int = 0
    repetition_index: int = 0

    @property
    def seq(self) -> Sequence | None:
        return self.sequence


def make_executable_structure_fn(cls):
    structure_attrs = make_attrs_structure_fn(cls)

    def structure_fn(val, cls):
        if isinstance(val, cls):
            return val

        try:
            subclass = qwip.converter.structure(val.get("__class__"), type)
        except NameError:
            subclass = QuantumExecutable
            logger.warning(
                f"No executable subclass found. Attempting to structure {val} as {cls}."
            )

        if subclass is QuantumExecutable:
            return structure_attrs(val, cls)

        return qwip.converter.structure(val, subclass)

    return structure_fn


def make_executable_unstructure_fn(cls):
    overrides = {"sequence": cattr.override(omit_if_default=False, omit=False)}
    unstructure_attrs = make_attrs_unstructure_fn(cls, overrides=overrides)

    def unstructure_fn(obj):
        unstructured = {
            **unstructure_attrs(obj),
            "__class__": qwip.converter.unstructure(type(obj)),
        }
        if unstructured["sequence"] is None:
            unstructured.pop("sequence")

        return unstructured

    return unstructure_fn


qwip.converter.register_structure_hook_factory(
    lambda cls: cls is QuantumExecutable, make_executable_structure_fn
)

qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, QuantumExecutable), make_executable_unstructure_fn
)


@qfrozen
class BatchedExecutable:
    exe: QuantumExecutable = field(eq=id)
    repetitions: int
    timeline_index: int = 0
    repetition_index: int = 0
    batch_index: tuple[int, int] = (0, 0)
    upload: bool = True


@qdefine
class IntermediateProgram(Program):
    instructions: list[Instruction] = field(factory=list)
    waveforms: list[WaveformMemory | None] = field(factory=dict)
    markers: list[list[tuple[tuple[int, int], float]]] = field(factory=list)
    trigger: TriggerInfo | None = None
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
        programs = {
            dev.name: IntermediateProgram(device=dev.name, trigger=dev.trigger)
            for dev in devices
        }

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
class HardwareCompiler: ...


@qdefine
class QWiPCompiler:
    """
    Attributes:
        devices: A mapping from device names to `DeviceInfo` instances that
            contain information about the channels.
        frames: A mapping from named frames for phase tracking to frames with a numeric
            frequency value.
        subcompilers: A mapping of subcompilers to compile intermediate programs down
            to their hardware-specific representation.
    """

    devices: dict[str, DeviceInfo] = field(factory=dict)
    frames: dict[str, Frame] = field(factory=dict)
    subcompilers: dict[str, HardwareCompiler] = field(factory=dict)

    @property
    @deprecated(
        version="23.10.0",
        removed="23.12.0",
        message="Use `compiler.frames` instead.",
    )
    def modulations(self) -> dict[str, Frame]:
        return self.frames

    @property
    @deprecated(
        version="23.11.0", removed="23.12.0", message="Use `compiler.devices` instead."
    )
    def channels(self) -> dict[str, DeviceInfo]:
        return self.devices

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
        devices = {dev.name: dev for dev in devices}

        return cls(devices=devices, **kwargs)

    def get_channel_info(self, name: str) -> ChannelInfo | None:
        """Returns the `ChannelInfo` with the given name.

        It is assumed that there are no repeated channel names between channel groups,
        so this method will short circuit on the first channel that matches the name.

        Args:
            name: The name of the channel to get.

        Returns:
            A `ChannelInfo` or `None`, if no channel matching the name exists.
        """
        for device in self.devices.values():
            try:
                return device[name]
            except KeyError:
                continue

        return None

    def compile_phases(
        self, locations: Iterable[tuple[float, Waveform]], /
    ) -> PhaseTracker:
        """Returns a new phase tracker instance with all virtual phase updates.

        Every waveform that has an `update_phase_tracker` method will be called
        on the `PhaseTracker`.

        Args:
            locations: An iterable of `(location, waveform)` tuples.

        Returns:
            An updated phase tracker.
        """
        phase_tracker = PhaseTracker.from_frames(self.frames)

        for loc, op in locations:
            loc = _to_python_number(loc)

            if not isinstance(op, PhaseUpdater):
                continue

            op.update_phase_tracker(loc, phase_tracker, self.frames)

        return phase_tracker

    def compile_instruction(
        self,
        exe: QWiPExecutable,
        instructions: list[Instruction],
        start: int,
        end: int,
        wave: Waveform,
        device: DeviceInfo,
        instruction_cache: dict[tuple[int, str], list[Instruction]] = {},
    ) -> None:
        program = exe.programs[device.name]

        ch_info = device[wave.channel]

        if ch_info.read:
            instructions.append(
                ReadInstruction(
                    samples=end - start,
                    sample_rate=device.sample_rate,
                    channel=(ch_info.index,),
                )
            )
            program.read_registers.add(ch_info.index)

        match wave:
            case TriggeredWaveform():
                self.compile_timeline(
                    exe,
                    wave.target,
                    instruction_cache=instruction_cache,
                )

                program.markers[-1].append(
                    ((ch_info.index, ch_info.subchannel), start / device.sample_rate)
                )

    def compile_waveforms(
        self,
        exe: QWiPExecutable,
        tmln: Timeline,
        wmem: WaveformMemory,
        device: DeviceInfo,
        phase_tracker: PhaseTracker,
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

        for loc, w in tmln:
            loc = _to_python_number(loc)
            if w.channel not in device.channel_names():
                continue

            width = w.width
            start, end = loc, loc + width

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
                frames=self.frames,
            )

            if issubclass(wmem[device[w.channel]].dtype.type, np.floating):
                w_t = w_t.real

            wmem[device[w.channel]][s_idx:e_idx] += w_t

            self.compile_instruction(
                exe,
                instructions,
                s_idx,
                e_idx,
                w,
                device,
                instruction_cache,
            )

        return instructions

    def compile_timeline(
        self,
        exe: QWiPExecutable,
        tmln: Timeline,
        substitutions: dict = {},
        instruction_cache: dict[tuple[int, str], list[Instruction]] = {},
    ) -> None:
        """Compiles a single pulse timelines.

        This method makes two passes through location waveform mapping. The
        first pass compiles all phase jumps and the second pass evaluates the
        pulse timepoints.

        Args:
            exe: The resulting executable.
            tmln: The timeline to compile.
            substitutions: A dictionary mapping variables to substitutions that get
                passed to `Timeline.resolve`.
            instruction_cache: The instruction cache.
        """
        tmln.resolve(inplace=True, **substitutions)

        # Compile phases
        phase_tracker = self.compile_phases(tmln)
        t_end = _to_python_number(tmln.width)

        for name, device in self.devices.items():
            program = exe.programs[name]
            sample_rate = device.sample_rate
            num_timepoints = int(t_end * sample_rate)

            # Skip compilation for device if no waveforms on channel
            if not (tmln.channels & device.channel_names()):
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
            program.markers.append([])

            channels = (ch for ch in device.channels if ch.name in tmln.channels)

            if (id(tmln), device.name) in instruction_cache:
                instructions = instruction_cache[id(tmln), device.name]
            else:
                wmem = WaveformMemory.from_channels(
                    num_timepoints, sample_rate, channels, dtype=device.dtype
                )
                instructions = self.compile_waveforms(
                    exe,
                    tmln,
                    wmem,
                    device,
                    phase_tracker,
                    instruction_cache,
                )
                instruction_cache[id(tmln), device.name] = instructions

            program.instructions.extend(instructions)
            exe.num_reads[-1] += sum(
                [isinstance(ins, ReadInstruction) for ins in instructions]
            )

    def compile(
        self,
        seq: Sequence,
        batch_size: int | None = None,
        substitutions: dict = {},
    ) -> QWiPExecutable:
        """Compiles a sequence.

        This function takes an abstract sequence and compiles it into a concrete
        set of timepoints.

        Args:
            seq: The sequence to compile.
            substitutions: A dictionary mapping variables to substitutions that get
                passed to `Timeline.resolve`.

        Returns:
            A `QWiPExecutable` instance.
        """
        exes = []

        instruction_cache = dict()

        num_timelines = len(seq.flat)
        batch_size = batch_size or num_timelines

        flattened_seq = seq.flatten()
        for tmln_idx in range(0, num_timelines, batch_size):
            exe = QWiPExecutable.from_devices(
                sequence=None, devices=self.devices.values(), timeline_index=tmln_idx
            )

            exe.sequence = batch_seq = flattened_seq[tmln_idx : tmln_idx + batch_size]
            for tmln in batch_seq:
                exe.num_reads.append(0)
                self.compile_timeline(exe, tmln, substitutions, instruction_cache)

            for dev, program in exe.programs.items():
                if dev in self.subcompilers:
                    exe.programs[dev] = self.subcompilers[dev].compile(
                        program, device=self.devices[dev]
                    )

            exes.append(exe)

        return exes


register_compiler(QWiPCompiler)


@qdefine
class QWiPExePlotter:
    def program_order(self, exe: QWiPExecutable) -> list[IntermediateProgram]:
        """Orders the programs based on their trigger dependencies."""
        programs = deque(exe.programs.values())

        ordered = []
        visited = set()

        while programs:
            prog = programs.popleft()

            if prog.device in visited:
                raise ValueError("Trigger cycle detected!")
            elif prog.trigger is None or prog.trigger.device in ordered:
                ordered.append(prog.device)
                visited = set()
            else:
                visited.add(prog.device)
                programs.append(prog)

        return ordered

    def marker_timestamps(self, exe: QWiPExecutable) -> dict[TriggerInfo, list[float]]:
        timestamps = defaultdict(list)
        ordered = self.program_order(exe)

        for device in ordered:
            program = exe.programs[device]
            for trig_id, markers in enumerate(program.markers):
                if program.trigger:
                    t0 = timestamps[program.trigger][trig_id]
                else:
                    t0 = exe.reset_delay * trig_id

                for (ch, subch), t in markers:
                    timestamps[
                        TriggerInfo(device=program.device, index=ch, subchannel=subch)
                    ].append(t + t0)

        return timestamps

    def plot_trigger(
        self,
        fig,
        exe: QWiPExecutable,
        ins: Instruction,
        time: float,
        trigger_counter: Counter,
        marker_timestamps: dict[TriggerInfo, list[float]],
        **kwargs,
    ) -> float:
        if ins.device is None:
            t0 = exe.reset_delay * trigger_counter[None]
            trigger_counter[None] += 1

            fig.add_vline(x=t0)
        else:
            tinfo = TriggerInfo(
                device=ins.device, index=ins.index, subchannel=ins.subchannel
            )
            t0 = marker_timestamps[tinfo][trigger_counter[tinfo]]
            trigger_counter[tinfo] += 1

        return t0

    def plot_play(
        self,
        fig,
        exe: QWiPExecutable,
        ins: Instruction,
        time: float,
        device: str,
        colors: dict[str, int],
        timelines: list[int] | None = None,
        **kwargs,
    ) -> float:
        wmem = exe.programs[device].waveforms[ins.waveform_index]
        ts = np.arange(wmem.samples) / wmem.sample_rate + time

        tmln_index = int(time // exe.reset_delay)
        if timelines is None or tmln_index in timelines:
            for (ch, subch), arr in wmem.data.items():
                label = f"{device} - CH{ch}" + (f"- {subch}" if subch else "")
                showlegend = label not in colors
                colors[label] = colors.get(label, len(colors))

                template_colors = fig.layout.template.layout.colorway
                c = template_colors[colors[label] % len(template_colors)]
                fig.add_trace(
                    go.Scatter(
                        x=ts,
                        y=arr,
                        name=label,
                        showlegend=showlegend,
                        legendgroup=label,
                        mode="lines",
                        marker_color=c,
                    )
                )

        return time + wmem.samples / wmem.sample_rate

    def make_dropdown(
        self,
        fig,
        exe: QWiPExecutable,
        timeline_end_times,
        timelines: list[int] | None = None,
    ):
        timelines = timelines or range(len(exe.num_reads))
        buttons = [
            dict(
                args=["xaxis", dict(range=[None, None])],
                label="Show all",
                method="relayout",
            )
        ] + [
            dict(
                args=[
                    "xaxis",
                    dict(range=[tmln * exe.reset_delay, timeline_end_times[tmln]]),
                ],
                label=f"Timeline {tmln}",
                method="relayout",
            )
            for tmln in timelines
        ]

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    x=0,
                    xanchor="left",
                    y=1.02,
                    yanchor="bottom",
                )
            ],
            margin=dict(t=0, b=0, l=0, r=0),
        )

    def plot(self, exe: QWiPExecutable, timelines: list[int] | None = None):
        fig = go.Figure()
        fig.update_yaxes(fixedrange=True)

        colors = {}
        marker_timestamps = self.marker_timestamps(exe)
        timeline_end_times = []

        for dev, prog in exe.programs.items():
            trigger_counter = Counter()
            time = 0
            for ins in prog.instructions:
                match ins:
                    case WaitTriggerInstruction():
                        time = self.plot_trigger(
                            fig, exe, ins, time, trigger_counter, marker_timestamps
                        )
                    case PlayInstruction():
                        time = self.plot_play(
                            fig, exe, ins, time, dev, colors, timelines
                        )

                tmln = int(time // exe.reset_delay)
                try:
                    timeline_end_times[tmln] = max(time, timeline_end_times[tmln])
                except IndexError:
                    timeline_end_times.append(time)

        self.make_dropdown(fig, exe, timeline_end_times, timelines)
        fig.update_layout(
            yaxis_title="Amplitude",
            xaxis_title="Time (s)",
        )

        fig.show(config=dict(scrollZoom=True))
        return fig


__all__ = [
    "ChannelInfo",
    "DeviceInfo",
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
    "QWiPExePlotter",
]
