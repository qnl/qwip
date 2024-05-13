from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from attrs import field
from loguru import logger

try:
    import qubic.toolchain as tc
    from distproc.compiler import CompiledProgram
    from distproc.compiler import Compiler as _QubicInternalCompiler
    from distproc.hwconfig import FPGAConfig
    from distproc.ir import passes
    from distproc.ir.instructions import Pulse, VirtualZ
    from qubic.rpc_client import CircuitRunnerClient
    from qubitconfig.qchip import QChip
except ImportError as e:
    logger.warning("Unable to import qubic dependencies.")
    raise e

from qwip.attrs import qdefine, qfrozen
from qwip.backends.backend import QuantumBackend
from qwip.flatdict import FlatDict
from qwip.processing.processors import IQResult
from qwip.sequencer.compilation import (
    QuantumExecutable,
    QWiPCompiler,
    register_compiler,
)
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.utils import Location
from qwip.sequencer.waveform import (
    BasicWaveform,
    Marker,
    ModulatedWaveform,
    VirtualZWaveform,
    Waveform,
)

if TYPE_CHECKING:
    from qwip.qpu.qpu import QPU


def get_compiler_passes(
    fpga_config: FPGAConfig,
    qchip: QChip,
    qubit_grouping: tuple[str, ...] = ("{qubit}.qdrv", "{qubit}.rdrv", "{qubit}.rdlo"),
    proc_grouping: list[tuple[str, ...]] = [
        ("{qubit}.qdrv", "{qubit}.rdrv", "{qubit}.rdlo")
    ],
):
    """Constructs the default compiler passes for the internal Qubic compiler.

    We do the scheduling already before converting to the expected format for Qubic, so
    we skip the scheduling pass in the Qubic compiler. This allows us to work with
    explicit timings. The compiler passes are defined in `distproc.ir`.

    Args:
        fgpa_config: A qubic FPGAConfig
        qchip: A qchip instance.
        qubit_grouping: A grouping of qubits to channels.
        proc_grouping: ...

    Returns:
        A list of compiler of passes for Qubic's internal compiler.
    """
    return [
        passes.FlattenProgram(),
        passes.MakeBasicBlocks(),
        passes.ScopeProgram(qubit_grouping),
        passes.RegisterVarsAndFreqs(qchip),
        passes.ResolveGates(qchip, qubit_grouping),
        passes.GenerateCFG(),
        passes.ResolveHWVirtualZ(),
        passes.ResolveVirtualZ(),
        passes.ResolveFreqs(),
        passes.ResolveFPROCChannels(fpga_config),
    ]


@qfrozen
class QubicChannelConfig:
    """A Qubic channel config object.

    Attributes:
        device: The device that the channel belongs to. Should be one of
            `[qdrv, rdrv, rdlo]`.
        core_ind: The core index for the channel.
        elem_ind: The element index for the channel.
        elem_params: A dictionary with the clock samples and interpolation ratio.
        env_mem_name: The name of the envelope memory for the channel.
        freq_mem_name: The name of the frequency memory for the channel.
        acc_mem_name: The name of the acc buffer for the channel.
    """

    device: str
    core_ind: int
    elem_ind: int = 0
    elem_params: dict[str, int] = dict(samples_per_clk=16, interp_ratio=1)
    env_mem_name: str = field()
    freq_mem_name: str = field()
    acc_mem_name: str = field()

    @env_mem_name.default
    def _default_env_mem_name(self) -> str:
        return f"{self.device}env{self.core_ind}"

    @freq_mem_name.default
    def _default_freq_mem_name(self) -> str:
        return f"{self.device}freq{self.core_ind}"

    @acc_mem_name.default
    def _default_acc_mem_name(self) -> str:
        return f"accbuf{self.core_ind}"


@qfrozen
class QubicExecutable(QuantumExecutable):
    program: CompiledProgram = field(eq=id)
    # cattrs will always copy a dict when converting, so we disable autoconversion
    # to allow QubicExecutable's to be copied with the exact same assembly
    assembly: dict = field(
        eq=id, metadata=dict(auto_convert=False), repr=lambda asm: asm.keys()
    )
    repetition_delay: float
    reads_per_timeline: tuple[int, ...] = tuple()

    @property
    def total_reads(self) -> int:
        return sum(self.reads_per_timeline)

    @property
    def num_timelines(self) -> int:
        return len(self.reads_per_timeline)


def find_constant_segments(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds locations of all constant segments in an array.

    This function returns three arrays specifying the starting index (in the original
    array), value, and length of each constant segment. The original array can be
    reconstructed from these arrays as `np.repeat(values, lengths)`.

    Args:
        arr: The array to break into constant segments. It is assumed to be 1D.

    Returns:
        A tuple `(locations, values, lengths)` of arrays.
    """
    mask = np.r_[True, arr[1:] != arr[:-1], True]

    locs = np.flatnonzero(mask)
    vals = arr[locs[:-1]]
    lengths = np.diff(locs)

    return locs[:-1], vals, lengths


@register_compiler
@qdefine
class QubicCompiler(QWiPCompiler):
    """Qubic-specific compiler"""

    fpga_config: FPGAConfig = field(factory=FPGAConfig)
    reset_delay: float = 500e-6
    start_offset: int = 5

    def envelope_to_pulses(
        self,
        w_t,
        channel,
        frequency,
        phase,
        amplitude,
        pulse_width,
        start_cycle,
        sample_rate,
        cw_threshold: int | None = 16,
    ):
        """Compiles a single envelope into a list of Qubic pulse instructions.

        Args:
            loc: The location of the waveform.
            wave: The waveform to compile.
            waveform_cache: A cache for reusing envelope timepoints. Envelopes are
                cached by the waveform definition and sample rate.
            reads: A counter for tracking the number of reads on each channel. This is
                incremented if the waveform is on a read channel.
            pulse_kwargs: Any additional constraints to pass to the waveform when
                resolving timepoints.
            t0: The start time, in clock cycles, of the pulse timeline. The default
                clock cycle period is 2ns on Qubic.
        """
        clock_multiplier = int(sample_rate * self.fpga_config.fpga_clk_period)
        locs, vals, lengths = find_constant_segments(w_t)

        if cw_threshold is None:
            cw_threshold = len(w_t)

        # Find starting index of all constant segments in w_t that are longer than
        # a certain threshold. The clock_multiplier ensures that the start_times of
        # all the pulses line up with a fpga clock cycle.
        (cw_idx,) = np.where(lengths >= clock_multiplier * cw_threshold)

        if len(cw_idx) == 0:
            return [
                Pulse(
                    env=w_t,
                    dest=channel,
                    freq=frequency,
                    phase=phase,
                    amp=amplitude,
                    twidth=pulse_width,
                    start_time=start_cycle,
                )
            ]

        # Find start and end points of all CW segments. These will get replaced by a
        # pulse where env = "CW"
        cw_segments = np.stack([locs[cw_idx], locs[cw_idx] + lengths[cw_idx]]).T

        # Round to closest time that aligns with a clock cycle and ensure that no
        # instructions will have start times within 3 clock cycles of each other
        min_clk_cycles = 3
        cw_segments[:, 0] = np.ceil(cw_segments[:, 0] / clock_multiplier).astype(int)
        cw_segments[:, 1] = np.floor(cw_segments[:, 1] / clock_multiplier).astype(int)

        num_clk_cycles = cw_segments[:, 0] - np.r_[0, cw_segments[:-1, 1]]
        start_padding = min_clk_cycles - num_clk_cycles
        start_padding[start_padding < 0] = 0
        cw_segments[:, 0] += start_padding

        num_clk_cycles = (
            np.r_[cw_segments[1:, 0], len(w_t) // clock_multiplier] - cw_segments[:, 1]
        )
        end_padding = min_clk_cycles - num_clk_cycles
        end_padding[end_padding < 0] = 0
        cw_segments[:, 1] -= end_padding

        # Convert back to samples
        cw_segments *= clock_multiplier
        cw_amplitudes = vals[cw_idx]

        instructions = []

        prev_cw_e = 0
        for (cw_s, cw_e), env_amp in zip(cw_segments, cw_amplitudes):
            if cw_s > prev_cw_e:
                ins = Pulse(
                    env=w_t[prev_cw_e:cw_s],
                    dest=channel,
                    freq=frequency,
                    phase=phase,
                    amp=amplitude,
                    twidth=(cw_s - prev_cw_e) / sample_rate,
                    start_time=start_cycle + int(prev_cw_e / clock_multiplier),
                )
                instructions.append(ins)

            ins = Pulse(
                env="cw",
                dest=channel,
                freq=frequency,
                phase=phase + np.angle(env_amp),
                amp=amplitude * np.abs(env_amp),
                twidth=(cw_e - cw_s) / sample_rate,
                start_time=start_cycle + int(cw_s / clock_multiplier),
            )
            instructions.append(ins)

            prev_cw_e = cw_e

        if prev_cw_e < len(w_t):
            ins = Pulse(
                env=w_t[prev_cw_e:],
                dest=channel,
                freq=frequency,
                phase=phase,
                amp=amplitude,
                twidth=(len(w_t) - prev_cw_e) / sample_rate,
                start_time=start_cycle + int(prev_cw_e / clock_multiplier),
            )
            instructions.append(ins)

        return instructions

    def compile_instruction(
        self,
        loc: Location,
        wave: Waveform,
        *,
        waveform_cache: dict[tuple[Waveform, int], np.ndarray],
        reads: Counter[str],
        pulse_kwargs: dict = {},
        t0: int = 0,
        **kwargs,
    ) -> list:
        """Compiles a single waveform into a list of qubic instructions

        Args:
            loc: The location of the waveform.
            wave: The waveform to compile.
            waveform_cache: A cache for reusing envelope timepoints. Envelopes are
                cached by the waveform definition and sample rate.
            reads: A counter for tracking the number of reads on each channel. This is
                incremented if the waveform is on a read channel.
            pulse_kwargs: Any additional constraints to pass to the waveform when
                resolving timepoints.
            t0: The start time, in clock cycles, of the pulse timeline. The default
                clock cycle period is 2ns on Qubic.
            **kwargs: Additional keyword arguments are passed to the `envelope_to_pulse`
                method.
        """

        instructions = []

        width = wave.width.resolve(**pulse_kwargs)
        start, end = loc, loc + width

        match wave.channels:
            case ():
                channel = None
            case (channel,):
                ...
            case _:
                raise ValueError(
                    f"Waveforms on qubic should only address a single channel. Got "
                    f"{wave} with channels {wave.channels}"
                )

        if channel:
            if (ch_info := self.get_channel_info(channel)) is None:
                raise ValueError(f"Channel {channel} is not a valid channel.")

            if ch_info.read:
                reads[channel] += 1

            dtype = self.devices[ch_info.device].dtype
        else:
            dtype = None

        # np.round().astype() will return a np.int32 instead of an int
        start_cycle = t0 + int(
            np.round(start.offset / self.fpga_config.fpga_clk_period)
        )

        match wave:
            case VirtualZWaveform(frame=frame, phase=phase):
                if frame.references:
                    raise ValueError(
                        f"Qubic does not support vector mod keys, got {frame}."
                    )

                if not isinstance(frame.offset, str):
                    raise ValueError(
                        f"Virtual Z frames must be named on Qubic, got {frame}"
                    )

                match frame.offset.split("."):
                    case (qubit, *freqname):
                        qubit = qubit
                        freqname = ".".join(freqname)
                    case qubit:
                        qubit = qubit
                        freqname = None

                instructions.append(
                    VirtualZ(qubit=qubit, phase=phase * np.pi / 180, freq=freqname)
                )

            case ModulatedWaveform(envelope=env, modulation=mod):
                sample_rate = self.devices[ch_info.device].sample_rate
                # First check if we've evaluated this envelope already
                if env in waveform_cache:
                    w_t = waveform_cache[env, int(sample_rate)]
                else:
                    N = np.ceil(width.offset * sample_rate).astype(int)
                    ts_wave = np.arange(N) / sample_rate
                    w_t = wave(
                        ts_wave,
                        frames=self.frames,
                        complex_out=issubclass(dtype, np.complexfloating),
                        **pulse_kwargs,
                    )

                    waveform_cache[env, int(sample_rate)] = w_t

                if mod.hardware_modulation:
                    if mod.frequency.references:
                        raise ValueError(
                            f"Qubic does not support vector mod keys, got {mod.frequency}."
                        )

                    freq = mod.frequency.offset
                    amplitude = mod.amplitude
                else:
                    freq = 0
                    amplitude = 1

                ins = self.envelope_to_pulses(
                    w_t,
                    channel,
                    frequency=freq,
                    phase=0,  # Easier to always build phase into envelope
                    amplitude=amplitude,
                    pulse_width=width.offset,
                    start_cycle=start_cycle,
                    sample_rate=sample_rate,
                    **kwargs,
                )
                instructions.extend(ins)

            case Marker():
                ...

            case BasicWaveform():
                sample_rate = self.devices[ch_info.device].sample_rate

                if wave in waveform_cache:
                    w_t = waveform_cache[wave, int(sample_rate)]
                else:
                    ch_info = self.get_channel_info(channel)

                    N = np.ceil(width.offset * sample_rate).astype(int)
                    ts_wave = np.arange(N) / sample_rate
                    w_t = wave(ts_wave, frames=self.frames, **pulse_kwargs)

                    waveform_cache[wave, int(sample_rate)] = w_t

                ins = self.envelope_to_pulses(
                    w_t.astype(dtype),
                    channel,
                    frequency=0,
                    phase=0,
                    amplitude=1,
                    pulse_width=width.offset,
                    start_cycle=start_cycle,
                    sample_rate=sample_rate,
                    **kwargs,
                )
                instructions.extend(ins)

        return instructions

    def compile_timeline(
        self,
        locations: dict[Location, list[Waveform]],
        *,
        waveform_cache: dict[tuple[Waveform, int], np.ndarray] = {},
        pulse_kwargs: dict = {},
        t0: int = 0,
        **kwargs,
    ) -> tuple[list, Counter[str]]:
        """Compiles a single pulse timeline.

        Args:
            locations: A mapping from locations to a list of waveforms.
            waveform_cache: A cache for reusing envelope timepoints. Envelopes are
                cached by the waveform definition and sample rate.
            pulse_kwargs: Any extra constraints to pass to waveforms when resolving them
                into timepoints.
            t0: The start time, in clock cycles, of the pulse timeline. The default
                clock cycle period is 2ns on Qubic.
            **kwargs: Additional keyword arguments are passed to the
                `compile_instruction` method.

        Returns:
            A list of instructions and a counter specifying the number of reads on each
            channel.
        """
        instructions = []
        start_times = []
        reads = Counter()

        for loc, waves in locations.items():
            waves = sorted(
                waves, key=lambda w: 0 if isinstance(w, VirtualZWaveform) else 1
            )
            for w in waves:
                new_instructions = self.compile_instruction(
                    loc=loc,
                    wave=w,
                    waveform_cache=waveform_cache,
                    reads=reads,
                    pulse_kwargs=pulse_kwargs,
                    t0=t0,
                    **kwargs,
                )

                # Maintain start_time ordering when adding instructions.
                for ins in new_instructions:
                    N = len(instructions)

                    if not hasattr(ins, "start_time"):
                        if start_times:
                            st = start_times[-1] + getattr(
                                instructions[-1], "twidth", 0
                            )
                        else:
                            st = 0

                        instructions.append(ins)
                        start_times.append(st)
                        continue

                    for i in range(N):
                        # old_ins = instructions[N - i - 1]
                        prev_start = start_times[N - i - 1]

                        if prev_start > ins.start_time:
                            continue

                        instructions.insert(N - i, ins)
                        start_times.insert(N - i, ins.start_time)

                        # if not hasattr(old_ins, "start_time"):
                        #     instructions.insert(N - i, ins)
                        # elif instructions[N - i - 1].start_time > ins.start_time:
                        #     continue
                        # else:
                        #     instructions.insert(N - i, ins)

                        break
                    else:
                        instructions.insert(0, ins)
                        start_times.insert(0, ins.start_time)

        return instructions, reads

    def construct_circuit(
        self,
        seq: Sequence,
        reset_delay: float,
        location_kwargs: dict = {},
        pulse_kwargs: dict = {},
        **kwargs,
    ) -> tuple[list, list[int]]:
        """Constructs a Qubic instruction list from a sequence.

        This method makes a pass through the sequence, compiling each pulse timeline
        to a list of qubic instructions. These instruction lists are then concatenated
        with a specified reset delay in between. The number of reads per timeline is
        also returned.

        Args:
            seq: The sequence to compile into a circuit.
            reset_delay: The delay time in seconds between the start times of
                consecutive timelines.
            location_kwargs: Any extra constraints to pass to the location resolver.
            pulse_kwargs: Any extra constraints to pass to waveforms when resolving them
                into timepoints.
            **kwargs: Additional keyword arguments are passed to the `compile_timeline`
                method.

        Returns:
            A tuple `(circuit, reads_per_timeline)`.
        """
        markers = {}
        waveform_cache = {}
        locations = [
            tmln.resolve_locations(end_marker="end", markers=markers, **location_kwargs)
            for tmln in seq.flat
        ]

        reads_per_timeline = []
        circuit = []
        for i, (tmln_locs, tmln) in enumerate(zip(locations, seq.flat)):
            t_end = markers["end"]

            if t_end > reset_delay:
                raise ValueError(
                    f"Timeline length {t_end} is greater than reset delay "
                    f"{reset_delay}."
                )

            instructions, reads = self.compile_timeline(
                tmln_locs,
                waveform_cache=waveform_cache,
                pulse_kwargs=tmln.constraints | pulse_kwargs,
                t0=self.start_offset
                + int(
                    np.round((i + 1) * reset_delay / self.fpga_config.fpga_clk_period)
                ),
                **kwargs,
            )

            reads_per_channel = set(cts[1] for cts in reads.most_common())
            if len(reads_per_channel) > 1:
                logger.warning(
                    f"Timeline {i} has an unequal number of reads accross channels. "
                    f"{reads}"
                )

            circuit.extend(instructions)
            reads_per_timeline.append(max(reads_per_channel))

        return circuit, reads_per_timeline

    def compile(
        self,
        seq: Sequence,
        location_kwargs: dict = {},
        pulse_kwargs: dict = {},
        reset_delay: float | None = None,
        **kwargs,
    ) -> QubicExecutable:
        """Compiles a sequence to a Qubic executable format.

        Args:
            seq: The sequence to compile.
            location_kwargs: A mapping of location variables to concrete values. This is
                passed to `Timeline.resolve_locations`.

        Returns:
            The resulting compiled `QubicExecutable` that can then be run on hardware.
        """
        reset_delay = reset_delay or self.reset_delay
        circuit, reads_per_timeline = self.construct_circuit(
            seq, reset_delay, location_kwargs, pulse_kwargs, **kwargs
        )
        qchip = self.get_qchip()
        channel_config = self.get_channel_config()

        passes = kwargs.get("passes", get_compiler_passes(self.fpga_config, qchip))
        qubic_compiler = _QubicInternalCompiler(circuit)
        qubic_compiler.run_ir_passes(passes)

        prog = qubic_compiler.compile()
        asm = tc.run_assemble_stage(prog, channel_config)
        return QubicExecutable(
            sequence=seq,
            program=prog,
            assembly=asm,
            repetition_delay=len(seq.flat) * reset_delay,
            reads_per_timeline=reads_per_timeline,
        )

    def get_qchip(self) -> QChip:
        """Returns a Qubic QChip object with the named modulation frequencies.

        Returns:
            A Qubic `QChip` instance.
        """
        frames = FlatDict(
            {k.replace(".", "/"): f.offset for k, f in self.frames.items()}
        )

        return QChip(dict(Qubits=frames, Gates=dict()))

    def get_channel_config(self) -> dict:
        """Return a QubicChannelConfig with the channel information.

        Returns:
            A Qubic `QubicChannelConfig` instance.
        """
        channel_config = dict(fpga_clk_freq=self.fpga_config.fpga_clk_freq)

        for dev in self.devices.values():
            sample_rate = dev.sample_rate

            for ch in dev.channels:
                _, device = ch.name.split(".")
                samples_per_clk = 4 if device.lower() == "rdlo" else 16
                interp_ratio = round(
                    samples_per_clk / (sample_rate * self.fpga_config.fpga_clk_period)
                )

                channel_config[ch.name] = QubicChannelConfig(
                    device=device,
                    core_ind=ch.index,
                    elem_ind=ch.subchannel,
                    elem_params=dict(
                        samples_per_clk=samples_per_clk, interp_ratio=interp_ratio
                    ),
                )

        return channel_config


@qdefine
class QubicBackend(QuantumBackend):
    """A hardware backend for interfacing with qubic."""

    runner: CircuitRunnerClient
    result_map: dict = field(factory=dict)

    def upload(self, exe: QubicExecutable, **kwargs: Any) -> None:
        """Simulate a circuit upload onto the qubic board.

        The actual upload happens with the acquire method call.

        Args:
            exe: A `QubicExecutable` which contains the assembly dict to load.
            **kwargs: Additional keyword arguments are passed to `CircuitRunner.load_circuit`.
        """
        self.uploaded = exe

    def acquire(self, repetitions: int = 512, **kwargs) -> dict[str, IQResult]:
        """Acquires data from the qubic board.

        Args:
            repetitions: The number of shots to acquire for the given circuit.
        """
        if self.uploaded is None:
            raise ValueError(
                "No asm to execute! Call upload to load an executable or pass an "
                "executable to acquire."
            )

        exe = self.uploaded

        result = self.runner.run_circuit_batch(
            [exe.assembly],
            repetitions,
            reads_per_shot=exe.total_reads,
        )

        tmlns = np.r_[
            tuple(np.repeat(i, n) for i, n in enumerate(exe.reads_per_timeline))
        ]
        reads = np.r_[tuple(np.arange(n) for n in exe.reads_per_timeline)]
        shots = np.arange(repetitions)

        index = pd.MultiIndex.from_arrays(
            [
                np.repeat(shots, exe.total_reads),
                np.tile(tmlns, repetitions),
                np.tile(reads, repetitions),
            ],
            names=["shot", "timeline", "readout"],
        )

        iq_results = {}
        for k, data in result.items():
            df = pd.DataFrame(data[0].flatten().conj(), index=index, columns=["IQ"])

            name = self.result_map.get(k, k)
            iq_results[name] = IQResult(name=name, data=df)

        return iq_results

    def update_parameters(self, qpu: "QPU", **kwargs):
        """Updates parameters from the QPU.

        This method updates the mapping from readout indices to readout channels.

        Args:
            qpu: A QPU instance.
        """

        for devices in qpu.compiler.devices.values():
            for ch_info in devices.channels:
                if not ch_info.read:
                    continue

                self.result_map.update({str(ch_info.index): ch_info.name})
