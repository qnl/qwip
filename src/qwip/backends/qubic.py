import itertools as it
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from attrs import field
from loguru import logger

try:
    import qubic.toolchain as tc
    from distproc.compiler import CompiledProgram
    from distproc.compiler import Compiler as _QubicInternalCompiler
    from distproc.compiler import CompilerFlags, get_passes
    from distproc.executable import Executable
    from distproc.hwconfig import ChannelConfig as QubicChannelConfig
    from distproc.hwconfig import FPGAConfig, FPROCChannel
    from distproc.ir import passes
    from distproc.ir.instructions import (
        BranchFproc,
        DeclareFreq,
        Idle,
        Pulse,
        VirtualZ,
    )
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
    ChannelInfo,
    QuantumExecutable,
    QWiPCompiler,
    register_compiler,
)
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.timeline import Timeline
from qwip.sequencer.utils import _to_python_number
from qwip.sequencer.waveform import (
    BasicWaveform,
    BranchOperation,
    DCWaveform,
    Marker,
    ModulatedWaveform,
    Operation,
    ResetOperation,
    VirtualZWaveform,
    Waveform,
)

if TYPE_CHECKING:
    from qwip.qpu.qpu import QPU


def _sort_virtual_z(loc_op: tuple[float, Operation]) -> tuple[float, int]:
    """Sort key function so that VirtualZWaveforms come first."""
    loc, op = loc_op
    return (loc, int(not isinstance(op, VirtualZWaveform)))


def get_compiler_passes(
    fpga_config: FPGAConfig,
    qchip: QChip,
    qubit_grouping: tuple[str, ...] = ("{qubit}.qdrv", "{qubit}.rdrv", "{qubit}.rdlo"),
    proc_grouping: list[tuple[str, ...]] = [
        ("{qubit}.qdrv", "{qubit}.rdrv", "{qubit}.rdlo"),
        ("{qubit}.qdrv2", "{qubit}.dcoffs"),
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
class QubicExecutable(QuantumExecutable):
    program: CompiledProgram = field(eq=id)
    # cattrs will always copy a dict when converting, so we disable autoconversion
    # to allow QubicExecutable's to be copied with the exact same assembly
    assembly: Executable = field(
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


def _get_board_and_core(devname: str) -> tuple[str, str]:
    match devname.split("_"):
        case (board, core, sig_gen):
            return board, core, sig_gen
        case (core, sig_gen):
            return "", core, sig_gen
        case _:
            raise ValueError(f"Malformed QubiC device name: {devname}")


def _get_memory_name(channel: ChannelInfo) -> dict[str, str]:
    board, core, sig_gen = _get_board_and_core(channel.device)

    memory = dict(
        env_mem_name=f"{core}_{sig_gen}_env{channel.index}",
        freq_mem_name=f"{core}_{sig_gen}_freq{channel.index}",
    )

    if channel.read:
        memory["acc_mem_name"] = f"{core}_accbuf{channel.index}"

    return memory


# distproc's Schedule pass starts the first basic block at this clock.
_DISTPROC_START_NCLKS = 5


class _TimedBranch:
    """A `BranchFproc` tagged with the nominal start of its arms.

    `compile_timeline` orders instructions by ``start_time``, and
    `QubicCompiler._rebase_blocks` needs the branch's nominal start and the
    readout channel whose FPROC result it reads. The wrapper is removed (and the
    bare `BranchFproc` emitted) by `_rebase_blocks`.
    """

    __slots__ = ("branch", "start_time", "twidth", "channel", "name")

    def __init__(self, branch, start_time: int, twidth: float, channel: str, name):
        self.branch = branch
        self.start_time = start_time
        self.twidth = twidth
        self.channel = channel
        self.name = name

    def __repr__(self):
        return (
            f"_TimedBranch(name={self.name!r}, start_time={self.start_time}, "
            f"channel={self.channel!r}, branch={self.branch!r})"
        )


@register_compiler
@qdefine
class QubicCompiler(QWiPCompiler):
    """Qubic-specific compiler"""

    fpga_config: FPGAConfig = field(factory=FPGAConfig)
    frame_scopes: dict[str, str] = field(factory=dict)
    start_offset: int = 5

    def __attrs_post_init__(self):
        self._register_readout_fproc_channels()

    def _register_readout_fproc_channels(self) -> None:
        """Register an FPROC channel for every readout channel on this device.

        A measurement-conditioned branch reads its discrimination result, and
        anchors its settling Hold, through an FPROC channel that references the
        readout channel. distproc's default ``FPGAConfig`` only describes the
        ``Q{i}.rdlo`` naming convention, so a device that names readout channels
        differently (e.g. ``CH7.rdlo``) would have no FPROC channel for them.
        For each readout (``read=True``) channel not already referenced by an
        existing FPROC channel, add one anchored on that channel.
        """
        referenced = {
            chan_name
            for fproc in self.fpga_config.fproc_channels.values()
            for chan_name in fproc.hold_after_chans
        }
        for device in self.devices.values():
            for ch in device.channels:
                if not ch.read or ch.name in referenced:
                    continue
                prefix = ch.name.rsplit(".", 1)[0]
                self.fpga_config.fproc_channels[f"{prefix}.meas"] = FPROCChannel(
                    id=(ch.name, "core_ind"),
                    hold_after_chans=[ch.name],
                    hold_nclks=self.fpga_config.fproc_meas_clks,
                )

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
        save_result: bool | None = None,
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
                    save_result=save_result,
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
                    save_result=save_result,
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
                save_result=save_result,
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
                save_result=save_result,
            )
            instructions.append(ins)

        return instructions

    def compile_instruction(
        self,
        loc: float,
        wave: Waveform,
        *,
        waveform_cache: dict[tuple[Waveform, int], np.ndarray],
        reads: Counter[str],
        t0: int = 0,
        _in_reset: bool = False,
        fold_phases: dict | None = None,
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
            t0: The start time, in clock cycles, of the pulse timeline. The default
                clock cycle period is 2ns on Qubic.
            fold_phases: If given, virtual-Z waveforms are not emitted as
                `VirtualZ` instructions; their phase is accumulated in this
                mapping (frame -> radians) and added to the phase of later
                hardware-modulated pulses on the same frame. Used inside the
                arms of a reset, see `compile_timeline`.
            **kwargs: Additional keyword arguments are passed to the `envelope_to_pulse`
                method.
        """

        instructions = []

        width = _to_python_number(wave.width)
        start = _to_python_number(loc)
        end = start + width

        if (ch_info := self.get_channel_info(wave.channel)) is None:
            raise ValueError(f"Channel {wave.channel} is not a valid channel.")

        if ch_info.read:
            reads[wave.channel] += 1

        # Active-reset measurements are demodulated for the conditional branch's
        # discrimination but must not be stored to the acc buffer (the runner would
        # otherwise misalign its reshape). save_result=False makes the gateware run
        # the demod without storing the IQ value or advancing the write pointer.
        save_result = False if (ch_info.read and _in_reset) else None

        devinfo = self.devices[ch_info.device]
        dtype = devinfo.dtype
        sample_rate = devinfo.envelope_sample_rate or devinfo.sample_rate
        is_dc = bool(sample_rate == 0)

        # np.round().astype() will return a np.int32 instead of an int
        start_cycle = t0 + int(np.round(start / self.fpga_config.fpga_clk_period))

        match wave:
            case VirtualZWaveform(frame=frame, phase=phase):
                if frame.references:
                    raise ValueError(
                        f"Qubic does not support vector mod keys, got {frame}."
                    )

                if isinstance(frame.offset, str):
                    match frame.offset.split("."):
                        case (qubit, *freqname):
                            qubit = qubit
                            freqname = ".".join(freqname)
                        case qubit:
                            qubit = qubit
                            freqname = None
                else:
                    qubit = None
                    freqname = frame.offset

                if fold_phases is not None:
                    fold_phases[frame.offset] = (
                        fold_phases.get(frame.offset, 0.0) + phase * np.pi / 180
                    )
                else:
                    instructions.append(
                        VirtualZ(
                            qubit=qubit,
                            phase=phase * np.pi / 180,
                            freq=freqname,
                            scope=wave.channel,
                        )
                    )

            case DCWaveform():
                instructions.append(
                    Pulse(
                        env=None,
                        dest=wave.channel,
                        freq=None,
                        phase=0,
                        amp=wave.amplitude,
                        twidth=0,
                        start_time=start_cycle,
                    )
                )

            case ModulatedWaveform(envelope=env, modulation=mod):
                # First check if we've evaluated this envelope already
                if env in waveform_cache:
                    w_t = waveform_cache[env, int(sample_rate)]
                else:
                    N = np.ceil(width * sample_rate).astype(int)
                    ts_wave = np.arange(N) / sample_rate
                    w_t = wave(ts_wave, frames=self.frames)

                    waveform_cache[env, int(sample_rate)] = w_t

                if mod.hardware_modulation:
                    if mod.frequency.references:
                        raise ValueError(
                            f"Qubic does not support vector mod keys, got {mod.frequency}."
                        )

                    freq = mod.frequency.offset
                    amplitude = mod.amplitude
                    phase = mod.phase * np.pi / 180
                    if fold_phases is not None:
                        phase += fold_phases.get(freq, 0.0)
                else:
                    freq = 0
                    amplitude = 1
                    phase = 0

                ins = self.envelope_to_pulses(
                    w_t,
                    wave.channel,
                    frequency=freq,
                    phase=phase,
                    amplitude=amplitude,
                    pulse_width=width,
                    start_cycle=start_cycle,
                    sample_rate=sample_rate,
                    save_result=save_result,
                    **kwargs,
                )
                instructions.extend(ins)

            case Marker():
                ...

            case BasicWaveform():
                if (wave, int(sample_rate)) in waveform_cache:
                    w_t = waveform_cache[wave, int(sample_rate)]
                else:
                    ch_info = self.get_channel_info(wave.channel)

                    N = np.ceil(width * sample_rate).astype(int)
                    ts_wave = np.arange(N) / sample_rate
                    w_t = wave(ts_wave, frames=self.frames)

                    waveform_cache[wave, int(sample_rate)] = w_t

                ins = self.envelope_to_pulses(
                    w_t.astype(dtype),
                    wave.channel,
                    frequency=0,
                    phase=0,
                    amplitude=1,
                    pulse_width=width,
                    start_cycle=start_cycle,
                    sample_rate=sample_rate,
                    save_result=save_result,
                    **kwargs,
                )
                instructions.extend(ins)

            case BranchOperation():
                reads[wave.channel] -= 1
                left = wave.left or Timeline(width=0)
                right = wave.right or Timeline(width=0)

                # The arms are laid out at absolute times like everything else;
                # `_rebase_blocks` converts them to block-relative times later.
                #
                # Inside a reset the arm's virtual-Z markers are folded into its
                # pulse phases here instead of being emitted for distproc. The
                # arm must leave every frame's phase unchanged (the qubit is
                # being returned to |0>, so the frame phase carries no
                # information), and distproc checks that the two arms end at
                # exactly equal accumulated phases: adding and then subtracting
                # the markers on top of a nonzero incoming phase can differ by
                # one floating-point ulp and fail that check.
                arm_kwargs = dict(
                    waveform_cache=waveform_cache,
                    t0=start_cycle,
                    reset_delay=np.inf,
                    zero_dc=False,
                )
                if _in_reset:
                    left_ins, _ = self.compile_timeline(left, fold_phases={}, **arm_kwargs)
                    right_ins, _ = self.compile_timeline(right, fold_phases={}, **arm_kwargs)
                else:
                    left_ins, _ = self.compile_timeline(left, **arm_kwargs)
                    right_ins, _ = self.compile_timeline(right, **arm_kwargs)

                # Find the FPROC channel that reads this readout channel. distproc
                # uses it to insert a wait (a Hold) before the branch, so the branch
                # does not run until the measurement has finished and its result is
                # ready. That wait is measured from the end of the most recent pulse
                # on the readout channel, so it stays correct even if the measure
                # pulse's declared width is shorter than its real duration.
                func_ids = [
                    name
                    for name, chan in self.fpga_config.fproc_channels.items()
                    if wave.channel in chan.hold_after_chans
                ]
                if len(func_ids) != 1:
                    raise ValueError(
                        f"Expected exactly one FPROC channel referencing "
                        f"{wave.channel!r} in hold_after_chans, found {sorted(func_ids)}; "
                        f"cannot compile a measurement-conditioned branch on "
                        f"{wave.channel!r}. Configured FPROC channels: "
                        f"{sorted(self.fpga_config.fproc_channels)}"
                    )
                func_id = func_ids[0]

                # The scope is widened to every channel in the circuit by
                # `_rebase_blocks`, which also converts the absolute timestamps
                # in the arms to the block-relative form distproc expects.
                branch = BranchFproc(
                    cond_lhs=1,  # left half plane
                    alu_cond="eq",
                    func_id=func_id,
                    scope=list(left.channels | right.channels),
                    true=left_ins,
                    false=right_ins,
                )
                instructions.append(
                    _TimedBranch(
                        branch,
                        start_time=start_cycle,
                        twidth=width,
                        channel=wave.channel,
                        name=wave.name,
                    )
                )

            case ResetOperation():
                # Cancel the spurious +1 from the read-channel check above; the
                # measurements happen inside the expanded sub-timeline and any
                # user-visible read accounting belongs to the outer measurement
                # at the end of the parent timeline (matches BranchOperation).
                reads[wave.channel] -= 1

                # Each branch needs `fproc_delay` of room after the measurement:
                # `_rebase_blocks` holds every core until `fproc_meas_clks` after
                # the demod window ends, and the jump itself takes
                # `pulse_load_clks + jump_fproc_clks` before the pi pulse can start.
                layers = []
                for n in range(wave.n_resets):
                    if n > 0 or wave.measure_first:
                        layers.append(wave.measurement)
                    layers.append(wave.fproc_delay)
                    layers.append(
                        BranchOperation(
                            name=f"{wave.name}_branch{n}",
                            channel=wave.channel,
                            left=wave.x_pulse,
                            right=None,
                            width=wave.x_pulse.width,
                        )
                    )
                expanded = Timeline.from_layers(layers)
                sub_ins, _ = self.compile_timeline(
                    expanded,
                    waveform_cache=waveform_cache,
                    t0=start_cycle,
                    reset_delay=np.inf,
                    zero_dc=False,
                    _in_reset=True,
                )
                instructions.extend(sub_ins)

        return instructions

    def compile_timeline(
        self,
        tmln: Timeline,
        substitutions: dict = {},
        *,
        waveform_cache: dict[tuple[Waveform, int], np.ndarray] = {},
        t0: int = 0,
        reset_delay: float = 0,
        zero_dc: bool = True,
        _in_reset: bool = False,
        **kwargs,
    ) -> tuple[list, Counter[str]]:
        """Compiles a single pulse timeline.

        Args:
            locations: A mapping from locations to a list of waveforms.
            waveform_cache: A cache for reusing envelope timepoints. Envelopes are
                cached by the waveform definition and sample rate.
            t0: The start time, in clock cycles, of the pulse timeline. The default
                clock cycle period is 2ns on Qubic.
            **kwargs: Additional keyword arguments are passed to the
                `compile_instruction` method.

        Returns:
            A list of instructions and a counter specifying the number of reads on each
            channel.
        """
        tmln.resolve(inplace=True, sort=_sort_virtual_z, **substitutions)

        phase_tracker = self.compile_phases(tmln)

        instructions = []
        start_times = []
        reads = Counter()

        for loc, w in tmln:
            new_instructions = self.compile_instruction(
                loc=loc,
                wave=w,
                waveform_cache=waveform_cache,
                reads=reads,
                t0=t0,
                _in_reset=_in_reset,
                **kwargs,
            )

            # Maintain start_time ordering when adding instructions.
            for ins in new_instructions:
                N = len(instructions)

                if not hasattr(ins, "start_time"):
                    if start_times:
                        st = start_times[-1] + getattr(instructions[-1], "twidth", 0)
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

                    break
                else:
                    instructions.insert(0, ins)
                    start_times.insert(0, ins.start_time)

        if zero_dc:
            t_end = _to_python_number(tmln.width)
            end_cycle = t0 + int(np.round(t_end / self.fpga_config.fpga_clk_period))
            for device in self.devices.values():
                if device.sample_rate != 0:
                    continue

                for ch in device.channels:
                    instructions.append(
                        Pulse(
                            env=None,
                            dest=ch.name,
                            freq=None,
                            phase=0,
                            amp=0,
                            twidth=0,
                            start_time=end_cycle,
                        )
                    )

        return instructions, reads

    def construct_circuit(
        self,
        seq: Sequence,
        reset_delay: float | None,
        preamble: list = [],
        substitutions: dict = {},
        **kwargs,
    ) -> tuple[list, list[int], float]:
        """Constructs a Qubic instruction list from a sequence.

        Timelines are packed back-to-back: each timeline's slot is its own
        `width` (when `reset_delay is None`) or `reset_delay` (when an explicit
        value is given). This means short timelines no longer have to share the
        longest timeline's slot size — useful for swept experiments where the
        timeline duration varies dramatically across sweep points.

        Args:
            seq: The sequence to compile into a circuit.
            reset_delay: The per-shot period in seconds. If `None`, each
                timeline's slot equals its own `width` (tight pack — appropriate
                when each timeline includes its own active reset, or when the
                timeline already accounts for passive decay). If a float, every
                timeline gets a uniform slot of `reset_delay` seconds, and
                construction raises `ValueError` if any timeline is longer than
                this value.
            preamble: Instructions to prepend to the circuit.
            substitutions: Variable substitutions passed to timeline resolution.
            **kwargs: Additional keyword arguments are passed to the `compile_timeline`
                method.

        Returns:
            A tuple `(circuit, reads_per_timeline, total_duration)` where
            `total_duration` is the sum of all timeline slots in seconds.
        """
        if reset_delay is not None:
            max_width = max(
                (
                    w
                    for w in (
                        _to_python_number(tmln.width) for tmln in seq.flat
                    )
                    if w is not None
                ),
                default=0,
            )
            if reset_delay < max_width:
                raise ValueError(
                    f"reset_delay ({reset_delay:g} s) is shorter than the longest "
                    f"timeline width ({max_width:g} s). Pass a larger value, or "
                    f"pass reset_delay=None to tight-pack timelines using each "
                    f"timeline's own width."
                )

        waveform_cache = {}

        reads_per_timeline = []
        circuit = preamble.copy() or []
        clk = self.fpga_config.fpga_clk_period
        t0_cycles = self.start_offset
        for tmln in seq.flat:
            instructions, reads = self.compile_timeline(
                tmln,
                substitutions,
                waveform_cache=waveform_cache,
                t0=t0_cycles,
                reset_delay=np.inf,
                **kwargs,
            )

            slot = reset_delay if reset_delay is not None else _to_python_number(tmln.width)
            t0_cycles += int(np.round(slot / clk))

            reads_per_channel = set(cts[1] for cts in reads.most_common())
            if len(reads_per_channel) > 1:
                logger.warning(
                    f"Timeline {len(reads_per_timeline)} has an unequal number of "
                    f"reads accross channels. {reads}"
                )

            circuit.extend(instructions)
            reads_per_timeline.append(max(reads_per_channel | {0}))

        circuit.extend(self._end_of_batch_hold(circuit, t0_cycles))
        circuit = self._rebase_blocks(circuit)

        total_duration = (t0_cycles - self.start_offset) * clk
        return circuit, reads_per_timeline, total_duration

    def _end_of_batch_hold(self, circuit: list, end_cycles: int) -> list:
        """An `Idle` that keeps the program running until `end_cycles`.

        A slot works by delaying the start of the next timeline until the slot
        is over. The last timeline has no next timeline, so without this idle
        the program ends when its last pulse does and the next shot starts at
        once. The idle is applied to a single core. That is enough, because
        distproc makes every core wait for the latest instruction in the
        program, and it avoids a distproc bug that schedules an idle applied
        to several cores at the wrong time. Returns an empty list if the last
        instruction already reaches `end_cycles`.
        """
        clk = self.fpga_config.fpga_clk_period
        load = self.fpga_config.pulse_load_clks
        last, last_dest = 0, None

        def visit(instrs):
            nonlocal last, last_dest
            for ins in instrs:
                if isinstance(ins, Pulse):
                    width = int(np.ceil((ins.twidth or 0) / clk))
                    end = ins.start_time + width + load
                    if end >= last:
                        last, last_dest = end, ins.dest
                elif isinstance(ins, _TimedBranch):
                    visit(ins.branch.true)
                    visit(ins.branch.false)

        visit(circuit)
        if last_dest is None or end_cycles <= last:
            return []
        return [Idle(end_time=end_cycles, scope=[last_dest])]

    def _rebase_blocks(self, circuit: list) -> list:
        """Converts absolute timestamps to the block-relative ones distproc expects.

        distproc's ``Schedule`` pass, as configured by `compile` (local scheduling
        off), treats every timestamp as relative to the start of the basic block
        it is in and starts each block where its predecessors in the control-flow
        graph ended. A straight-line program is a single block starting at clock
        ``_DISTPROC_START_NCLKS``, so absolute times pass through unchanged. Each
        `BranchFproc` opens new blocks (the jump, the two arms, and the merge),
        and absolute times inside them would be shifted by the accumulated end
        time of everything before, so the program length grew roughly
        quadratically with the number of branches.

        This pass replays distproc's block splitting, scoping, control-flow graph
        and schedule bookkeeping on the absolute-time ``circuit`` and subtracts
        each block's start from the timestamps in it, so distproc's re-basing
        puts every pulse back at its nominal time. Before each branch it inserts
        an `Idle`, scoped to the branch's cores, that holds them until just
        before the branch's nominal start (and at least ``fproc_meas_clks``
        after the demod window ends), so the jump reads the state bit as late
        as the layout allows and the hardware actually waits for the result,
        which distproc's own Hold does not guarantee when it decides the hold
        is already satisfied. Branches are scoped to the processor cores their
        arms use, so branches on different cores that follow each other
        directly (two qubits reset at once) are scheduled side by side.

        distproc does not let basic blocks overlap in time, so no pulse may span
        a branch, and nothing may be scheduled between a branch's hold and the
        end of its longer arm. Violations raise `ValueError` with the timing
        that would be needed.

        Args:
            circuit: The instruction list built by `construct_circuit`, with
                `_TimedBranch` wrappers where `compile_instruction` lowered a
                `BranchOperation`.

        Returns:
            The instruction list with bare `BranchFproc` instructions, `Idle`
            holds before each branch, and block-relative timestamps.
        """
        fc = self.fpga_config
        clk = fc.fpga_clk_period
        load = fc.pulse_load_clks
        start_nclks = _DISTPROC_START_NCLKS

        group_of: dict[str, tuple] = {}
        for grp in self.get_proc_grouping():
            for ch in grp:
                group_of[ch] = grp

        def ns(t):
            return f"{t * clk * 1e9:.0f} ns"

        def nclks(twidth):
            return int(np.ceil(twidth / clk))

        def instr_scope(ins) -> set:
            scope = getattr(ins, "scope", None)
            return set(scope) if scope else set()

        def arm_dests(instrs):
            for ins in instrs:
                if isinstance(ins, Pulse):
                    yield ins.dest
                elif isinstance(ins, _TimedBranch):
                    yield from arm_dests(ins.branch.true)
                    yield from arm_dests(ins.branch.false)
                else:
                    yield from instr_scope(ins)

        if not any(isinstance(ins, _TimedBranch) for ins in circuit):
            # A single basic block: distproc adds `start_nclks` to every time.
            for ins in circuit:
                if isinstance(ins, Pulse):
                    ins.start_time -= start_nclks
                elif isinstance(ins, Idle):
                    ins.end_time -= start_nclks
            return circuit

        def branch_scope(tb: _TimedBranch) -> set:
            dests = set(arm_dests(tb.branch.true)) | set(arm_dests(tb.branch.false))
            scope = set()
            for d in dests:
                scope |= set(group_of[d])
            return scope or set(group_of[tb.channel])

        class Block:
            __slots__ = (
                "ops", "scope", "preds", "jump_to", "timed", "context",
                "start", "end", "pulse_end",
            )

            def __init__(self, context, scope=()):
                self.ops = []  # distproc-level operations that affect timing
                self.scope = set(scope)
                self.preds = set()
                self.jump_to = None
                self.timed = []  # Pulse/Idle objects to shift by the block start
                self.context = context

        blocks: list[Block] = []

        def new_block(context, scope=()):
            block = Block(context, scope)
            blocks.append(block)
            return block

        def emit(instrs, cur: Block):
            """Lays ``instrs`` out into basic blocks the way distproc's
            FlattenProgram and MakeBasicBlocks passes would, starting in ``cur``.
            Returns the rebuilt instruction list and the block the list ends in."""
            out = []
            i = 0
            while i < len(instrs):
                ins = instrs[i]
                if not isinstance(ins, _TimedBranch):
                    if isinstance(ins, Pulse):
                        cur.ops.append(("pulse", ins))
                        cur.scope.add(ins.dest)
                        cur.timed.append(ins)
                    else:
                        cur.scope |= instr_scope(ins)
                        if isinstance(ins, Idle):
                            cur.ops.append(("idle", ins))
                            cur.timed.append(ins)
                    out.append(ins)
                    i += 1
                    continue

                # Branches that directly follow each other (e.g. several qubits
                # reset at once) get their holds up front, in the same block, so
                # that each branch's control block hangs off that block rather
                # than off the previous branch's arms.
                run = [ins]
                while i + len(run) < len(instrs) and isinstance(
                    instrs[i + len(run)], _TimedBranch
                ):
                    run.append(instrs[i + len(run)])
                for tb in run:
                    scope = branch_scope(tb)
                    idle = Idle(end_time=0, scope=sorted(scope))
                    cur.ops.append(("hold_idle", idle, tb.channel, tb.start_time))
                    cur.scope |= scope
                    cur.timed.append(idle)
                    out.append(idle)
                for tb in run:
                    cur = emit_branch(tb, cur, out)
                i += len(run)
            return out, cur

        def emit_branch(tb: _TimedBranch, cur: Block, out: list) -> Block:
            br = tb.branch
            scope = branch_scope(tb)
            br.scope = sorted(scope)
            label = f"branch {tb.name!r} at {ns(tb.start_time)}"

            ctrl = new_block(f"jump of {label}", scope)
            ctrl.ops = [("hold", tb.channel, scope), ("jump_fproc", scope)]

            false_block = new_block(f"false arm of {label}", scope)
            br.false, false_end = emit(br.false, false_block)
            false_end.ops.append(("jump_i", scope))
            false_end.scope |= scope

            if br.true:
                true_block = new_block(f"true arm of {label}", scope)
                br.true, _ = emit(br.true, true_block)
                ctrl.jump_to = true_block

            merge = new_block(f"after {label}", scope)
            false_end.jump_to = merge
            if ctrl.jump_to is None:
                ctrl.jump_to = merge
            out.append(br)
            return merge

        first = new_block("start of program")
        circuit, _ = emit(circuit, first)
        blocks = [b for b in blocks if b.ops or b.scope]

        # GenerateCFG: implicit edges from the last block that touched each
        # channel, explicit edges from jumps.
        universe = set().union(*(b.scope for b in blocks))
        lastblock = {d: None for d in universe}
        for block in blocks:
            for d in block.scope:
                if lastblock[d] is not None:
                    block.preds.add(lastblock[d])
            last = block.ops[-1][0] if block.ops else None
            if last == "jump_fproc":
                block.jump_to.preds.add(block)
                for d in block.scope:
                    lastblock[d] = block
            elif last == "jump_i":
                block.jump_to.preds.add(block)
                for d in block.scope:
                    lastblock[d] = None
            else:
                for d in block.scope:
                    lastblock[d] = block

        # Schedule: blocks are created in program order, which is topological.
        for block in blocks:
            pulse_cur = {d: start_nclks for d in universe}
            for pred in block.preds:
                for d in pred.scope:
                    pulse_cur[d] = max(pulse_cur[d], pred.pulse_end[d])
            S = max((pred.end for pred in block.preds), default=start_nclks)
            block.start = S

            groups = {group_of[d] for d in block.scope}
            issue = {g: S for g in groups}
            busy = set()
            cur_t = {d: S for d in block.scope}

            def groups_of(scope):
                return {group_of[d] for d in scope}

            for op in block.ops:
                match op:
                    case ("pulse", ins):
                        g = group_of[ins.dest]
                        if ins.start_time < issue[g]:
                            if g in busy:
                                raise ValueError(
                                    f"Pulse on {ins.dest} at {ns(ins.start_time)} "
                                    f"is within {load} clocks of the previous "
                                    f"instruction on its core, which is busy "
                                    f"until {ns(issue[g])} ({block.context})."
                                )
                            raise ValueError(
                                f"Pulse on {ins.dest} at {ns(ins.start_time)} "
                                f"starts before its basic block can begin at "
                                f"{ns(issue[g])} ({block.context}). Pulses may "
                                f"not overlap a measurement-conditioned branch, "
                                f"and a branch's arms cannot start until "
                                f"fproc_meas_clks + pulse_load_clks + "
                                f"jump_fproc_clks after its measurement ends."
                            )
                        busy.add(g)
                        issue[g] = ins.start_time + load
                        end = ins.start_time + nclks(ins.twidth)
                        cur_t[ins.dest] = end
                        pulse_cur[ins.dest] = end
                    case ("hold_idle", idle, channel, branch_start):
                        # Hold until the arms can start exactly at the branch's
                        # nominal start: the jump then reads the state bit as
                        # late as the layout allows, so the gap laid out before
                        # the branch (`fproc_delay` in a reset) is the time the
                        # measurement result has to reach the processor.
                        # `fproc_meas_clks` after the demod window is the floor.
                        idle_groups = groups_of(idle.scope)
                        ready = pulse_cur[channel] + fc.fproc_meas_clks - load
                        latest = branch_start - load - fc.jump_fproc_clks
                        end = max([ready, latest] + [issue[g] for g in idle_groups])
                        idle.end_time = end
                        for g in idle_groups:
                            busy.add(g)
                            issue[g] = end + load
                    case ("idle", ins):
                        for g in groups_of(ins.scope):
                            issue[g] = max(issue[g], ins.end_time + load)
                    case ("hold", channel, scope):
                        idle_end = max(pulse_cur[channel] + fc.fproc_meas_clks, S)
                        for g in groups_of(scope):
                            if issue[g] < idle_end:
                                issue[g] = idle_end + load
                    case ("jump_fproc", scope):
                        for g in groups_of(scope):
                            issue[g] += fc.jump_fproc_clks
                    case ("jump_i", scope):
                        for g in groups_of(scope):
                            issue[g] += fc.jump_cond_clks

            block.end = max([S] + list(cur_t.values()) + list(issue.values()))
            block.pulse_end = pulse_cur

        for block in blocks:
            for ins in block.timed:
                if isinstance(ins, Pulse):
                    ins.start_time -= block.start
                else:
                    ins.end_time -= block.start

        return circuit

    def compile(
        self,
        seq: Sequence,
        batch_size: int | None = None,
        substitutions: dict = {},
        frame_scopes: dict = {},
        proc_grouping: list | None = None,
        reset_delay: float | None = None,
        channel_config: QubicChannelConfig | dict = {},
        **kwargs,
    ) -> QubicExecutable:
        """Compiles a sequence to a Qubic executable format.

        Args:
            seq: The sequence to compile.
            reset_delay: Per-shot period in seconds. If `None` (default), each
                timeline's slot equals its own `width` — appropriate when each
                timeline already includes its own active reset, or when the
                timeline already accounts for passive decay. If a float, every
                timeline gets a uniform slot of `reset_delay` seconds, and
                compilation raises `ValueError` if any timeline is longer than
                this value.

        Returns:
            The resulting compiled `QubicExecutable` that can then be run on hardware.
        """

        # Pre-compilation
        frame_declarations = self.get_frame_declarations(
            frame_scopes or self.frame_scopes
        )

        channel_config = channel_config or self.get_channel_config()
        proc_grouping = proc_grouping or self.get_proc_grouping()
        qb_grouping = list(it.chain(*proc_grouping))

        default_passes = get_passes(
            self.fpga_config,
            qchip=self.get_qchip(),
            compiler_flags=CompilerFlags(
                schedule=False, resolve_gates=False, multi_board=True
            ),
            qubit_grouping=qb_grouping,
            proc_grouping=proc_grouping,
        )
        passes = kwargs.get("passes", default_passes)

        num_timelines = len(seq.flat)
        batch_size = batch_size or num_timelines

        exes = []
        flattened_seq = seq.flatten()

        for tmln_idx in range(0, num_timelines, batch_size):
            batch_seq = flattened_seq[tmln_idx : tmln_idx + batch_size]
            circuit, reads_per_timeline, batch_duration = self.construct_circuit(
                batch_seq,
                reset_delay,
                frame_declarations,
                substitutions,
                **kwargs,
            )

            qubic_compiler = _QubicInternalCompiler(
                circuit, proc_grouping=proc_grouping
            )
            qubic_compiler.run_ir_passes(passes)

            prog = qubic_compiler.compile()
            asm = tc.run_assemble_stage(prog, channel_config)
            for channel, value in self._state_thresholds(batch_seq).items():
                board, register = self._threshold_register(channel)
                asm.write_registers[f"{board}:{register}"] = int(value)
            exe = QubicExecutable(
                sequence=batch_seq,
                timeline_index=tmln_idx,
                program=prog,
                assembly=asm,
                repetition_delay=batch_duration,
                reads_per_timeline=reads_per_timeline,
            )
            exes.append(exe)

        return exes

    @staticmethod
    def _state_thresholds(seq: Sequence) -> dict[str, int]:
        """Gateware state thresholds requested by the resets in ``seq``.

        Every `ResetOperation` with a `threshold` contributes an entry for its
        readout channel. Timelines flatten when composed, so the resets are
        top-level operations of each timeline.

        Raises:
            ValueError: If two resets in the batch ask for different thresholds
                on the same channel; the register is per channel, not per
                timeline, and holds one value for the whole program.
        """
        thresholds: dict[str, int] = {}
        for tmln in seq.flat:
            for op in tmln.operations:
                if not isinstance(op, ResetOperation) or op.threshold is None:
                    continue
                value = int(op.threshold)
                if thresholds.setdefault(op.channel, value) != value:
                    raise ValueError(
                        f"Resets in one batch request different state thresholds "
                        f"on {op.channel!r}: {thresholds[op.channel]} and {value}. "
                        f"The threshold register is per channel; use one value "
                        f"per batch."
                    )
        return thresholds

    def _threshold_register(self, channel: str) -> tuple[str, str]:
        """(board, register name) of the state threshold for ``channel``.

        The gateware names the register after the readout core and signal
        generator, ``thresh_<core>_<siggen>_<index>``, matching the device
        naming used by `get_channel_config`.
        """
        for devname, dev in self.devices.items():
            for ch in dev.channels:
                if ch.name != channel:
                    continue
                if not ch.read:
                    raise ValueError(
                        f"{channel!r} is not a readout channel; cannot set a state threshold"
                    )
                board, core, siggen = _get_board_and_core(devname)
                return board, f"thresh_{core}_{siggen}_{ch.index}"
        raise ValueError(f"No device has a channel named {channel!r}")

    def get_qchip(self) -> QChip:
        """Returns a Qubic QChip object with the named modulation frequencies.

        Returns:
            A Qubic `QChip` instance.
        """
        frames = FlatDict(
            {k.replace(".", "/"): f.offset for k, f in self.frames.items()}
        )

        return QChip(dict(Qubits=frames, Gates=dict()))

    def get_frame_declarations(self, frame_scopes: dict[str, str]) -> list[DeclareFreq]:
        declarations = []

        for frame, freq in self.frames.items():
            if frame not in frame_scopes:
                continue

            ins = DeclareFreq(
                scope=[frame_scopes[frame]], freqname=frame, freq=freq.offset
            )
            declarations.append(ins)

        return declarations

    def get_proc_grouping(self) -> list:
        cores = defaultdict(list)

        for devinfo in self.devices.values():
            board, core, _ = _get_board_and_core(devinfo.name)
            for ch in devinfo.channels:
                cores[board, core, ch.index].append(ch.name)

        return [tuple(grp) for grp in cores.values()]

    def get_channel_config(self) -> dict:
        """Return a QubicChannelConfig with the channel information.

        Returns:
            A Qubic `QubicChannelConfig` instance.
        """
        channel_config = dict(fpga_clk_freq=self.fpga_config.fpga_clk_freq)

        for devname, devinfo in self.devices.items():
            sample_rate = devinfo.sample_rate
            env_sample_rate = devinfo.envelope_sample_rate

            board, core, siggen = _get_board_and_core(devname)

            for ch in devinfo.channels:
                if sample_rate == 0:
                    elem_params = {}
                    elem_type = "dc"
                    memory = {}
                else:
                    samples_per_clk = round(
                        sample_rate * self.fpga_config.fpga_clk_period
                    )
                    interp_ratio = round(sample_rate / env_sample_rate)

                    elem_params = dict(
                        samples_per_clk=samples_per_clk, interp_ratio=interp_ratio
                    )

                    if ch.read:
                        elem_type = "rf_mix"
                    else:
                        elem_type = "rf"

                    memory = _get_memory_name(ch)

                channel_config[ch.name] = QubicChannelConfig(
                    core_ind=ch.index,
                    elem_type=elem_type,
                    elem_ind=ch.subchannel,
                    elem_params=elem_params,
                    core_name=core,
                    board_name=board,
                    **memory,
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
        )[0]

        tmlns = np.r_[
            tuple(
                np.repeat(i + exe.timeline_index, n)
                for i, n in enumerate(exe.reads_per_timeline)
            )
        ]
        reads = np.r_[tuple(np.arange(n) for n in exe.reads_per_timeline)]
        shots = exe.repetition_index + np.arange(repetitions)

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
            df = pd.DataFrame(
                np.array(data).flatten().conj(), index=index, columns=["IQ"]
            )

            name = self.result_map.get(k, k)
            iq_results[name] = IQResult(name=name, data=df)

        return iq_results

    def update_parameters(self, qpu: "QPU", **kwargs):
        """Updates parameters from the QPU.

        This method updates the mapping from readout indices to readout channels.

        Args:
            qpu: A QPU instance.
        """

        ro_config = qpu.db.config.readout[qpu.pipeline.name]
        for reg_name, reg_info in ro_config.registers.items():
            self.result_map.update({reg_info.channel: reg_name})

        # ch_reg_mapping = {
        #     reg_info.channel: reg_name for reg_name, reg_info in ro_config.registers.items()
        # }

        # for devices in qpu.compiler.devices.values():
        #     for ch_info in devices.channels:
        #         if not ch_info.read:
        #             continue

        #         self.result_map.update({str(ch_info.index): ch_reg_mapping.get(ch_info.name, ch_info.name)})
