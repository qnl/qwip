from collections import Counter, defaultdict

import attrs
import cattrs
import numpy as np
import pandas as pd
from attrs import field
from loguru import logger

try:
    import qubic.rpc_client as rc
    import qubic.toolchain as tc
    from distproc.compiler import CompiledProgram
    from distproc.hwconfig import FPGAConfig, load_channel_configs
    from qubic.rpc_client import CircuitRunnerClient
    from qubitconfig.qchip import QChip
except ImportError:
    ...

import qwip
from qwip.attrs import qdefine, qfrozen
from qwip.backends.backend import QuantumBackend
from qwip.flatdict import FlatDict
from qwip.processing.processors import IQResult
from qwip.sequencer.compilation import (
    WaveformSequencer,
    find_end_marker,
    register_sequencer,
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


@qfrozen
class QubicInstruction:
    """A single instruction that is appended to a qubic circuit."""

    name: str

    def todict(self) -> dict:
        """Converts a QubicInstruction instance to a dictionary for compilation."""

        return qwip.converter.unstructure(self)


@qfrozen
class PulseInstruction(QubicInstruction):
    """A pulse instruction.

    Attributes:
        env: The waveform envelope. Can be either a dictionary specifying a qubic
            waveform or a numpy array of envelope sample timepoints.
        dest: The destination channel
        freq: The modulation frequency of the pulse.
        phase: The phase offset of the pulse.
        amp: The amplitude of the pulse.
        twidth: The width of the pulse in seconds.
    """

    name: str = field(default="pulse", init=False, metadata=dict(serialize=True))
    env: np.ndarray | dict = field(
        eq=id,
        metadata=dict(
            unstructure_override=cattrs.override(unstruct_hook=lambda arr: arr)
        ),
    )
    dest: str
    freq: float | str = field(default=0, metadata=dict(serialize=True))
    phase: float = field(default=0, metadata=dict(serialize=True))
    amp: float = field(default=1, metadata=dict(serialize=True))
    twidth: float = field(default=0, metadata=dict(serialize=True))


@qfrozen
class VirtualZInstruction(QubicInstruction):
    name: str = field(default="virtualz", init=False, metadata=dict(serialize=True))
    qubit: tuple[str]
    phase: float = field(default=0, metadata=dict(serialize=True))
    freqname: str = "freq"


@qfrozen
class BarrierInstruction(QubicInstruction):
    """"""

    name: str = field(default="barrier", init=False, metadata=dict(serialize=True))
    qubits: tuple[int, ...]


@qfrozen
class DelayInstruction(QubicInstruction):
    """Adds a delay on one or more channels.

    Attributes:
        qubits: One or more channels on which to add a delay. If `None`, a delay is
            added on all channels.
        t: The delay time in seconds.
    """

    name: str = field(default="delay", init=False, metadata=dict(serialize=True))
    qubits: tuple[str, ...] | None = field(
        default=None,
        metadata=dict(unstructure_override=cattrs.override(omit_if_default=True)),
    )
    t: float = field(default=0, metadata=dict(serialize=True))


@qfrozen
class QubicFPGAConfig:
    """Qubic FPGA config.

    This class maps onto a qubic `FPGAConfig` class. Holds information about the timing
    parameters of the onboard FPGA.
    """

    alu_instr_clks: int = 5
    fpga_clk_freq: float = 500e6
    fpga_clk_period: float = 2e-9
    jump_cond_clks: int = 5
    jump_fproc_clks: int = 5
    pulse_regwrite_clks: int = 5


@qfrozen
class QubicChannelConfig:
    """A Qubic channel config object.

    Attributes:
        group: The channel group that the channel belongs to. Should be one of
            `[qdrv, rdrv, rdlo]`.
        core_ind: The core index for the channel.
        elem_ind: The element index for the channel.
        elem_params: A dictionary with the clock samples and interpolation ratio.
        env_mem_name: The name of the envelope memory for the channel.
        freq_mem_name: The name of the frequency memory for the channel.
        acc_mem_name: The name of the acc buffer for the channel.
    """

    group: str
    core_ind: int
    elem_ind: int = 0
    elem_params: dict[str, int] = dict(samples_per_clk=16, interp_ratio=1)
    env_mem_name: str = field()
    freq_mem_name: str = field()
    acc_mem_name: str = field()

    @env_mem_name.default
    def _default_env_mem_name(self) -> str:
        return f"{self.group}env{self.core_ind}"

    @freq_mem_name.default
    def _default_freq_mem_name(self) -> str:
        return f"{self.group}freq{self.core_ind}"

    @acc_mem_name.default
    def _default_acc_mem_name(self) -> str:
        return f"accbuf{self.core_ind}"


@qfrozen
class QubicExecutable:
    program: CompiledProgram = field(eq=id)
    # cattrs will always copy a dict when converting, so we disable autoconversion
    # to allow QubicExecutable's to be copied with the exact same assembly
    assembly: dict = field(eq=id, metadata=dict(auto_convert=False))
    repetition_delay: float
    reads_per_element: tuple[int, ...] = tuple()

    @property
    def total_reads(self) -> int:
        return sum(self.reads_per_element)

    @property
    def num_elements(self) -> int:
        return len(self.reads_per_element)


@register_sequencer
@qdefine
class QubicSequencer(WaveformSequencer):
    """Qubic-specific compiler"""

    fpga_config: QubicFPGAConfig = field(factory=QubicFPGAConfig)
    reset_delay: float = 500e-6

    def compile_instruction(
        self,
        loc: Location,
        wave: Waveform,
        unique_waveforms: dict[Waveform, np.ndarray],
        channel_times: dict[str, Location],
        reads: Counter[str],
        pulse_kwargs: dict = {},
    ) -> list[QubicInstruction]:
        """Compiles a single waveform into a list of qubic instructions"""

        instructions = []

        width = wave.width.resolve(**pulse_kwargs)
        start, end = loc.offset, loc.offset + width.offset

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

            if start > channel_times[channel]:
                delay = start - channel_times[channel]
                instructions.append(DelayInstruction(t=delay.offset, qubits=(channel,)))
                channel_times[channel] = end

        match wave:
            case VirtualZWaveform(mod_key=mod_key, phase=phase):
                if mod_key.references:
                    raise ValueError(
                        f"Qubic does not support vector mod keys, got {mod_key}."
                    )

                if not isinstance(mod_key.offset, str):
                    raise ValueError(
                        f"Virtual Z mod_keys must be named on Qubic, got {mod_key}"
                    )

                match mod_key.offset.split("."):
                    case (qubit, *freqname):
                        qubit = (qubit,)
                        freqname = ".".join(freqname)
                    case qubit:
                        qubit = (qubit,)
                        freqname = None

                instructions.append(
                    VirtualZInstruction(qubit=qubit, phase=phase, freqname=freqname)
                )

            case ModulatedWaveform(envelope=env, modulation=mod):
                # First check if we've evaluated this envelope already
                if env in unique_waveforms:
                    w_t = unique_waveforms[env]
                else:
                    sample_rate = self.channels[ch_info.group].sample_rate

                    N = np.ceil(width.offset * sample_rate).astype(int)
                    ts_wave = np.arange(N) / sample_rate
                    w_t = wave(ts_wave, modulations=self.modulations, **pulse_kwargs)

                    unique_waveforms[env] = w_t

                if mod.hardware_modulation:
                    if mod.frequency.references:
                        raise ValueError(
                            f"Qubic does not support vector mod keys, got {mod.frequency}."
                        )

                    freq = mod.frequency.offset
                    phase = mod.phase
                    amplitude = mod.amplitude
                else:
                    freq = phase = 0
                    amplitude = 1

                ins = PulseInstruction(
                    env=w_t.astype(np.float64),
                    dest=channel,
                    freq=freq,
                    phase=phase,
                    amp=amplitude,
                    twidth=width.offset,
                )
                instructions.append(ins)

            case Marker():
                ...

            case BasicWaveform():
                if env in unique_waveforms:
                    w_t = unique_waveforms[env]
                else:
                    ch_info = self.get_channel_info(channel)
                    sample_rate = ch_info.group.sample_rate

                    N = np.ceil(width * sample_rate).astype(int)
                    ts_wave = np.arange(N) / sample_rate
                    w_t = wave(ts_wave, modulations=self.modulations, **pulse_kwargs)

                    unique_waveforms[env] = w_t

                ins = PulseInstruction(
                    env=w_t.astype(np.float64), dest=channel, twidth=width.offset
                )
                instructions.append(ins)

        return instructions

    def compile_sequence_element(
        self,
        locations: dict[Location, list[Waveform]],
        pulse_kwargs: dict = {},
    ) -> tuple[list[QubicInstruction], Counter[str]]:
        """Compiles a single sequence element."""
        instructions = []
        unique_waveforms = dict()
        channel_times = defaultdict(Location)
        reads = Counter()

        for loc, waves in locations.items():
            for w in waves:
                instructions.extend(
                    self.compile_instruction(
                        loc=loc,
                        wave=w,
                        unique_waveforms=unique_waveforms,
                        channel_times=channel_times,
                        reads=reads,
                        pulse_kwargs=pulse_kwargs,
                    )
                )

        return instructions, reads

    def compile(
        self,
        seq: Sequence,
        location_kwargs: dict = {},
        pulse_kwargs: dict = {},
        **kwargs,
    ) -> QubicExecutable:
        locations = [
            se.resolve_locations(end_marker=self.end_marker, **location_kwargs)
            for se in seq.flat
        ]

        reset_delay = kwargs.get("reset_delay", self.reset_delay)

        repetition_delay = 0
        reads_per_element = []
        circuit = []
        for i, (se_locs, se) in enumerate(zip(locations, seq.flat)):
            t_se = find_end_marker(se_locs, self.end_marker).offset + reset_delay
            repetition_delay += t_se

            instructions, reads = self.compile_sequence_element(
                se_locs,
                se.constraints | pulse_kwargs,
            )

            reads_per_channel = set(cts[1] for cts in reads.most_common())
            if len(reads_per_channel) > 1:
                logger.warning(
                    f"Sequence element {i} has an unequal number of reads accross"
                    f"channels. {reads}"
                )

            reset = DelayInstruction(t=reset_delay)

            sub_circuit = [reset.todict(), *(ins.todict() for ins in instructions)]
            circuit.extend(sub_circuit)
            reads_per_element.append(max(reads_per_channel))

        qchip = self.get_qchip()
        channel_config = self.get_channel_config()

        prog = tc.run_compile_stage(circuit, self.fpga_config, qchip)
        asm = tc.run_assemble_stage(prog, channel_config)
        return QubicExecutable(
            program=prog,
            assembly=asm,
            repetition_delay=repetition_delay,
            reads_per_element=reads_per_element,
        )

    def get_qchip(self) -> QChip:
        """Returns a Qubic QChip object with the named modulation frequencies."""
        modulations = FlatDict(
            {k.replace(".", "/"): f.offset for k, f in self.modulations.items()}
        )

        return QChip(dict(Qubits=modulations, Gates=dict()))

    def get_channel_config(self) -> dict:
        """"""
        channel_config = dict(fpga_clk_freq=self.fpga_config.fpga_clk_freq)

        for ch_group in self.channels.values():
            sample_rate = ch_group.sample_rate

            for ch in ch_group.channels:
                _, group = ch.name.split(".")
                samples_per_clk = 4 if group.lower() == "rdlo" else 16
                interp_ratio = round(
                    samples_per_clk / (sample_rate * self.fpga_config.fpga_clk_period)
                )

                channel_config[ch.name] = QubicChannelConfig(
                    group=group,
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
    delay_buffer: float = 50e-6
    uploaded: QubicExecutable | None = None
    result_map: dict = field(factory=dict)

    def upload(self, exe: QubicExecutable, **kwargs) -> None:
        """Loads a circuit onto the qubic board.

        Args:
            exe: A `QubicExecutable` which contains the assembly dict to load.
            kwargs: Additional keyword arguments are passed to `CircuitRunner.load_circuit`.
        """
        self.runner.load_circuit(exe.assembly, **kwargs)
        self.uploaded = exe

    def acquire(
        self, exe: QubicExecutable | None = None, repetitions: int = 512, **kwargs
    ) -> dict[str, IQResult]:
        """Acquires data from the qubic board.

        Args:
            exe: A `QubicExecutable` to run. If `None`, the last uploaded program is run.
            repetitions: The number of shots to acquire for the given circuit.
        """
        if exe is None and self.uploaded is None:
            raise ValueError(
                "No asm to execute! Call upload to load an executable or pass an "
                "executable to acquire."
            )

        if exe != self.uploaded:
            self.runner.load_circuit(exe.assembly)
            self.uploaded = exe

        exe = self.uploaded

        delay_per_shot = exe.repetition_delay + self.delay_buffer * exe.num_elements
        result = self.runner.run_circuit(
            repetitions,
            reads_per_shot=exe.total_reads,
            delay_per_shot=delay_per_shot,
        )

        num_se = exe.num_elements

        iq_results = {}
        for k, data in result.items():
            index = pd.MultiIndex.from_tuples(
                (
                    (se, ro)
                    for se in range(num_se)
                    for ro in range(exe.reads_per_element[se])
                ),
                names=["element", "readout"],
            )

            columns = pd.RangeIndex(repetitions, name="shot")
            df = pd.DataFrame(data.T, index=index, columns=columns)

            name = self.result_map.get(k, k)
            iq_results[name] = IQResult(name=name, data=df)

        return iq_results

    def update_parameters(self, qpu: "QPU", **kwargs):
        """Updates parameters from the QPU.

        This method updates the mapping from readout indices to readout channels.
        """

        for groups in qpu.sequencer.channels.values():
            for ch_info in groups.channels:
                if not ch_info.read:
                    continue

                self.result_map.update({str(ch_info.index): ch_info.name})
