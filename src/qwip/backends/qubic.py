from collections import Counter, defaultdict

import attrs
import cattrs
import numpy as np
from attrs import field

try:
    import qubic.rpc_client as rc
    import qubic.toolchain as tc
    from distproc.compiler import CompiledProgram
    from distproc.hwconfig import FPGAConfig, load_channel_configs
    from qubic.rpc_client import CircuitRunnerClient
except ImportError:
    ...

import qwip
from qwip.attrs import qdefine, qfrozen
from qwip.backends.backend import QuantumBackend
from qwip.sequencer.compilation import (
    CompiledSequence,
    WaveformSequencer,
    find_end_marker,
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


@qdefine
class QubicExecutable:
    programs: list[CompiledProgram]
    assembly: list[list]
    repetition_delay: float


@qdefine
class QubicCompiler(WaveformSequencer):
    """Qubic-specific compiler"""

    fpga_config: QubicFPGAConfig = field(factory=QubicFPGAConfig)
    reset_delay: float = 500e-6
    interleave: bool = False

    def compile_instruction(
        self,
        loc: Location,
        wave: Waveform,
        unique_waveforms: dict[Waveform, np.ndarray],
        channel_times: dict[str, Location],
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

        if channel and start > channel_times[channel]:
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
                    if (ch_info := self.get_channel_info(channel)) is None:
                        raise ValueError(f"Channel {channel} is not a valid channel.")
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
    ):
        """Compiles a single sequence element."""
        instructions = []
        channel_times = defaultdict(Location)

        unique_waveforms = dict()

        for loc, waves in locations.items():
            for w in waves:
                instructions.extend(
                    self.compile_instruction(
                        loc=loc,
                        wave=w,
                        unique_waveforms=unique_waveforms,
                        channel_times=channel_times,
                        pulse_kwargs=pulse_kwargs,
                    )
                )

        return instructions

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
        batch = []
        for i, (se_locs, se) in enumerate(zip(locations, seq.flat)):
            t_se = find_end_marker(se_locs, self.end_marker).offset + reset_delay

            if self.interleave:
                repetition_delay += t_se
            else:
                repetition_delay = max(repetition_delay, t_se)

            instructions = self.compile_sequence_element(
                se_locs,
                se.constraints | pulse_kwargs,
            )

            reset = DelayInstruction(t=reset_delay)

            circuit = [reset.todict(), *(ins.todict() for ins in instructions)]
            if self.interleave:
                batch.extend(circuit)
            else:
                batch.append(circuit)

        if self.interleave:
            # batch is really a single instruction list in this case
            batch = [batch]

        return batch


@qdefine
class QubicBackend(QuantumBackend):
    """A hardware backend for interfacing with qubic."""

    runner: CircuitRunnerClient

    def upload(self, cseq: CompiledSequence, **kwargs) -> None:
        ...
        # self.meta.write_sequence(cseq)

    def acquire(self, cseq: CompiledSequence, repetitions: int = 512, **kwargs) -> dict:
        # acquisition_kwargs = dict(n_reps=repetitions, save_data=False) | kwargs

        # meas = self.meta.acquire(**acquisition_kwargs)
        # iqdata = {
        #     k: meas[k]["Heterodyne"] for k in meas.keys() if re.match(r"R(\d+)", k)
        # }
        # return iqdata
        ...

    def update_parameters(self, qpu: "QPU", **kwargs):
        """Updates parameters from the QPU.

        The QTRL backend requires that all readout frequencies are specified in the
        variables config because this is where the ADCManager pulls the demod
        frequencies from.
        """
        # for sys in qpu.subsystems.values():
        #     match sys:
        #         case ReadoutResonator(name=n, frequency=f):
        #             if not (m := re.match(r"R(\d+)", n)):
        #                 raise ValueError(
        #                     f"Resonator names must be of the form 'R\\d+' for the "
        #                     f"QTRL backend. Got {n}"
        #                 )

        #             self.meta.variables[f"Q{m[1]}/res_freq"] = f
        ...
