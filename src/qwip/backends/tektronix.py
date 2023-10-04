from typing import TYPE_CHECKING

import numpy as np
from attrs import field
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
from typing_extensions import Self

import qwip
from qwip.attrs import qdefine, qfrozen
from qwip.backends.backend import DACBackend, QuantumBackend
from qwip.sequencer.compilation import (
    DelayInstruction,
    DeviceInfo,
    HardwareCompiler,
    IntermediateProgram,
    PlayInstruction,
    Program,
    QuantumExecutable,
    QWiPCompiler,
    QWiPExecutable,
    WaitTriggerInstruction,
)
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.utils import Location

if TYPE_CHECKING:
    from qwip.qpu.qpu import QPU


@qfrozen
class TektronixProgram(Program):
    channels: tuple[int, ...]
    waveforms: tuple[list[np.ndarray], ...] = field()
    marker1s: tuple[list[np.ndarray], ...] = field()
    marker2s: tuple[list[np.ndarray], ...] = field()
    repeats: list[int] = field(factory=list)
    waits: list[int] = field(factory=list)
    go_tos: list[int] = field(factory=list)
    jump_tos: list[int] = field(factory=list)

    @waveforms.default
    @marker1s.default
    @marker2s.default
    def _default_list(self) -> tuple[list[np.ndarray]]:
        return tuple([] for _ in self.channels)


@qdefine
class TektronixCompiler(HardwareCompiler):
    def compile(
        self,
        program: IntermediateProgram,
        device: DeviceInfo,
    ) -> TektronixProgram:
        """Compiles a sequence.

        This function takes an abstract sequence and compiles it into a concrete
        set of timepoints.

        Args:
            seq: The sequence to compile.
            location_kwargs: Any location constraints to add to the sequence
                before compilation.
            pulse_kwargs: A mapping of variables names to resolved pulse parameters.

        Returns:
            A `TektronixExecutable` instance.
        """

        indices = sorted(device.channel_indices())
        tek_program = TektronixProgram(device=device.name, channels=indices)

        waves = tuple([] for _ in tek_program.channels)
        m1s = tuple([] for _ in tek_program.channels)
        m2s = tuple([] for _ in tek_program.channels)

        for wmem in program.waveforms:
            for ch in range(len(tek_program.channels)):
                default = np.zeros(wmem.samples)
                wave = wmem.data.get((ch, 0), default)
                m1 = (wmem.data.get((ch, 1), default) != 0).astype(np.uint8)
                m2 = (wmem.data.get((ch, 2), default) != 0).astype(np.uint8)

                waves[ch].append(wave)
                m1s[ch].append(m1)
                m2s[ch].append(m2)

        element_index = 0
        for ins in program.instructions:
            match ins:
                case WaitTriggerInstruction():
                    wait = True
                    element_index += 1
                case PlayInstruction(waveform_index=w_idx):
                    for ch in range(len(tek_program.channels)):
                        tek_program.waveforms[ch].append(waves[ch][w_idx])
                        tek_program.marker1s[ch].append(m1s[ch][w_idx])
                        tek_program.marker2s[ch].append(m2s[ch][w_idx])

                    tek_program.repeats.append(1)
                    tek_program.waits.append(1 if wait else 0)
                    tek_program.jump_tos.append(0)

                    wait = False

        tek_program.go_tos[:] = np.roll(np.arange(len(tek_program.waits)) + 1, -1)

        return tek_program


@qdefine
class TektronixChannel:
    index: int
    amplitude: float = 2.0
    offset: float = 0.0
    marker_1: tuple[float, float] = (0, 2.0)
    marker_2: tuple[float, float] = (0, 2.0)

    def read_settings(self, awg: Tektronix_AWG5014) -> None:
        on_device = type(self).from_awg(awg, self.index)

        self.amplitude = on_device.amplitude
        self.offset = on_device.offset
        self.marker_1 = on_device.marker_1
        self.marker_2 = on_device.marker_2

    def write_settings(self, awg: Tektronix_AWG5014) -> None:
        ch = self.index + 1

        settings = {
            f"ch{ch}_amp": self.amplitude,
            f"ch{ch}_offset": self.offset,
            f"ch{ch}_m1_low": self.marker_1[0],
            f"ch{ch}_m1_high": self.marker_1[1],
            f"ch{ch}_m2_low": self.marker_2[0],
            f"ch{ch}_m2_high": self.marker_2[1],
        }

        for param, value in settings.items():
            getattr(awg, param)(value)

    @classmethod
    def from_awg(cls, awg: Tektronix_AWG5014, index: int) -> Self:
        ch = index + 1

        markers = [
            (getattr(awg, f"ch{ch}_m{m}_low")(), getattr(awg, f"ch{ch}_m{m}_high")())
            for m in range(1, 3)
        ]

        return cls(
            index=index,
            amplitude=getattr(awg, f"ch{ch}_amp")(),
            offset=getattr(awg, f"ch{ch}_offset")(),
            marker_1=markers[0],
            marker_2=markers[1],
        )


@qdefine
class TektronixBackend(DACBackend):
    """A hardware backend for interfacing with the Tektronix AWGs."""

    device: Tektronix_AWG5014
    channels: tuple[TektronixChannel, ...] = field()
    reset_delay: float = 100e-6

    @channels.default
    def _default_channels(self) -> tuple[TektronixChannel, ...]:
        return tuple(
            TektronixChannel.from_awg(self.device, index=i)
            for i in range(self.device.num_channels)
        )

    @property
    def num_channels(self) -> int:
        return self.device.num_channels

    @property
    def sample_rate(self) -> float:
        self.device.clock_freq()

    @classmethod
    def connect(cls, ip: str, name: str = "tektronix", **kwargs) -> Self:
        address = f"TCPIP0::{ip}::inst0::INSTR"

        dev = Tektronix_AWG5014(name=name, address=address)
        dev.run_mode("SEQ")
        dev.trigger_source("INT")
        backend = cls(device=dev, **kwargs)

        return backend

    def start(self):
        self.device.stop()
        self.device.all_channels_on()
        self.device.start()

    def stop(self):
        self.device.stop()

    def update_parameters(self, qpu, **kwargs):
        ...

    def upload(self, exe: QWiPExecutable):
        program = exe.programs[self.device.name]
        self.device.make_send_and_load_awg_file(
            waveforms=list(program.waveforms),
            m1s=list(program.marker1s),
            m2s=list(program.marker2s),
            nreps=program.repeats,
            trig_waits=program.waits,
            goto_states=program.go_tos,
            jump_tos=program.jump_tos,
            channels=[ch + 1 for ch in program.channels],
        )

        # Expects times in nanoseconds
        self.device.trigger_seq_timer(exe.reset_delay * 1e9)
