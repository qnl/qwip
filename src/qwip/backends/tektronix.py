import itertools as it
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
    """A representation of a tektronix sequencer program.

    Most of the attributes are passed directly to the `make_send_and_load_awg_file`
    method of the QCoDeS driver.

    Attributes:
        channels: The channels indices that are involved. Note that these are
            zero-indexed, but the Tektronix channels are 1-indexed on the device itself.
        waveforms: A tuple of lists of numpy arrays containing the waveform data. The
            outermost tuple indexes the channels and each list indexes the sequence
            element.
        marker1s: A tuple of lists of numpy arrays containing the values for the first
            marker on each channel, indexed the same way as waveforms. These values
            should either be 0 or 1.
        marker2s: A tuple of lists of numpy arrays containing the values for the second
            marker on each channel, indexed the same way as waveforms. These values
            should either be 0 or 1.
        repeats: The number of times to repeat the corresponding element.
        waits: Whether or not to wait for a trigger before playing the corresponding
            element.
        go_tos: The element to go to after completing the current sequence element.
        jump_tos: The event jump value for each element. Should be 0 for most cases.
    """

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

    def validate(self, raise_on_error: bool = False) -> bool:
        """Validates a program.

        The minimum number of samples for every element on the 5014c is 250, discovered
        via experimental programming.

        Args:
            raise_on_error: Whether to raise an exception or just return False.

        Raises:
            ValueError: If the program is invalid in any way.
        """
        channels = (
            len(self.channels)
            == len(self.waveforms)
            == len(self.marker1s)
            == len(self.marker2s)
        )

        if not channels:
            if raise_on_error:
                raise ValueError(
                    "The number of channels does not match across channels, waveforms, "
                    "and markers."
                )
            return False

        args = [self.repeats, self.waits, self.go_tos, self.jump_tos]
        elements = {
            len(l) for l in it.chain(self.waveforms, self.marker1s, self.marker2s, args)
        }

        if len(elements) != 1:
            if raise_on_error:
                raise ValueError("All lists must have the same number of elements.")
            return False

        num_elements = next(iter(elements))

        for i in range(num_elements):
            timesteps = {
                len(arrs[i])
                for arrs in it.chain(self.waveforms, self.marker1s, self.marker2s)
            }

            if len(timesteps) != 1:
                if raise_on_error:
                    raise ValueError(
                        f"Waveforms and markers for element {i} have a different number "
                        f"of samples."
                    )
                return False

            if (nsamples := next(iter(timesteps))) < 250:
                if raise_on_error:
                    raise ValueError(f"Element {i} has {nsamples} < 250 samples.")

        return True


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
            program: The intermediate representation to be compiled.
            device: Device information for the tektronix instrument being targeted.

        Returns:
            A `TektronixProgram` instance.
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
    """A hardware DAC backend for interfacing with the Tektronix AWGs.

    Attributes;
        device: The QCoDeS instrument corresponding to the device.
        channels: The channel information representing each channel.
    """

    device: Tektronix_AWG5014
    channels: tuple[TektronixChannel, ...] = field()

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
        return self.device.clock_freq()

    @sample_rate.setter
    def sample_rate(self, sample_rate: float):
        self.device.clock_freq(sample_rate)

    @property
    def active(self) -> bool:
        return self.device.state().lower() != "idle"

    @classmethod
    def connect(cls, ip: str, name: str = "tektronix", **kwargs) -> Self:
        """Creates a new backend instance from an IP address.

        Args:
            ip: The IP address for the Tektronix instrument.
            name: A unique name for the device. This should match the name specified
                in the configuration database.

        Returns:
             A new `TektronixBackend` instance.
        """
        address = f"TCPIP0::{ip}::inst0::INSTR"

        dev = Tektronix_AWG5014(name=name, address=address)
        dev.run_mode("SEQ")
        dev.trigger_source("INT")
        backend = cls(device=dev, **kwargs)

        return backend

    def start(self) -> None:
        """Starts the tektronix and turns on the output for all channels."""
        self.device.stop()
        self.device.all_channels_on()
        self.device.start()

    def stop(self) -> None:
        """Stops the tektronix and turns off the output for all channels."""
        self.device.stop()
        self.device.all_channels_off()

    def update_parameters(self, qpu: "QPU", **kwargs) -> None:
        """Updates device parameters from the QPU.

        This will update the sample rate of the Tektronix to match the specified sample
        rate in the device info.

        Args:
            qpu: A `QPU` instance.
        """
        self.sample_rate = qpu.compiler.channels[self.device.name].sample_rate

    def upload(self, exe: QWiPExecutable) -> None:
        """Uploads an executable to the AWG.

        Args:
            exe: A `QWiPExecutable` containing the `TektronixProgram` to upload. The
                program should be keyed by the device name.
        """
        program = exe.programs[self.device.name]
        program.validate(raise_on_error=True)
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
