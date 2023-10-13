from typing import TYPE_CHECKING

import numpy as np
from attrs import cmp_using, field
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


def _weights_equal(a, b):
    if len(a) != len(b):
        return False

    return all(np.array_equal(a_i, b_i, equal_nan=True) for a_i, b_i in zip(a, b))


@qfrozen
class HeterodyneProgram(Program):
    demods: np.ndarray = field(eq=cmp_using(np.array_equal))
    weights: tuple[np.ndarray, ...] = field(eq=cmp_using(_weights_equal))
    keys: tuple[str, ...]

    def __hash__(self) -> int:
        values = (
            self.device,
            self.demods.tobytes(),
            tuple(w.tobytes() for w in self.weights),
            self.keys,
        )

        return hash(values)


@qdefine
class HeterodyneCompiler(HardwareCompiler):
    def compile(
        self,
        program: IntermediateProgram,
        device: DeviceInfo,
    ) -> HeterodyneProgram:
        """Compiles a set of demodulation weights from the given program.

        Args:
            program: The sequence to compile.
            location_kwargs: Any location constraints to add to the sequence
                before compilation.
            pulse_kwargs: A mapping of variables names to resolved pulse parameters.

        Returns:
            A `HeterodyneCompiler` instance.
        """
        # Map channel index numbers to named measurement keys
        index_key_map = {}
        for ch in sorted(device.channels, key=lambda ch: ch.index):
            if ch.index in index_key_map:
                raise ValueError(
                    f"Device '{device.name}' has duplicate channel indices: {device.channels}"
                )
            index_key_map[ch.index] = ch.name

        used_channels = set()
        for wmem in program.waveforms:
            used_channels.update((ch for ch, _ in wmem.data))

        index_column_map = {ch: col for col, ch in enumerate(sorted(used_channels))}
        n_channels = len(used_channels)

        weights = []
        for wmem in program.waveforms:
            weights.append(
                np.full((n_channels, wmem.samples), np.nan, dtype=np.complex64)
            )

            for (ch, subch), arr in wmem.data.items():
                weights[-1][index_column_map[ch], :] = arr

            weights[-1].flags.writeable = False

        demods = []
        for ins in program.instructions:
            match ins:
                case PlayInstruction(waveform_index=wf_index):
                    demods.append(wf_index)

        demods = np.array(demods)
        demods.flags.writeable = False

        keys = tuple(index_key_map[ch] for ch in sorted(used_channels))

        return HeterodyneProgram(
            device=device.name, demods=demods, weights=weights, keys=keys
        )
