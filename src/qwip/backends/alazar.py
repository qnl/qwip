import numpy as np
import pandas as pd
from attrs import field
from loguru import logger

from qwip.attrs import qdefine, qfrozen
from qwip.backends.backend import ADCBackend
from qwip.processing.processors import IQTraceResult
from qwip.sequencer.compilation import (
    DeviceInfo,
    HardwareCompiler,
    IntermediateProgram,
    Program,
    QWiPExecutable,
    ReadInstruction,
)

try:
    from qwip.instruments.alazar.alazar import Alazar
except FileNotFoundError:
    logger.warning("Unable to import Alazar. Check that ATSApi64.dll is installed.")


def round_samples(samples: int) -> int:
    """Round number of samples for alazar compatibility.

    The Alazar card requires the number of samples to be at least 256 and also
    a multiple of 128.

    Returns:
        The next largest valid number of samples.
    """
    return max(256, (samples + 127 >> 7) << 7)


@qdefine
class AlazarProgram(Program):
    samples: list[int] = field(factory=list)


@qdefine
class AlazarCompiler(HardwareCompiler):
    def compile(
        self,
        program: IntermediateProgram,
        device: DeviceInfo,
    ) -> AlazarProgram:
        alazar_program = AlazarProgram(device=device.name)
        for ins in program.instructions:
            match ins:
                case ReadInstruction(samples=samples):
                    alazar_program.samples.append(samples)

        return alazar_program


@qdefine
class AlazarBackend(ADCBackend):
    device: "Alazar"
    exe: QWiPExecutable | None = None
    samples: int = 2048
    num_reads: int = 1

    @property
    def sample_rate(self) -> float:
        return self.device.get_true_sample_rate()

    @sample_rate.setter
    def sample_rate(self, value) -> None:
        self.device.capture_clock_settings(sample_rate=value)

    def get_record_size(self, samples: int) -> int:
        min_len, incr = self.device.board.record_limits

        samples = max(min_len, samples)
        return np.ceil(samples / incr).astype(int) * incr

    def acquire(self, **kwargs) -> np.ndarray:
        arr = self.device.acquire()
        arr = (
            np.ascontiguousarray(arr.transpose(1, 2, 3, 0).astype(np.float32))
            .view(np.complex64)
            .squeeze()
        )

        num_shots, num_readouts, num_samples = arr.shape

        shots = np.arange(num_shots)
        elems = np.r_[tuple(np.repeat(i, n) for i, n in enumerate(self.exe.num_reads))]
        reads = np.r_[tuple(np.arange(n) for n in self.exe.num_reads)]
        ts = np.arange(num_samples) / self.sample_rate

        idx = pd.MultiIndex.from_arrays(
            [
                np.repeat(shots, num_readouts),
                np.tile(elems, num_shots),
                np.tile(reads, num_shots),
            ],
            names=["shot", "element", "readout"],
        )

        data = (
            pd.DataFrame(
                arr.reshape(-1, num_samples),
                index=idx,
                columns=pd.Index(ts, name="time"),
            )
            .stack()
            .to_frame()
        )

        return {self.device.name: IQTraceResult(name=self.device.name, data=data)}

    def upload(self, exe: QWiPExecutable):
        self.exe = exe
        program = exe.programs[self.device.name]
        self.num_reads = sum(exe.num_reads)
        self.samples = self.get_record_size(max(program.samples))

    def start(self, repetitions: int, **kwargs) -> None:
        self.device.stop()
        self.device.start_acquire(
            self.samples, n_repetitions=repetitions, expected_triggers=self.num_reads
        )

    def stop(self):
        self.device.stop()

    def update_parameters(self, qpu: "QPU", **kwargs):
        ...
