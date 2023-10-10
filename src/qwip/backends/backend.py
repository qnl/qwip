from abc import ABCMeta, abstractmethod, abstractproperty
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from attrs import cmp_using, field
from numpy.random import Generator, default_rng

from qwip.attrs import _numpy_equals, qdefine
from qwip.processing.processors import GMMClassification, IQResult
from qwip.sequencer.compilation import QuantumExecutable, QWiPExecutable

if TYPE_CHECKING:
    from qwip.qpu.qpu import QPU


class DACBackend(metaclass=ABCMeta):
    @abstractproperty
    def sample_rate(self) -> float:
        ...

    @abstractmethod
    def upload(self, exe: QuantumExecutable, **kwargs) -> None:
        ...

    @abstractmethod
    def start(self, **kwargs) -> None:
        ...

    @abstractmethod
    def stop(self, **kwargs) -> None:
        ...


class ADCBackend(metaclass=ABCMeta):
    @abstractproperty
    def sample_rate(self) -> float:
        ...

    @abstractmethod
    def upload(self, exe: QuantumExecutable, **kwargs) -> None:
        ...

    @abstractmethod
    def start(self, **kwargs) -> None:
        ...

    @abstractmethod
    def stop(self, **kwargs) -> None:
        ...

    def acquire(self, **kwargs) -> np.ndarray:
        ...


class QuantumBackend(metaclass=ABCMeta):
    uploaded: QuantumExecutable | None = None

    @abstractmethod
    def upload(self, exe: QuantumExecutable, **kwargs) -> None:
        ...

    @abstractmethod
    def acquire(self, **kwargs) -> dict:
        ...

    @abstractmethod
    def update_parameters(self, qpu: "QPU", **kwargs):
        ...

    @property
    def exe_formats(self) -> set[type[QuantumExecutable]]:
        return set()


@qdefine
class QWiPBackend(QuantumBackend):
    """A backend for a heterogeneous hardware setup.

    This backend is a general backend for working with heterogeneous measurement setups
    where the different components (DAC/ADC) come from different vendors.
    """

    dac: DACBackend
    adc: ADCBackend

    def upload(self, exe: QuantumExecutable, **kwargs) -> None:
        self.dac.upload(exe, **kwargs)
        self.adc.upload(exe, **kwargs)

    def acquire(self, repetitions: int | None = None, **kwargs) -> dict:
        self.adc.start(repetitions=repetitions, **kwargs)
        self.dac.start(**kwargs)

        results = self.adc.acquire(**kwargs)
        self.dac.stop()
        self.adc.stop()

        return results

    def update_parameters(self, qpu: "QPU", **kwargs):
        self.dac.update_parameters(qpu, **kwargs)
        self.adc.update_parameters(qpu, **kwargs)


def random_data_sampler(
    num_states: int = 2,
    rng: Generator = default_rng(),
) -> Callable[..., np.ndarray]:
    def generate(
        readout_key: str,
        element_index: int,
        readout_index: int,
        repetitions: int,
    ) -> np.ndarray:
        return rng.choice(num_states, size=repetitions)

    return generate


def population_data_sampler(
    populations: np.ndarray,
    rng: Generator = default_rng(),
) -> Callable[..., np.ndarray]:
    def generate(
        readout_key: str,
        element_index: int,
        readout_index: int,
        repetitions: int,
    ) -> np.ndarray:
        p1 = populations[element_index, readout_index]
        return rng.choice(2, size=repetitions, p=[1 - p1, p1])

    return generate


@qdefine
class FakeBackend(QuantumBackend):
    """A test backend used for testing upstream code.

    This backend is meant to act like a real backend, by accepting an executable for
    upload and returning pre-configured data to be processed.

    Attributes:
        uploaded: Stores the last uploaded executable, which is referenced when
            generating data.
        data_func: Used to generate simulated data when calling acquire. Takes in the
            readout key, element index, readout index, repetitions and returns an
            array of qudit states per repetition.
        gmms: A mapping from resonator keys to GMM data. This is used to generate IQ
            data.
        rng: A `numpy.random.Generator` for sampling iq data from the expected gaussian
            distributions.
    """

    data_func: Callable[..., np.ndarray] = field(factory=random_data_sampler)
    gmms: dict[str, GMMClassification] = field(factory=dict)
    rng: Generator = field(factory=default_rng)

    def upload(
        self,
        exe: QWiPExecutable,
        data_func: Callable[..., np.ndarray] | None = None,
        **kwargs,
    ) -> None:
        """Simulates a sequence upload.

        Args:
            exe: The executable to "upload".
            data_func: Updates function used to generate the fake data.
        """
        self.uploaded = exe
        self.data_func = data_func or self.data_func

    def acquire(
        self,
        repetitions: int = 512,
        num_readouts: int = 1,
        **kwargs,
    ) -> dict:
        """Generates fake data to simulate data acquisition.

        Args:
            cseq: The compiled sequence.
            repetitions: The number of shots in the resulting data.
            num_readouts: The number of readouts per sequence element.

        Returns:
            A mapping of measurement keys to `IQResult`.
        """
        readout_keys = [f"R{r}" for r in sorted(self.uploaded.read_registers)]
        num_elements = len(self.uploaded.num_reads)

        data = dict()
        for key in readout_keys:
            states = np.zeros((repetitions, num_elements, num_readouts))
            for el in range(num_elements):
                for ro in range(num_readouts):
                    states[:, el, ro] = self.data_func(key, el, ro, repetitions)

            gmm = self.gmms[key]
            arr = np.zeros(states.shape, dtype=np.complex64)
            for s in range(gmm.num_states):
                iq = self.rng.multivariate_normal(
                    mean=gmm.means[s],
                    cov=np.identity(2) * gmm.covariances[s],
                    size=np.count_nonzero(states == s),
                ).astype(np.float32)

                arr[states == s] = iq.view(np.complex64).flatten()

            data[key] = IQResult.from_numpy(arr, name=key)

        return data

    def update_parameters(self, qpu: "QPU", **kwargs):
        """Updates parameters from the QPU.

        The TestBackend requires the GMM means and covariances in order to generate
        IQ data. update parameters will
        """
        for proc in qpu.pipeline.processors:
            match proc:
                case GMMClassification(measurement_key=key):
                    self.gmms[key] = proc


__all__ = ["FakeBackend"]
