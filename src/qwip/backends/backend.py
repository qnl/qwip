import itertools as it
import re
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
from attrs import cmp_using, field
from numpy.random import Generator, default_rng
from numpy.testing import assert_allclose
from typing_extensions import Self

try:
    from qtrl.managers import MetaManager
except ImportError:
    ...


from qwip.attrs import _numpy_equals, qdefine
from qwip.processing.processors import GMMClassification
from qwip.qpu.systems import ReadoutResonator
from qwip.sequencer.compilation import CompiledSequence

if TYPE_CHECKING:
    from qwip.qpu.qpu import QPU


class QuantumBackend(metaclass=ABCMeta):
    @abstractmethod
    def upload(self, cseq: CompiledSequence, **kwargs) -> None:
        ...

    @abstractmethod
    def acquire(self, cseq: CompiledSequence, **kwargs) -> dict:
        ...

    @abstractmethod
    def update_parameters(self, qpu: "QPU", **kwargs):
        ...


@qdefine
class QTRLBackend(QuantumBackend):
    """A hardware backend that interface with QTRL."""

    meta: "MetaManager"

    def upload(self, cseq: CompiledSequence, **kwargs) -> None:
        self.meta.write_sequence(cseq)

    def acquire(self, cseq: CompiledSequence, repetitions: int = 512, **kwargs) -> dict:
        acquisition_kwargs = dict(n_reps=repetitions, save_data=False) | kwargs

        meas = self.meta.acquire(**acquisition_kwargs)
        iqdata = {
            k: meas[k]["Heterodyne"] for k in meas.keys() if re.match(r"R(\d+)", k)
        }
        return iqdata

    def update_parameters(self, qpu: "QPU", **kwargs):
        """Updates parameters from the QPU.

        The QTRL backend requires that all readout frequencies are specified in the
        variables config because this is where the ADCManager pulls the demod
        frequencies from.
        """
        for sys in qpu.subsystems.values():
            match sys:
                case ReadoutResonator(name=n, frequency=f):
                    if not (m := re.match(r"R(\d+)", n)):
                        raise ValueError(
                            f"Resonator names must be of the form 'R\\d+' for the "
                            f"QTRL backend. Got {n}"
                        )

                    self.meta.variables[f"Q{m[1]}/res_freq"] = f


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

    This backend is meant to act like a real backend, by accepting sequences for upload
    and returning pre-configured data to be processed.

    Attributes:
        uploaded: Stores the last uploaded compiled sequence, which is referenced when
            generating data.
        data_func: Used to generate simulated data when calling acquire. Takes in the
            readout key, element index, readout index, repetitions and returns an
            array of qudit states per repetition.
        gmms: A mapping from resonator keys to GMM data. This is used to generate IQ
            data.
        rng: A `numpy.random.Generator` for sampling iq data from the expected gaussian
            distributions.
    """

    uploaded: CompiledSequence | None = None
    data_func: Callable[..., np.ndarray] = field(factory=random_data_sampler)
    gmms: dict[str, GMMClassification] = field(factory=dict)
    rng: Generator = field(factory=default_rng)

    def upload(
        self,
        cseq: CompiledSequence,
        data_func: Callable[..., np.ndarray] | None = None,
        **kwargs,
    ) -> None:
        """Simulates a sequence upload.

        Args:
            cseq: The sequence to "upload".
            data_func: Updates function used to generate the fake data.
        """
        self.uploaded = cseq
        self.data_func = data_func or self.data_func

    def acquire(
        self,
        cseq: CompiledSequence,
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
            A mapping of measurement keys to a numpy array of simulated data. The data
            has shape `(IQ, shots, elements, readout)` to match the legacy QTRL format.
        """
        readout_keys = [f"R{r}" for r in cseq._readout._readout.qubits]
        num_elements = cseq.shape[1]

        data = dict()
        for key in readout_keys:
            states = np.zeros((repetitions, num_elements, num_readouts))
            for el in range(num_elements):
                for ro in range(num_readouts):
                    states[:, el, ro] = self.data_func(key, el, ro, repetitions)

            data[key] = np.zeros((2, repetitions, num_elements, num_readouts))
            gmm = self.gmms[key]

            for s in range(gmm.num_states):
                iq = self.rng.multivariate_normal(
                    mean=gmm.means[s],
                    cov=np.identity(gmm.num_states) * gmm.covariances[s],
                    size=np.count_nonzero(states == s),
                ).T

                data[key][:, states == s] = iq

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


__all__ = ["QTRLBackend", "FakeBackend"]
