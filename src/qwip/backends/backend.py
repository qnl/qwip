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
from qwip.processing.processors import GMMClassification, IQResult
from qwip.qpu.systems import ReadoutResonator
from qwip.sequencer.compilation import CompiledSequence, QuantumExecutable
from qwip.sequencer.elements import SequenceElement

if TYPE_CHECKING:
    from qwip.qpu.qpu import QPU


class QuantumBackend(metaclass=ABCMeta):
    @abstractmethod
    def upload(self, exe: QuantumExecutable, **kwargs) -> None:
        ...

    @abstractmethod
    def acquire(self, exe: QuantumExecutable, **kwargs) -> dict:
        ...

    @abstractmethod
    def update_parameters(self, qpu: "QPU", **kwargs):
        ...

    @property
    def exe_formats(self) -> set[type[QuantumExecutable]]:
        return set()


def populate_unpaired(cseq: CompiledSequence, fill: float = 1 / 2**15) -> None:
    """Ensures that waveform data is paired.

    Currently there is still a bug when uploading waves to a ZI HDAWG where waves are
    uploaded incorrectly unless they are paired. This function modifies a compiled
    qtrl sequence so that all channels with waveform data are paired.

    Args:
        cseq: The compiled sequence to modify.
        fill: The DAC amplitude to set on an empty unpaired channel. This should be
            small to avoid any adverse affect on the system. This value is added to the
            first sample only.
    """
    has_wave = np.any(cseq.array, axis=2)[..., 0]  # ignore marker array

    for ch1, ch2 in np.arange(cseq.shape[0]).reshape(-1, 2):
        # bitwise xor to find all unpaired channels
        unpaired = has_wave[ch1] ^ has_wave[ch2]

        cseq.array[ch1, unpaired & ~has_wave[ch1], 0, 0] = fill
        cseq.array[ch2, unpaired & ~has_wave[ch2], 0, 0] = fill


def format_legacy_IQ(arr: np.ndarray) -> np.ndarray:
    """Reformats a QTRL result array.

    This function will reorder the axis so that the IQ data for each shot is
    contiguous. The legacy heterodyne array is a 4-D array where the axes correspond
    to `(IQ, shots, elements, readouts)`. This is reformatted to a complex numpy array
    where the shape is `(elements, readouts, shots)`.
    """
    match arr.dtype:
        case np.float32:
            cast = np.complex64
        case np.float64:
            cast = np.complex128
        case _:
            arr = arr.astype(float)
            cast = np.complex128

    return np.array(np.transpose(arr, [2, 3, 1, 0]), order="C").view(cast)[..., 0]


@qdefine
class QTRLBackend(QuantumBackend):
    """A hardware backend that interface with QTRL."""

    meta: "MetaManager"
    ro_se: SequenceElement = field(factory=SequenceElement)

    def upload(self, exe: CompiledSequence, **kwargs) -> None:
        populate_unpaired(exe)
        self.meta.write_sequence(exe)

    def acquire(self, exe: CompiledSequence, repetitions: int = 512, **kwargs) -> dict:
        acquisition_kwargs = dict(n_reps=repetitions, save_data=False) | kwargs

        meas = self.meta.acquire(**acquisition_kwargs)
        iqdata = {}

        for k in meas:
            if not re.match(r"R(\d+)", k):
                continue

            IQ = format_legacy_IQ(meas[k]["Heterodyne"])

            iqdata[k] = IQResult.from_numpy(IQ, name=k)

        return iqdata

    def update_parameters(
        self, qpu: "QPU", readout: dict | SequenceElement = {}, **kwargs
    ):
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

        match readout:
            case dict():
                ro_se = self.get_readout_sequence(qpu, **readout)
            case SequenceElement():
                ro_se = readout
            case _:
                raise ValueError(
                    f"Readout must be a sequence element or a dictionary of parameters. "
                    f"Got {readout}"
                )

        self.ro_se = ro_se

    def get_readout_sequence(
        self,
        qpu: "QPU",
        readout: str = "default",
        length: float | None = None,
        length_variable: str = "width",
    ) -> SequenceElement:
        """Constructs a readout sequence element from the readout config.

        Args:
            readout: The name of the readout config.
            length: The readout length in seconds.
            length_variable: The pulse variable that corresponds to the pulse width in
                the readout pulse.

        Returns:
            The readout sequence element.
        """
        readout_config = qpu.config.readout[readout]

        ro_se = SequenceElement()
        length = length or readout_config.length

        for r in qpu.sequencer.readout_qubits:
            pulse_name = readout_config.drives[f"R{r}"]

            ro_se += qpu.db.load_pulse(pulse_name, {length_variable: length})

        return ro_se

    def exe_formats(self) -> set[type[QuantumExecutable]]:
        return {CompiledSequence}


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
            A mapping of measurement keys to `IQResult`.
        """
        readout_keys = [f"R{r}" for r in cseq._readout._readout.qubits]
        num_elements = cseq.shape[1]

        data = dict()
        for key in readout_keys:
            states = np.zeros((num_elements, num_readouts, repetitions))
            for el in range(num_elements):
                for ro in range(num_readouts):
                    states[el, ro, :] = self.data_func(key, el, ro, repetitions)

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


__all__ = ["QTRLBackend", "FakeBackend"]
