import re
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from attrs import field
from numpy.random import Generator, default_rng

try:
    from qtrl.managers import MetaManager
except ImportError:
    ...

import qutip as qt
from qutip import Qobj

from qwip.attrs import qdefine
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
class OperatorChannelMap:
    operator: Qobj
    channels: tuple[int, int] = field(factory=tuple)  # int, int or int
    amplitude_factor: float = 40e6
    LO_frequency: float = 0
    name: str = ""


@qdefine
class SimulatorBackend(QuantumBackend):
    uploaded: CompiledSequence | None = None
    num_levels: int = 4
    static_hamiltonian: dict[str, Qobj] = field(factory=dict)
    drive_hamiltonian: dict[str, tuple[Qobj, np.ndarray]] = field(factory=dict)
    channel_map: list[OperatorChannelMap] = field(factory=list)
    H: dict[str, list[tuple[Qobj, np.ndarray]]] = field(factory=dict)
    ts: dict[str, np.ndarray] = field(factory=dict)

    def update_parameters(self, qpu: "QPU", **kwargs):
        channels = qpu.db["compilation"]["channels"]

        for ch in channels.keys():
            qubit, t = ch.split("_")

            if t == "I" and qubit[0] == "Q":
                if qubit + "_Q" in channels:
                    # Mapping of channels to its parameters
                    a, adag = qt.destroy(self.num_levels), qt.create(self.num_levels)
                    map_op = a + adag

                    map_channels = (
                        channels[f"{qubit}_I"]["index"],
                        channels[f"{qubit}_Q"]["index"],
                    )
                    map_LOfreq = qpu.db["hardware"]["local_oscillators"]["qubit"][
                        "frequency"
                    ]

                    ch_map = OperatorChannelMap(
                        operator=map_op,
                        channels=map_channels,
                        LO_frequency=map_LOfreq,
                        name=qubit,
                    )
                    self.channel_map.append(ch_map)

                    # Static hamiltonian for each qubit
                    Q = qpu.db["subsystems"][qubit]["parameters"]
                    self.static_hamiltonian[qubit] = (
                        2 * np.pi * Q.frequency * adag * a
                        + 2 * np.pi * (Q.anharmonicity / 2) * adag * adag * a * a
                    )

                else:
                    raise NotImplementedError(f"Q component of channel {qubit} missing")

            elif t == "Q" and qubit + "_I" not in channels:
                raise NotImplementedError(f"I component of channel {qubit} missing")

    def upload(self, cseq: CompiledSequence, **kwargs) -> None:
        self.drive_hamiltonian = {}
        self.H = {}
        self.ts = {}

        self.uploaded = cseq
        chs_on = on_channels(cseq.array)

        for map in self.channel_map:
            I_index, Q_index = map.channels

            # Create drive operators for correctly paired on channels
            if I_index in chs_on and Q_index in chs_on:
                Omega = 2 * np.pi * map.amplitude_factor
                sampling_rate = self.uploaded.waveforms.get("seq").sample_rate

                ch_I, ch_Q = (
                    self.uploaded.array[I_index, -1, :, 0],
                    self.uploaded.array[Q_index, -1, :, 0],
                )
                pulse = ch_I + 1j * ch_Q

                # Drive hamiltonian
                drive, ts = upconvert(sampling_rate, pulse, map.LO_frequency)
                self.drive_hamiltonian[map.name] = (
                    map.operator,
                    Omega * np.real(drive),
                )

                self.ts[map.name] = ts
                self.H[map.name] = [
                    [self.static_hamiltonian[map.name], np.ones_like(ts)],
                    [
                        self.drive_hamiltonian[map.name][0],
                        self.drive_hamiltonian[map.name][1],
                    ],
                ]

    def acquire(self, cseq: CompiledSequence, **kwargs) -> dict:
        results = {}
        if not self.H:
            raise NotImplementedError("No sequence has been uploaded")
        else:
            for qubit in self.H.keys():
                results[qubit] = simulate_H(
                    self.H[qubit], self.ts[qubit], self.num_levels
                )

        return results


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


__all__ = ["QTRLBackend", "SimulatorBackend", "FakeBackend"]


# Code from pypulse
def upconvert(sampling_rate, pulse, f_LO):
    sample_time = 1 / sampling_rate

    N = len(pulse)
    T = N * sample_time
    ts = np.arange(0, T, sample_time / 100)

    # This compensates for rounding error
    ts = ts[: N * 100]
    ts = np.append(ts, [T])

    envelope = np.interp(x=ts, xp=np.linspace(0, T, N + 1), fp=np.append(pulse, [0]))

    # Upconvert
    carrier = np.exp(1j * 2 * np.pi * f_LO * ts)
    drive = 2 * np.pi * carrier * envelope
    return drive, ts


def simulate_H(H, ts, N):
    psis = [qt.basis(N, i) for i in range(N)]
    N_ops = [psi * psi.dag() for psi in psis]
    result = qt.mesolve(H, psis[0], ts, e_ops=N_ops)
    return result


def on_channels(uploaded):
    chs_on = np.where(np.any(uploaded, axis=(1, 2, 3)))[0]
    return chs_on
