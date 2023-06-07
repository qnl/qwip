import re
import itertools as it
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


## SimulatorBackend utils

def upconvert(sampling_rate, pulse, f_LO):
    sample_time = 1 / sampling_rate

    N = len(pulse)
    T = N * sample_time

    ts = np.linspace(0, T, N * 100 + 1)

    # This compensates for rounding error
    ts = ts[: N * 100]
    ts = np.append(ts, [T])

    envelope = np.interp(x=ts, xp=np.linspace(0, T, N + 1), fp=np.append(pulse, [0]))

    # Upconvert
    carrier = np.exp(1j * 2 * np.pi * f_LO * ts)
    drive = 2 * np.pi * carrier * envelope
    return drive, ts

def get_active_channels(waveform_data: np.ndarray):
    N_channels = waveform_data.shape[0]
    active = np.any(waveform_data.reshape(N_channels, -1), axis=1)
    return np.where(active)[0]


@qdefine
class OperatorChannelMap:
    target: str
    operator: Qobj
    channels: tuple[int, ...] = field(factory=tuple)
    amplitude_factor: float = 40e6
    LO_frequency: float = 0

@qdefine
class TimeDependentHamiltonian:
    H: list[tuple[Qobj, np.ndarray]]
    ts: np.ndarray

    @property
    def dims(self) -> list | None:
        if not self.H:
            return None
        
        qobj, _ = self.H[0]
        return qobj.dims

    def simulate(self) -> np.ndarray:
        """"""
        N = self.dims[0]

        # Get all basis states for possibly multi-qubit states.
        psis = {
            i: qt.basis(N, list(i)) for i in it.product(*(range(Ni) for Ni in N))
        }
        N_ops = {i: psi * psi.dag() for i, psi in psis.items()}

        ground_state = tuple([0]*len(N))
        result = qt.mesolve(
            self.H,
            psis[ground_state],
            self.ts, e_ops=list(N_ops.values())
        )
        return result

@qdefine
class SimulatorBackend(QuantumBackend):
    uploaded: CompiledSequence | None = None
    num_levels: int = 4
    static_hamiltonian: dict[str, Qobj] = field(factory=dict)
    channel_map: list[OperatorChannelMap] = field(factory=list)
    H: list[TimeDependentHamiltonian] = field(factory=list)

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
                    map_LO_freq = qpu.db["hardware"]["local_oscillators"]["qubit"][
                        "frequency"
                    ]

                    ch_map = OperatorChannelMap(
                        target=qubit,
                        operator=map_op,
                        channels=map_channels,
                        LO_frequency=map_LO_freq,
                    )
                    self.channel_map.append(ch_map)

                    # Static hamiltonian for each qubit
                    Q = qpu.db["subsystems"][qubit]["parameters"]
                    self.static_hamiltonian[qubit] = (
                        2 * np.pi * Q.frequency * adag * a
                        + 2 * np.pi * (Q.anharmonicity / 2) * adag * adag * a * a
                    )

                else:
                    raise ValueError(f"Q component of channel {qubit} missing")

            elif t == "Q" and qubit + "_I" not in channels:
                raise ValueError(f"I component of channel {qubit} missing")

    def upload(self, cseq: CompiledSequence, **kwargs) -> None:
        self.H = [] # Clear list of hamiltonians to simulate
        self.uploaded = cseq

        N_elements = cseq.array.shape[1]
        sampling_rate = self.uploaded.waveforms.get("seq").sample_rate

        for element in range(N_elements):
            waveforms = cseq.array[:, element, ...]
            active_channels = get_active_channels(waveforms)

            drive_hamiltonians = dict()
            for ch_info in self.channel_map:
                match ch_info.channels:
                    case I_index, Q_index if {I_index, Q_index} <= set(active_channels):
                        Omega = 2 * np.pi * ch_info.amplitude_factor

                        I, Q = (waveforms[I_index, :, 0], waveforms[Q_index, :, 0])
                        pulse = I + 1j * Q

                        # Drive hamiltonian
                        drive, ts = upconvert(sampling_rate, pulse, ch_info.LO_frequency)

                        H_t = TimeDependentHamiltonian(
                            H=[
                                [self.static_hamiltonian[ch_info.target], np.ones_like(ts)],
                                [ch_info.operator, Omega * np.real(drive)]
                            ],
                            ts=ts
                        )

                        drive_hamiltonians[ch_info.target] = H_t

                    case I_index, Q_index: ...
                    case (ch_index,):
                        raise NotImplementedError("Only paired channels are currently supported.")
                    case _:
                        raise ValueError("OperatorChannelMap should have at most two channel indices.")
            
            
            if len(drive_hamiltonians) == 0:
                raise ValueError("No active channels to simulate. Check that sequence is non-empty.")

            if len(drive_hamiltonians) > 1:
                # TODO: combine hamiltonians with tensor product
                raise NotImplementedError("Only single qubit simulations are supported.")
            
            # If only one "qubit", then don't need to combine, can just take the first one.
            self.H.append(list(drive_hamiltonians.values())[0])

    def acquire(self, cseq: CompiledSequence, **kwargs) -> list:
        if not self.H:
            raise ValueError("No sequence has been uploaded")
        
        results = []

        for H_t in self.H:
            results.append(H_t.simulate())

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
