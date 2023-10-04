import itertools as it
from collections.abc import Callable
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
import qutip as qt
from attrs import cmp_using, field
from numpy.random import Generator, default_rng
from qutip import Qobj
from typing_extensions import Self

from qwip.attrs import _numpy_equals, qdefine
from qwip.backends.backend import QuantumBackend, random_data_sampler
from qwip.backends.qtrl import QTRLExecutable
from qwip.processing.processors import IQTraceResult
from qwip.qpu.systems import ReadoutResonator

if TYPE_CHECKING:
    from qwip.qpu.qpu import QPU


def upconvert(
    sampling_rate: float,
    pulse: np.ndarray,
    f_LO: float,
    interpolation_factor: int = 100,
) -> np.ndarray:
    """Upconverts a pulse at a given sampling rate.

    Since pulses are typically sampled at a much lower frequency than the final target
    frequency, the waveform envelope is linearly interpolated at a faster sampling rate.

    Args:
        sampling_rate: The sampling rate in samples/second of the pulse.
        pulse: The sampled complex amplitudes at each time point.
        f_LO: The carrier frequency to multiply against the pulse.
        interpolation_factor: The number of samples to interpolate between each sample
            of the pulse envelope. The final upconverted pulse will have length
            `N * interpolation_factor + 1` where `N` is the number of samples in the
            pulse.

    Returns:
        A tuple of numpy arrays corresponding to the complex valued amplitudes and the
        timepoints.
    """
    sample_time = 1 / sampling_rate

    N = len(pulse)
    T = N * sample_time

    ts = np.linspace(0, T, N * interpolation_factor + 1)

    envelope = np.interp(x=ts, xp=np.linspace(0, T, N + 1), fp=np.append(pulse, [0]))

    # Upconvert
    carrier = np.exp(1j * 2 * np.pi * f_LO * ts)
    drive = 2 * np.pi * carrier * envelope

    return drive, ts


def get_active_channels(waveform_data: np.ndarray) -> list[int]:
    """Return indices of active channels for a single sequence element.

    This function assumes that channels are indexed along the first dimension of the
    waveform data array.

    Args:
        waveform_data: Specific element of a compiled sequence array
            cseq.array[:, element, ...]

    Returns:
        List of integers indicating indices of active channels.
    """
    N_channels = waveform_data.shape[0]
    active = np.any(waveform_data.reshape(N_channels, -1), axis=1)
    return np.where(active)[0]


@qdefine
class OperatorChannelMap:
    """Maps a channel (or pair of IQ channels) to its simulation parameters.

    Attributes:
        target: Name of the subsytem corresponding to the drive operator (i.e. "Q0")
        operator: Operator corresponding to dynamic hamiltonian
        channels: A tuple of channel indices that map to the same drive operator.
        amplitude_factor: Drive amplitude
        LO_frequency: If non-zero, signals are upconverted (multiplied) by this
            frequency before simulation.

    """

    target: str
    operator: Qobj
    channels: tuple[int, ...] = field(factory=tuple)
    amplitude_factor: float = 40e6
    LO_frequency: float = 0


def _compare_H_list(
    H1: list[tuple[Qobj, np.ndarray]], H2: list[tuple[Qobj, np.ndarray]]
) -> bool:
    """Compares two hamiltonian lists.

    Returns `True` if `H1` and `H2` are equivalent.
    """
    if len(H1) != len(H2):
        return False

    for (o1, c1), (o2, c2) in zip(H1, H2):
        if o1 != o2:
            return False

        if not _numpy_equals(c1, c2):
            return False

    return True


@qdefine
class TimeDependentHamiltonian:
    """A time-dependent Hamiltonian.

    (including static and dynamic) and time corresponding to sequence element

    Attributes:
        H: A list of tuples consisting of the `QObj` operator and its amplitude at each
            point in time. Every operator in the list is expected to have the same
            dimension and same number of time-dependent coefficients. This is passed
            directly to `qutip.mesolve`.
        ts: An array of times. Should match the shapes of all time-dependendent
            coefficients in `H`.
        targets: A tuple of strings corresponding to which qubits the Hamiltonian acts
            upon based on ordering.
    """

    H: list[tuple[Qobj, np.ndarray]] = field(
        eq=cmp_using(_compare_H_list), factory=list
    )
    ts: np.ndarray | None = field(eq=cmp_using(eq=_numpy_equals), default=None)
    targets: tuple[str, ...] = field(factory=tuple)

    @property
    def dims(self) -> list | None:
        if not self.H:
            return None

        qobj, _ = self.H[0]
        return qobj.dims

    @property
    def shape(self) -> tuple | None:
        """Returns the shape of the Hamiltonians in H.

        It is assumed that all hamiltonians in the list have the same dimension, since
        this is required by `qutip.mesolve`.

        Returns:
            `None` if no hamiltonians are preesent.
        """
        if not self.H:
            return None

        qobj, _ = self.H[0]
        return qobj.shape

    def get_basis(self) -> dict:
        """Returns a dictionary mapping fock state labels to basis state vectors."""

        if self.dims is None:
            raise ValueError("No hamiltonian to retrieve basis of.")

        N = self.dims[0]

        psis = {i: qt.basis(N, list(i)) for i in it.product(*(range(Ni) for Ni in N))}
        return psis

    def simulate(self) -> np.ndarray:
        """Run mesolve on the object's hamiltonian.

        Returns:
            result: Qobj from mesolve
        """
        if not self.H:
            raise ValueError(
                "No Hamiltonian to simulate for this element. \
                             Empty Hamiltonian here to preserve indexing."
            )
        N = self.dims[0]

        # Get all basis states for possibly multi-qubit states.
        psis = self.get_basis()
        N_ops = {i: psi * psi.dag() for i, psi in psis.items()}

        ground_state = tuple([0] * len(N))
        result = qt.mesolve(
            self.H, psis[ground_state], self.ts, e_ops=list(N_ops.values())
        )
        return result

    @classmethod
    def tensor(cls, H1: Self, H2: Self) -> Self:
        """Return new instance of TimeDependentHamiltonian with all input hamiltonians
        expanded into the full multi-qubit Hilbert space.

        The resulting Hamiltonian will look like

        $$H_\\mathrm{joint} = \\sum_n H_n \\otimes I_M + \\sum_m I_N \\otimes H_m$$

        where $N$ and $M$ are the dimensions of $H_1$ and $H_2$.

        Args:
            H1: A `TimeDependentHamiltonian`
            H2: A `TimeDependentHamiltonian`

        Returns:
            New instance of `TimeIndepedentHamiltonian`
        """
        if H1.dims is None:
            return cls(H=H2.H, ts=H2.ts)

        if H2.dims is None:
            return cls(H=H1.H, ts=H1.ts)

        if not np.all((H1.ts == H2.ts)):
            raise ValueError("Times not equal (pulse length not equal)")

        H_list = []

        for H, coeffs in H1.H:
            eye_H2 = qt.Qobj(np.eye(H2.shape[0]), dims=H2.dims)
            H1_expanded = qt.tensor(H, eye_H2)
            H_list.append((H1_expanded, coeffs))

        for H, coeffs in H2.H:
            eye_H1 = qt.Qobj(np.eye(H1.shape[0]), dims=H1.dims)
            H2_expanded = qt.tensor(eye_H1, H)
            H_list.append((H2_expanded, coeffs))

        new_targets = H1.targets + H2.targets

        return cls(H=H_list, ts=H1.ts, targets=new_targets)


def random_state_sampler(
    populations: np.ndarray,
    rng: Generator = default_rng(),
) -> Callable[..., np.ndarray]:
    def generate(
        readout_key: str,
        element_index: int,
        readout_index: int,
        repetitions: int,
        num_states: int,
    ) -> np.ndarray:
        p = p[:-1] + [1 - np.sum(p[:-1])]
        return rng.choice(num_states, size=repetitions, p=p)

    return generate


@qdefine
class QutipBackend(QuantumBackend):
    """A simulator backend used to simulate pulse sequences.

    This backend is meant to act like a real backend, accepting sequences for upload
    and returning the acquired simulated results through qutip's mesolve.

    Attributes:
        uploaded: Stores the last uploaded compiled sequence, which is referenced when
            generating data.
        num_levels: Number of energy levels to be simulated.
        static_hamiltonian: A mapping from every single qubit to its corresponding
            static hamiltonian.
        channel_map: A list of mappings from an IQ channel to its parameters.
        H: A list of TimeDependentHamiltonian's for each element with active channels.
        rng: A `numpy.random.Generator` for sampling iq data from the expected gaussian
            distributions.
        readouts: A mapping from readout resonator to its correspondign ReadoutResonator
            object which stores its parameters and can solve for its field equation.
    """

    uploaded: QTRLExecutable | None = None
    num_levels: int = 4
    static_hamiltonian: dict[str, Qobj] = field(factory=dict)
    channel_map: list[OperatorChannelMap] = field(factory=list)
    H: list[TimeDependentHamiltonian] = field(factory=list)
    rng: Generator = field(factory=default_rng)
    readouts: dict[str, ReadoutResonator] = field(factory=dict)

    def update_parameters(self, qpu: "QPU", **kwargs):
        """Updates parameters from the QPU -- creating mappings from channels to
        parameters and qubits to static hamiltonians.

        Args:
            qpu: Loaded config file
        """

        channels = qpu.db["compilation"]["channels"]

        for ch in channels.keys():
            if len(ch.split("_")) != 2:
                ...
            else:
                qubit, t = ch.split("_")

                if t == "I" and qubit[0] == "Q":
                    if qubit + "_Q" in channels:
                        # Mapping of channels to its parameters
                        a, adag = qt.destroy(self.num_levels), qt.create(
                            self.num_levels
                        )
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

        self.readouts = {}
        for name, sys in qpu.subsystems.items():
            match sys:
                case ReadoutResonator():
                    self.readouts[name] = sys

    def upload(self, exe: QTRLExecutable, **kwargs: Any) -> None:
        """Constructs the drive hamiltonians for elements of active channels
        to later be simulated.

        Stores a list of TimeIndependentHamiltonian's in self.H which correspond
        to each element with active channels. Both the drive and static hamiltonian
        of all active channels are in the attribute H of TimeIndependentHamiltonian.
        If there is more than one active channel, dimensions of individual hamiltonians
        are properly adjusted by tensor producting with the identity.

        Args:
            exe: A compiled sequence
        """
        self.H = []  # Clear list of hamiltonians to simulate
        self.uploaded = exe
        cseq = exe

        N_elements = cseq.array.shape[1]
        sampling_rate = self.uploaded.waveforms.get("seq").sample_rate

        for element in range(N_elements):
            waveforms = cseq.array[:, element, ...]
            active_channels = get_active_channels(waveforms)

            drive_hamiltonians = dict()
            for ch_info in self.channel_map:
                match ch_info.channels:
                    # Map active channels to its parameters
                    case I_index, Q_index if {I_index, Q_index} <= set(active_channels):
                        Omega = 2 * np.pi * ch_info.amplitude_factor

                        I, Q = (waveforms[I_index, :, 0], waveforms[Q_index, :, 0])
                        pulse = I + 1j * Q

                        # Drive hamiltonian
                        drive, ts = upconvert(
                            sampling_rate, pulse, ch_info.LO_frequency
                        )

                        # Encode static and dynamic hamiltonian info
                        H_t = TimeDependentHamiltonian(
                            H=[
                                (
                                    self.static_hamiltonian[ch_info.target],
                                    np.ones_like(ts),
                                ),
                                (ch_info.operator, Omega * np.real(drive)),
                            ],
                            ts=ts,
                            targets=(ch_info.target,),
                        )

                        drive_hamiltonians[ch_info.target] = H_t

                    case I_index, Q_index:
                        ...
                    case (ch_index,):
                        raise NotImplementedError(
                            "Only paired channels are currently supported."
                        )
                    case _:
                        raise ValueError(
                            "OperatorChannelMap should have at most two channel indices."
                        )

            if len(drive_hamiltonians) > 0:
                H_t = reduce(
                    TimeDependentHamiltonian.tensor, drive_hamiltonians.values()
                )
                self.H.append(H_t)

            else:
                self.H.append(TimeDependentHamiltonian())

        if self.H == [TimeDependentHamiltonian()] * N_elements:
            raise ValueError(
                "No active channels to simulate. Check that sequence is non-empty."
            )

    def acquire(
        self,
        exe: QTRLExecutable,
        repetitions: int = 512,
        elements: list[int] = [-1],
        drive: float = 1e6,
        **kwargs,
    ) -> list:
        """Simulate the Hamiltonians using mesolve.

        Args:
            exe: Compiled sequence.
            repetitions: Number of shots
            elements: A list of indices indicating which elements to simulate.
            drive: Readout driving amplitude.

        Returns:
            results: A dictionary mapping all qubits to their corresponding IQ outputs
                in the form of a 3D array with shape (elements x shots x time steps).
                For not targeted qubits, they default to the ground trajectory. And for
                states with no corresponding chi value, they default to zero values.
        """
        cseq = exe
        to_simulate = np.arange(cseq.shape[1])[elements]

        # Hard code drive envelope and times for now.
        def drive_envelope(t):
            return 2 * np.pi * drive

        ts = np.linspace(0, 1e-5, 1000)

        # Dictionary mapping from qubit to results, cavity fields
        all_targets = list(self.static_hamiltonian.keys())
        results = {
            target: np.zeros((repetitions, len(elements), 1, len(ts)), dtype=complex)
            for target in all_targets
        }
        readout_fields = {
            target: R.solve_cavity_field_equation(ts, drive_envelope)
            for target, R in self.readouts.items()
        }

        for i, el in enumerate(to_simulate):
            H_obj = self.H[el]
            H_basis = H_obj.get_basis()

            # Expectation values for all possible states at last time step
            qt_result = H_obj.simulate()
            qt_expect = [r[-1] for r in qt_result.expect]

            # List of N tuples with length number of targets, states chosen
            # using rng.choice with non-uniform distribution matching last time step
            shots_sample = default_rng().choice(
                np.array(list(H_basis.keys())), repetitions, p=qt_expect
            )

            for shot, state in enumerate(shots_sample):
                H_targets = [H_obj.targets[q].split("Q")[1] for q in range(len(state))]

                for q in range(len(all_targets)):
                    if str(q) in H_targets:
                        level = state[H_targets.index(str(q))]
                    else:
                        level = 0  # untargeted qubits are by default at ground state

                    # pull out trajectories matching this multi-qubit state
                    if level < len(readout_fields[f"R{q}"]):
                        noise = gaussian_noise(self.readouts[f"R{q}"].eta, len(ts))
                        trajectory = (readout_fields[f"R{q}"][level]) + noise
                        results[f"Q{q}"][shot, i, 0, :] = trajectory

        return {
            k: IQTraceResult.from_numpy(trace, name=k) for k, trace in results.items()
        }


def gaussian_noise(eta, num_samples):
    noise_scaling = 1 / (2**0.5) / eta
    return noise_scaling * (
        np.random.normal(size=(num_samples, 2)).view(np.complex128).squeeze()
    )


def get_multi_qubit_populations(results, labels, N_sys):
    labeled = dict(zip(labels, results.expect))

    qdicts = [dict() for _ in range(N_sys)]
    for q in range(N_sys):
        for state, populations in labeled.items():
            if state[q] not in qdicts[q]:
                qdicts[q][state[q]] = np.array(populations)
            else:
                qdicts[q][state[q]] += populations
    return qdicts
