import itertools as it
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import qutip as qt
from attrs import cmp_using, field
from qutip import Qobj
from typing_extensions import Self

from qwip.attrs import _numpy_equals, qdefine
from qwip.backends.backend import QuantumBackend
from qwip.sequencer.compilation import CompiledSequence

if TYPE_CHECKING:
    from qwip.qpu.qpu import QPU


def upconvert(
    sampling_rate: float,
    pulse: np.ndarray,
    f_LO: float,
    interpolation_factor: int = 100,
):
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


def get_active_channels(waveform_data: np.ndarray):
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
    """

    H: list[tuple[Qobj, np.ndarray]] = field(
        eq=cmp_using(_compare_H_list), factory=list
    )
    ts: np.ndarray | None = field(eq=cmp_using(eq=_numpy_equals), default=None)

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
    def tensor(cls, H1: Self, H2: Self):
        """Return new instance of TimeDependentHamiltonian with all input hamiltonians
        expanded into the full multi-qubit Hilbert space.

        The resulting Hamiltonian will look like

        $$H_\mathrm{joint} = \sum_n H_n \otimes I_M + \sum_m I_N \otimes H_m$$

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

        return cls(H=H_list, ts=H1.ts)


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
    """

    uploaded: CompiledSequence | None = None
    num_levels: int = 4
    static_hamiltonian: dict[str, Qobj] = field(factory=dict)
    channel_map: list[OperatorChannelMap] = field(factory=list)
    H: list[TimeDependentHamiltonian] = field(factory=list)

    def update_parameters(self, qpu: "QPU", **kwargs):
        """Updates parameters from the QPU -- creating mappings from channels to
        parameters and qubits to static hamiltonians.

        Args:
            qpu: Loaded config file
        """

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
        """Constructs the drive hamiltonians for elements of active channels
        to later be simulated.

        Stores a list of TimeIndependentHamiltonian's in self.H which correspond
        to each element with active channels. Both the drive and static hamiltonian
        of all active channels are in the attribute H of TimeIndependentHamiltonian.
        If there is more than one active channel, dimensions of individual hamiltonians
        are properly adjusted by tensor producting with the identity.

        Args:
            cseq: Compiled sequence
        """
        self.H = []  # Clear list of hamiltonians to simulate
        self.uploaded = cseq

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

        if len(self.H) == 0:
            raise ValueError(
                "No active channels to simulate. Check that sequence is non-empty."
            )

        # Make sure H same length as number of elements so indexing is preserved
        if len(self.H) < N_elements:
            for i in range(N_elements - len(self.H)):
                self.H.insert(0, TimeDependentHamiltonian())

    def acquire(
        self,
        cseq: CompiledSequence,
        elements: list[int] = [-1],  # how to set default value?
        **kwargs,
    ) -> list:
        """Simulate the Hamiltonians using mesolve.

        Args:
            elements: A list of indices indicating which elements to simulate.

        Returns:
            results: A list of mesolve outputs, which are expectation values of the
                basis vectors (made from get_basis() of TimeIndependentHamiltonian).

        """
        if not self.H:
            raise ValueError("No sequence has been uploaded")

        results = []

        to_simulate = np.arange(cseq.shape[1])[elements]

        for el in to_simulate:
            results.append(self.H[el].simulate())

        return results
