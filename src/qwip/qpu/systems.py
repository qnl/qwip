from collections.abc import Callable

import numpy as np
from attrs import field
from loguru import logger
from scipy.integrate import solve_ivp

import qwip
from qwip._cattr import make_attrs_structure_fn, make_attrs_unstructure_fn
from qwip.attrs import qdefine
from qwip.sequencer.phase_tracker import Frame

REGISTERED_QSYSTEMS: dict[str, "QuantumSystem"] = dict()


def register_qsystem(cls) -> type:
    if not issubclass(cls, QuantumSystem):
        raise TypeError(f"Registered quantum model must subclass {QuantumSystem}")

    REGISTERED_QSYSTEMS[cls.__name__] = cls

    return cls


@qdefine
class QuantumSystem:
    name: str

    def get_frames(self, **kwargs) -> dict[str, Frame]:
        return dict()


def _rename_frame_key(key: str) -> str:
    if "{mod_key}" in key:
        logger.warning(
            "'mod_key' has been deprecated in favor of 'subspace'. Please update all "
            "frame keys accordingly."
        )

        return key.replace("mod_key", "subspace")

    return key


@register_qsystem
@qdefine
class Transmon(QuantumSystem):
    frequency: float
    anharmonicity: float | None = None
    frame_key: str = field(
        converter=_rename_frame_key, default="{name}.freq_{subspace}"
    )

    @property
    def frequency_EF(self) -> float:
        alpha = self.anharmonicity
        return None if alpha is None else self.frequency + alpha

    @property
    def subspaces(self) -> tuple[str, ...]:
        return ("01", "12")

    def get_frames(self, **kwargs) -> dict[str, Frame]:
        """Get the frame mapping associated with this system.

        Args:
            LO_map: A mapping of local oscillator names to their current frequencies.

        Returns:
            A mapping of named frames to frequencies.
        """
        frames = dict()
        name = self.frame_key.format(name=self.name, subspace="01")
        frames[name] = self.frequency

        if self.frequency_EF is not None:
            name = self.frame_key.format(name=self.name, subspace="12")
            frames[name] = self.frequency_EF

        return frames

    def mod_frequency(self, subspace: str = "01") -> Frame:
        """Returns the modulation frequency for a specific subspace frame."""
        name = self.frame_key.format(name=self.name, subspace=subspace)
        return self.get_frames()[name]

    @staticmethod
    def compute_transmon_parameters(omega, alpha, tolerance=1e-5):
        """Computes Ej and Ec given the frequency and anharmonicity of the qubit.

        This function uses the fourth order taylor expansion of $\\omega$ and 
        $\\alpha$ to solve $E_C$ and $E_J$. The taylor expansions are in terms
        of the small parameter $\\eta = \\sqrt{\\frac{2E_C}{E_J}}$, as given in
        [1]. 
        
        The units of omega and alpha, in principle, do not matter as long as
        they are consistent. However, working in units of GHz tends to give
        better results.

        Args:
            omega (float): The frequency (01) of the transmon. See above on units.
            alpha (float): The anharmonicity of the transmon, defined as f12 - f01. 
                This should be a negative number. See above on units.

        Returns:
            tuple: A tuple (Ec, Ej) with the same units as omega and alpha.

        References:
            [1]: See https://arxiv.org/abs/1706.06566.
        """
        p_a = np.poly1d([46899/(2**15), 4635/(2**12), 81/(2**7), 9/(2**4), 1])
        p_w = np.poly1d([-5319/(2**15), -19/(2**7), -21/(2**7), -1/4, -1, 4])

        eta = np.poly1d([1, 0])

        p_t = eta*omega*p_a + alpha*p_w

        roots = p_t.r[-1]

        eta_0 = np.real(roots[np.isreal(roots)][0])

        if not np.isclose(p_t(eta_0), 0, atol=tolerance):
            logger.warning("Error: Unable to find a solution.")

        Ec = -alpha/p_a(eta_0)

        Ej = 2*Ec/(eta_0**2)

        return Ec, Ej


@register_qsystem
@qdefine
class ReadoutResonator(QuantumSystem):
    frequency: float
    kappa: float | None = None
    chi: tuple[float, ...] | None = None
    eta: float | None = None
    frame_key: str = "{name}.freq"

    def get_frames(self, **kwargs) -> dict[str, Frame]:
        """Get the frame dictionary associated with this system.

        Args:
            LO_map: A mapping of local oscillator names to their current frequencies.

        Returns:
            A mapping of named frames to frequencies.
        """

        frames = dict()
        name = self.frame_key.format(name=self.name)
        frames[name] = self.frequency
        return frames

    def mod_frequency(self) -> Frame:
        """Returns the modulation frequency for a specific subspace frame"""

        name = self.frame_key.format(name=self.name)
        return self.get_frames()[name]

    def get_cavity_field_equation(
        self,
        drive_envelope: Callable[[float], complex],
        drive_frequency: float | None = None,
    ) -> Callable[[float, complex], complex]:
        """Returns the semi-classical cavity field equation for a given resonator.

        Args:
            drive_envelope: A function that takes in single time value and returns the
                amplitude of the driving field envelope at that point in time.
            drive_frequency: The modulation frequency of the resonator driving field.
                If `None`, it is assumed to be equal to the resonator frequency.

        Returns:
            A function that takes in time value(s) and the complex cavity field amplitude
            at that point in time and returns the evaluated derivative of the cavity
            field alpha for all states, dependent on the number chi values provided.
        """

        drive_frequency = drive_frequency or self.frequency
        detuning = self.frequency - drive_frequency

        def alpha_derivative(t, alpha_t):
            return -2 * np.pi * self.kappa * alpha_t / 2 - 1j * (
                2 * np.pi * drive_envelope(t)
                + 2 * np.pi * (detuning + np.array(self.chi).reshape(-1, 1)) * alpha_t
            )

        return alpha_derivative

    def solve_cavity_field_equation(
        self,
        ts: np.ndarray,
        drive_envelope: Callable[[float], complex],
        drive_frequency: float | None = None,
        alpha_0: np.ndarray[complex] | None = None,
    ) -> np.ndarray:
        """Solves the semi-classical cavity field equation for all states of a qubit.

        Args:
            ts: A list of time points at which to evaluate the field amplitude.
            drive_envelope: A function that takes in single time value and returns the
                amplitude of the driving field envelope at that point in time.
            drive_frequency: The modulation frequency of the resonator driving field.
                If `None`, it is assumed to be equal to the resonator frequency.
            alpha_0: The initial cavity field amplitudes for all states in an array.
                Must be the same length as tuple of chi values.

        Returns:
            The cavity field amplitude at each of the given time points for each state.
            Has a shape of (S, N) where S = number of states, N = len(ts)
        """
        if len(ts) == 0:
            raise ValueError("No time points given.")

        # Default start value is 0.0 for all possible levels
        if not alpha_0:
            alpha_0 = np.zeros_like(self.chi).astype(complex)

        if len(self.chi) != len(alpha_0):
            raise ValueError(
                "Number of initial chi values does not match number of states to be "
                "simulated."
            )

        time_interval = [ts[0], ts[-1]]
        field_equation = self.get_cavity_field_equation(
            drive_envelope,
            drive_frequency,
        )

        alphas = solve_ivp(
            field_equation, time_interval, alpha_0, t_eval=ts, vectorized=True
        ).y
        return alphas


def make_quantum_system_structure_fn(cls):
    structure_attrs = make_attrs_structure_fn(cls)

    def structure_fn(obj, cls):
        if isinstance(obj, cls):
            return obj

        if "modulation_name" in obj:
            logger.warning(
                "The attribute `modulation_name` has been renamed to `frame_key` and is"
                " now deprecated. Update all unstructured systems accordingly."
            )

            obj["frame_key"] = obj["modulation_name"]
            del obj["modulation_name"]

        return structure_attrs(obj, cls)

    return structure_fn


def make_quantum_system_unstructure_fn(cls):
    unstructure_attrs = make_attrs_unstructure_fn(cls)

    def unstructure_fn(obj):
        return {**unstructure_attrs(obj), "__class__": type(obj).__name__}

    return unstructure_fn


qwip.converter.register_structure_hook_factory(
    lambda cls: issubclass(cls, QuantumSystem), make_quantum_system_structure_fn
)

qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, QuantumSystem), make_quantum_system_unstructure_fn
)


__all__ = ["Transmon", "ReadoutResonator"]
