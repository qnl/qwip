from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

import qwip
from qwip._cattr import make_attrs_unstructure_fn
from qwip.attrs import qdefine
from qwip.sequencer.phase_tracker import ModulationFrequency

REGISTERED_QSYSTEMS: dict[str, "QuantumSystem"] = dict()


def register_qsystem(cls) -> type:
    if not issubclass(cls, QuantumSystem):
        raise TypeError(f"Registered quantum model must subclass {QuantumSystem}")

    REGISTERED_QSYSTEMS[cls.__name__] = cls

    return cls


@qdefine
class QuantumSystem:
    name: str

    def get_modulations(self, **kwargs) -> dict[str, ModulationFrequency]:
        return dict()


@register_qsystem
@qdefine
class Transmon(QuantumSystem):
    frequency: float
    anharmonicity: float | None = None
    local_oscillator: str | None = None
    modulation_name: str = "{name}.mod_{mod_key}"

    @property
    def frequency_EF(self) -> float:
        alpha = self.anharmonicity
        return None if alpha is None else self.frequency + alpha

    @property
    def mod_keys(self) -> tuple[str, ...]:
        return ("GE", "EF")

    def get_modulations(
        self, LO_map: dict = {}, **kwargs
    ) -> dict[str, ModulationFrequency]:
        """Get the modulation dictionary associated with this system.

        Args:
            LO_map: A mapping of local oscillator names to their current frequencies.

        Returns:
            A mapping of named modulation keys to frequencies.
        """
        if self.local_oscillator is None:
            lo_freq = 0
        else:
            try:
                lo_freq = LO_map[self.local_oscillator]
            except KeyError as e:
                raise KeyError(
                    f"Specified LO '{self.local_oscillator}' is not present in "
                    f"{LO_map}."
                ) from e

        modulations = dict()
        mod_name = self.modulation_name.format(name=self.name, mod_key="GE")
        modulations[mod_name] = self.frequency - lo_freq

        if self.frequency_EF is not None:
            mod_name = self.modulation_name.format(name=self.name, mod_key="EF")
            modulations[mod_name] = self.frequency_EF - lo_freq

        return modulations

    def mod_frequency(
        self, mod_key: str = "GE", LO_map: dict = {}
    ) -> ModulationFrequency:
        """Returns the modulation frequency for a specific modulation_key"""
        name = self.modulation_name.format(name=self.name, mod_key=mod_key)
        return self.get_modulations(LO_map)[name]


@register_qsystem
@qdefine
class ReadoutResonator(QuantumSystem):
    frequency: float
    kappa: float | None = None
    chi: tuple[float, ...] | None = None
    eta: float | None = None
    local_oscillator: str | None = None
    modulation_name: str = "{name}.mod"

    def get_modulations(
        self, LO_map: dict = {}, **kwargs
    ) -> dict[str, ModulationFrequency]:
        """Get the modulation dictionary associated with this system.

        Args:
            LO_map: A mapping of local oscillator names to their current frequencies.

        Returns:
            A mapping of named modulation keys to frequencies.
        """
        if self.local_oscillator is None:
            lo_freq = 0
        else:
            try:
                lo_freq = LO_map[self.local_oscillator]
            except KeyError as e:
                raise KeyError(
                    f"Specified LO '{self.local_oscillator}' is not present in "
                    f"{LO_map}."
                ) from e

        modulations = dict()
        mod_name = self.modulation_name.format(name=self.name, mod_key="GE")
        modulations[mod_name] = self.frequency - lo_freq
        return modulations

    def mod_frequency(
        self, mod_key: str = "GE", LO_map: dict = {}
    ) -> ModulationFrequency:
        """Returns the modulation frequency for a specific modulation_key"""
        name = self.modulation_name.format(name=self.name, mod_key=mod_key)
        return self.get_modulations(LO_map)[name]

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

        if len(alpha_0) != len(self.chi):
            raise ValueError(
                "Number of initial values does not match number of states to be "
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


def make_quantum_system_unstructure_fn(cls):
    unstructure_attrs = make_attrs_unstructure_fn(cls)

    def unstructure_fn(obj):
        return {**unstructure_attrs(obj), "__class__": type(obj).__name__}

    return unstructure_fn


qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, QuantumSystem), make_quantum_system_unstructure_fn
)


__all__ = ["Transmon", "ReadoutResonator"]
