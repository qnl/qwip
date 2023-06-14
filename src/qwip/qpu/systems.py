import numpy as np
from collections.abc import Callable

import qwip
from qwip._cattr import make_attrs_unstructure_fn
from qwip.attrs import qdefine
from qwip.sequencer.phase_tracker import ModulationFrequency

from scipy.integrate import solve_ivp

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
        qubit_state: int = 0,
    ) -> Callable[[float, complex], complex]:
        """Returns the semi-classical cavity field equation for a given qubit state.
        
        Args:
            drive_envelope: A function that takes in single time value and returns the 
                amplitude of the driving field envelope at that point in time.
            drive_frequency: The modulation frequency of the resonator driving field.
                If `None`, it is assumed to be equal to the resonator frequency.
            qubit_state: The corresponding qubit state. This determines the sign of the
                dispersive shift in the cavity field equation.

        Returns:
            A function that takes in a time value and the complex cavity field amplitude
            at that point in time and returns the evaluated derivative of the cavity
            field alpha.
        """
        if qubit_state >= len(self.chi):
            return None

        drive_frequency = drive_frequency or self.frequency
        detuning = self.frequency - drive_frequency

        chi = self.chi[qubit_state]
        diff_field_equation = lambda t, field: -self.kappa*field/2 - \
                                    (drive_envelope(t) + (detuning + chi)*field)*1j
        return diff_field_equation

    def solve_cavity_field_equation(
        self,
        ts: np.ndarray,
        drive_envelope: Callable[[float], complex],
        drive_frequency: float | None = None,
        qubit_state: int = 0,
        alpha_0: np.ndarray[complex] = [0+0j],
    ) -> np.ndarray:
        """Solves the semi-classical cavity field equation for a given qubit state.
        
        Args:
            ts: A list of time points at which to evaluate the field amplitude. 
            drive_envelope: A function that takes in single time value and returns the 
                amplitude of the driving field envelope at that point in time.
            drive_frequency: The modulation frequency of the resonator driving field.
                If `None`, it is assumed to be equal to the resonator frequency.
            qubit_state: The corresponding qubit state. This determines the sign of the
                dispersive shift in the cavity field equation.
            alpha_0: The initial cavity field amplitude in an array.

        Returns:
            The cavity field amplitude at each of the given time points.
        """
        if ts.size == 0:
            raise ValueError("No time points given.")
        
        time_interval = [ts[0], ts[-1]]
        field_equation = self.get_cavity_field_equation(drive_envelope, drive_frequency, qubit_state)

        # When qubit state > # of provided chi values, the field_equation is None
        if not field_equation:
            return np.full(ts.shape, np.nan)   # return NaN values
        
        field_solution = solve_ivp(field_equation, time_interval, alpha_0, t_eval=ts).y
        return field_solution


def make_quantum_system_unstructure_fn(cls):
    unstructure_attrs = make_attrs_unstructure_fn(cls)

    def unstructure_fn(obj):
        return {**unstructure_attrs(obj), "__class__": type(obj).__name__}

    return unstructure_fn


qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, QuantumSystem), make_quantum_system_unstructure_fn
)


__all__ = ["Transmon", "ReadoutResonator"]
