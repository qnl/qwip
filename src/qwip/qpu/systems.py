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
    chi: float | None = None
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


def make_quantum_system_unstructure_fn(cls):
    unstructure_attrs = make_attrs_unstructure_fn(cls)

    def unstructure_fn(obj):
        return {**unstructure_attrs(obj), "__class__": type(obj).__name__}

    return unstructure_fn


qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, QuantumSystem), make_quantum_system_unstructure_fn
)


__all__ = ["Transmon", "ReadoutResonator"]
