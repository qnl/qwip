import qwip
from qwip._cattr import make_attrs_unstructure_fn
from qwip.settings.settings import qdefine

REGISTERED_QSYSTEMS: dict[str, "QuantumSystem"] = dict()


def register_qsystem(cls) -> type:
    if not issubclass(cls, QuantumSystem):
        raise TypeError(f"Registered quantum model must subclass {QuantumSystem}")

    REGISTERED_QSYSTEMS[cls.__name__] = cls

    return cls


@qdefine
class QuantumSystem:
    name: str


@register_qsystem
@qdefine
class Transmon(QuantumSystem):
    frequency: float
    anharmonicity: float | None = None
    local_oscillator: str | None = None

    @property
    def frequency_EF(self) -> float:
        alpha = self.anharmonicity
        return None if alpha is None else self.frequency + alpha

    @property
    def mod_keys(self) -> tuple[str, ...]:
        return ("GE", "EF")

    def mod_frequency(self, local_oscillators, key="GE"):
        lo_freq = local_oscillators.get(self.local_oscillator)

        if not lo_freq:
            raise KeyError(
                f"Specified LO '{self.local_oscillator}' is not present in {local_oscillators}."
            )

        match key:
            case "GE":
                return self.frequency - lo_freq
            case "EF":
                return self.frequency_EF - lo_freq if self.frequency_EF else None
            case _:
                raise KeyError(
                    f"'{key}' is not a valid frequency key for {type(self).__name__}."
                )


@register_qsystem
@qdefine
class ReadoutResonator(QuantumSystem):
    frequency: float
    kappa: float | None = None
    chi: float | None = None
    local_oscillator: str | None = None

    def mod_frequency(self, local_oscillators) -> float:
        lo_freq = local_oscillators.get(self.local_oscillator)

        if not lo_freq:
            raise KeyError(
                f"Specified LO '{self.local_oscillator}' is not present in {local_oscillators}."
            )

        return self.frequency - lo_freq


def make_quantum_system_unstructure_fn(cls):
    unstructure_attrs = make_attrs_unstructure_fn(cls)

    def unstructure_fn(obj):
        return {**unstructure_attrs(obj), "__class__": type(obj).__name__}

    return unstructure_fn


qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, QuantumSystem), make_quantum_system_unstructure_fn
)


__all__ = ["Transmon", "ReadoutResonator"]