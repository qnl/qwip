import itertools as it
from collections import defaultdict
from functools import lru_cache
from numbers import Number
from typing import get_args

import attrs
import numpy as np
from attrs import field, validators
from loguru import logger
from scipy.fft import fft, fftfreq, fftshift
from typing_extensions import Self

import qwip
from qwip._cattr import make_attrs_structure_fn, make_attrs_unstructure_fn
from qwip.defaults import dynamic_default
from qwip.sequencer.phase_tracker import ModulationFrequency, PhaseJump, PhaseTracker
from qwip.sequencer.utils import LinearExpression, Location
from qwip.settings.settings import qdefine, qfrozen
from qwip.typing import is_union_type

REGISTERED_WAVEFORMS: dict[str, "Waveform"] = dict()


def register_waveform(cls) -> type:
    if not issubclass(cls, Waveform):
        raise TypeError(f"Registered waveforms must subclass {Waveform}.")

    REGISTERED_WAVEFORMS[cls.__name__] = cls

    return cls


@qfrozen(kw_only=False)
class Channel:
    name: str
    subchannel: int = 0


def update_fields(inst, /, **kwargs) -> dict:
    fields = {}

    for field in attrs.fields(type(inst)):
        if not field.metadata.get("allow_override", True):
            continue

        value = getattr(inst, field.name)

        if field.type is Location:
            value = value.resolve(**kwargs)

            if value.resolved:
                fields[field.name] = value.offset
            else:
                value = value if len(value.references) else value.offset
                fields[field.name] = kwargs.get(field.name, value)

            continue

        fields[field.name] = kwargs.pop(value, kwargs.pop(field.name, value))

    return fields | {k: v for k, v in kwargs.items() if k not in fields}


@qfrozen
class Waveform:
    name: str = field(metadata=dict(allow_override=False))

    @property
    def resolved(self) -> bool:
        """True if a Waveform contains no variables.

        Returns:
            A boolean that specifies if a waveform has any variables.
        """
        return not bool(self.variables())

    @name.default
    def _default_name(self):
        return type(self).__name__

    def __call__(self, ts: np.ndarray, **kwargs) -> np.ndarray:
        kwargs = update_fields(self, **kwargs)

        try:
            wave = self.evaluate_timepoints(ts.astype(np.float32), **kwargs)

            return wave
        except TypeError as e:
            variables = set()
            for f in attrs.fields(type(self)):
                v = getattr(self, f.name)

                if not isinstance(v, Number) and f.name not in ("name", "channels"):
                    variables.add((f.name, v))

            text = (
                f"The following string variables need to be resolved:\n\t"
                + "\n\t".join(f"{n} = {v}" for n, v in variables)
            )

            if variables:
                raise ValueError(text) from e
            else:
                raise e

    def __copy__(self) -> Self:
        """Overrides copy for Waveform objects.

        Since Waveforms are immutable and only contain references
        to other immutable objects we just return self instead of
        unnecessarily creating new objects.

        Returns:
            The Waveform object.
        """
        return self

    def __deepcopy__(self, memo) -> Self:
        """Overrides deepcopy for Waveform objects.

        Since Waveforms are immutable and only contain references
        to other immutable objects we just return self instead of
        unnecessarily creating new objects.

        Returns:
            The Waveform object.
        """
        return self

    def evaluate_timepoints(self, ts: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError(
            f"Method evaluate_timepoints not defined for {type(self)}!"
        )

    def plot(self):
        ...

    def fft(self, ts, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        wave = self(ts, **kwargs)

        if len(wave.shape) > 1 and wave.shape[0] == 2:
            wave = 1j * wave[1] + wave[0]

        N = len(ts)
        fs = fft(wave)
        ks = fftfreq(N, ts[1] - ts[0])

        return fftshift(ks), fftshift(fs)

    @lru_cache
    def variables(self) -> frozenset[str]:
        """Returns the set of variables referenced in the waveform."""
        varset = set()

        for f in attrs.fields(type(self)):
            var = getattr(self, f.name)

            if isinstance(var, Waveform):
                varset.update(var.variables())
            elif isinstance(var, LinearExpression):
                varset.update(var.variables(return_string=True))
            elif isinstance(var, str) and is_union_type(f.type):
                varset.add(var)

        return frozenset(varset)

    def resolve(self, **variable_map) -> Self:
        variable_map = {k: v for k, v in variable_map.items() if k in self.variables()}

        if not variable_map:
            return self

        to_update = {}

        for f in attrs.fields(type(self)):
            orig = getattr(self, f.name)

            if isinstance(orig, (LinearExpression, Waveform)):
                to_update[f.name] = orig.resolve(**variable_map)
            elif (
                isinstance(orig, str)
                and is_union_type(f.type)
                and (updated := variable_map.get(orig)) is not None
            ):
                to_update[f.name] = updated

        return attrs.evolve(self, **to_update)

    def evolve(self, **updates):
        # Separate out fields that are also waveforms.
        groupby = it.groupby(
            attrs.fields(type(self)),
            key=lambda f: isinstance(getattr(self, f.name), Waveform),
        )

        field_names = dict(waveform=[], other=[])
        for k, fields in groupby:
            if k:
                field_names["waveform"] += [f.name for f in fields]
            else:
                field_names["other"] += [f.name for f in fields]

        wave_name_set = set(field_names["waveform"])
        # Pull out nested updates
        nested_updates = defaultdict(dict)
        for key in list(updates):
            if "_" not in key:
                continue

            # Need to handle field names that also contain '_'
            tokens = key.split("_")
            combined_tokens = it.accumulate(tokens, lambda acc, x: f"{acc}_{x}")

            for i, name in enumerate(combined_tokens):
                if name not in wave_name_set:
                    continue

                subkey = "_".join(tokens[i + 1 :])
                if subkey:
                    nested_updates[name][subkey] = updates.pop(key)

                break

        to_update = dict()
        # First make pass through non-nested attributes
        for name in field_names["other"]:
            if name in updates:
                to_update[name] = updates.pop(name)

        for name in field_names["waveform"]:
            if name in updates:
                to_update[name] = updates[name]
            else:
                old = getattr(self, name)
                updates_to_wave = updates | nested_updates[name]
                to_update[name] = old.evolve(**updates_to_wave)

        return attrs.evolve(self, **to_update)

    def __contains__(self, var: str) -> bool:
        """Returns whether a variable is referenced in the waveform."""
        return var in self.variables()


@register_waveform
@qfrozen
class BasicWaveform(Waveform):
    channels: tuple[Channel, ...] = field(
        factory=tuple, metadata=dict(allow_override=False)
    )
    width: Location = Location()
    amplitude: float | str = 1
    t0: float | str = 0


@register_waveform
@qfrozen
class InfiniteWaveform(BasicWaveform):
    width: Location = Location(np.inf)


@register_waveform
@qfrozen
class Marker(Waveform):
    @property
    def channels(self):
        return set()

    @property
    def width(self):
        return Location()


@register_waveform
@qfrozen
class CompositeWidthMarker(Marker):
    ...


@register_waveform
@qfrozen
class TriggerMarker(Marker):
    ...


@register_waveform
@qfrozen
class ReadoutMarker(Marker):
    ...


@register_waveform
@qfrozen
class DCWaveform(InfiniteWaveform):
    def evaluate_timepoints(
        self, ts: np.ndarray, amplitude: float, t0: float, **kwargs
    ) -> np.ndarray:
        return amplitude * np.ones_like(ts, dtype=np.float32)


@register_waveform
@qfrozen
class CWWaveform(InfiniteWaveform):
    frequency: ModulationFrequency
    phase: float | str = 0
    mod_key: ModulationFrequency | None = None

    @dynamic_default(phase_unit="units/phase")
    def evaluate_timepoints(
        self,
        ts: np.ndarray,
        amplitude: float,
        phase: float,
        phase_tracker: PhaseTracker | None = None,
        modulations: dict[str, ModulationFrequency] = {},
        phase_unit: str = None,
        complex_out: str = False,
        **kwargs,
    ):
        """Single frequency waveform.

        A CW waveform creates a single frequency valued waveform with infinite
        support. If a phase tracker is supplied, the relevant phases will also
        be adjusted when computing timepoints.

        Args:
            ts: Time values at which to evaluate the pulse.
            amplitude: The amplitude of the modulation tone.
            phase: The starting phase of the modulation tone.
            phase_tracker: A phase tracking dictionary mapping modulation channels
                to a ndarray of times and discrete phase jumps.
            phase_unit: Eithe r degrees or radians, specifies the phase units.
                Defaults to the value set in `qsettings['units/phase']`.

        Returns:
            A ndarray containing the function w(t) evaluated at the specified
            times.
        """
        freq = 2 * np.pi * self.frequency.resolve(**modulations).offset

        if phase_tracker:
            phis = phase_tracker.compute_integrated_phase(
                self.mod_key or self.frequency, ts
            )
        else:
            phis = np.zeros_like(ts)

        if phase_unit.lower() == "degrees":
            phase *= np.pi / 180
            phis *= np.pi / 180

        wave = amplitude * np.exp(1j * (freq * ts + phis + phase), dtype=np.complex64)

        if complex_out:
            return wave
        elif len(self.channels) <= 1:
            return wave.real
        elif len(self.channels) == 2:
            return wave.view(np.float32).reshape(-1, 2).T
        else:
            shape = (max(1, len(self.channels)), len(wave))
            return np.broadcast_to(wave.real, shape)


@register_waveform
@qfrozen
class ModulatedWaveform(Waveform):
    envelope: Waveform
    modulation: CWWaveform

    @property
    def width(self) -> float | str:
        return self.envelope.width

    @property
    def t0(self) -> float | str:
        return self.envelope.t0

    @property
    def amplitude(self) -> float | str:
        A_e = self.envelope.amplitude
        A_f = self.modulation.amplitude
        try:
            return A_e * A_f
        except TypeError:
            return f"{A_e} * {A_f}"

    @property
    def channels(self) -> tuple[Channel]:
        return self.modulation.channels

    def evaluate_timepoints(self, ts, **kwargs) -> np.ndarray:
        modulation = self.modulation(ts, complex_out=True, **kwargs)
        envelope = self.envelope(ts, **kwargs)

        wave = envelope * modulation

        if len(self.channels) <= 1:
            return wave.real
        elif len(self.channels) == 2:
            return wave.view(np.float32).reshape(-1, 2).T
        else:
            shape = (max(1, len(self.channels)), len(wave))
            return np.broadcast_to(wave.real, shape)


@register_waveform
@qfrozen
class VirtualZWaveform(Marker):
    mod_key: ModulationFrequency
    phase: float | str = 0

    def update_phase_tracker(
        self,
        time: float,
        phase_tracker: PhaseTracker,
    ) -> None:
        phase_tracker.append(self.mod_key, PhaseJump(time, self.phase))


@register_waveform
@qfrozen
class SquareWaveform(BasicWaveform):
    def evaluate_timepoints(
        self, ts: np.ndarray, width: float, amplitude: float, t0: float, **kwargs
    ) -> np.ndarray:
        """Square waveform.

        Args:
            ts: Time values at which to evaluate the pulse.
            width: The width of the square pulse.
            amplitude: The amplitude of the pulse.
            t0: The starting time of the pulse.

        Returns:
            A ndarray containing the function w(t) evaluated at the specified
            times.
        """
        ts = ts - t0
        wave = amplitude * ((ts > 0) & (ts < width)).astype(np.float32)

        return wave


@register_waveform
@qfrozen
class GaussianWaveform(BasicWaveform):
    cutoff: float | str = 3

    def evaluate_timepoints(
        self,
        ts: np.ndarray,
        width: float,
        amplitude: float,
        t0: float,
        cutoff: float,
        **kwargs,
    ) -> np.ndarray:
        """Gaussian waveform.

        Args:
            ts: Time values at which to evaluate the pulse.
            width: The width of the Gaussian pulse, the Gaussian waveform will be
                truncated such that w(t) == 0 for t < t0 and t > t0 + width.
            amplitude: The amplitude of the pulse.
            t0: The starting time of the pulse.
            cutoff: How many standard deviations to include in the pulse width.
                This defines the standard deviation as sigma = width / (2 * cutoff).

        Returns:
            A ndarray containing the function w(t) evaluated at the specified
            times.
        """
        ts = ts - t0
        sigma = width / (2 * cutoff)
        wave = amplitude * np.exp(-0.5 * (ts - width / 2) ** 2 / sigma**2)

        wave[~((0 <= ts) & (ts <= width))] = 0

        return wave


@register_waveform
@qfrozen
class CosineRampWaveform(BasicWaveform):
    ramp: float | str | None = None
    ramp_fraction: float | str | None = field(
        default=0.1, validator=[validators.le(0.5), validators.gt(0)]
    )

    def evaluate_timepoints(
        self,
        ts: np.ndarray,
        width: float,
        amplitude: float,
        t0: float,
        ramp: float = None,
        ramp_fraction: float = 0.1,
        **kwargs,
    ) -> np.ndarray:
        """Square pulse with cosine ramps.

        Args:
            ts: Time values at which to evaluate the pulse.
            width: The total width of the cosine ramp pulse, including ramp times.
            amplitude: The amplitude of the pulse.
            t0: The starting time of the pulse.
            ramp: The ramp fraction. The pulse will be smoothly varied from 0 to
                amplitude and vice versa over a time (width * ramp) at the
                beginning and end of the pulse.

        Returns:
            A ndarray containing the function w(t) evaluated at the specified
            times.
        """
        ts = ts - t0
        if ramp is not None:
            rlen = min(ramp, 0.5 * width)
        elif ramp_fraction is not None:
            rlen = ramp_fraction * width
        else:
            raise ValueError("One of ramp or ramp_fraction must be specified!")

        wave = np.zeros_like(ts, dtype=np.float32)

        ramp_up = (0 <= ts) & (ts < rlen)
        wave[ramp_up] = 0.5 * amplitude * (1 - np.cos(np.pi / rlen * ts[ramp_up]))

        const = (rlen <= ts) & (ts < width - rlen)
        wave[const] = amplitude

        ramp_down = (width - rlen <= ts) & (ts < width)
        wave[ramp_down] = (
            0.5
            * amplitude
            * (1 + np.cos(np.pi / rlen * (ts[ramp_down] - width + rlen)))
        )

        return wave


@register_waveform
@qfrozen
class DRAG(Waveform):
    envelope: Waveform
    lmbda: float | str = 0

    @property
    def width(self) -> float | str:
        return self.envelope.width

    @property
    def t0(self) -> float | str:
        return self.envelope.t0

    @property
    def amplitude(self) -> float | str:
        self.envelope.amplitude

    def evaluate_timepoints(self, ts, lmbda, **kwargs) -> np.ndarray:
        """Numerical DRAG pulse.

        Args:
            ts: Time values at which to evaluate the pulse.
            envelope: The envelope to apply the DRAG correction to.
            lmbda: The DRAG parameter used to control the amplitude of the
                quadrature correction.
            **kwargs: All keyword arguments are passed to the envelope function.

        Returns:
            A ndarray containing the function w(t) evaluated at the specified
            times.
        """
        envelope = self.envelope(ts, **kwargs)

        delta_t = ts[1] - ts[0]

        return envelope + 1j * lmbda / delta_t * np.gradient(envelope)


# ========== float | str converters ========== #


def convert_number_or_string(v, cls):
    if isinstance(v, cls):
        return v

    ntype, stype = get_args(cls)

    try:
        return qwip.converter.structure(v, ntype)
    except Exception:
        ...

    try:
        return qwip.converter.structure(v, stype)
    except Exception:
        ...

    return v


qwip.converter.register_structure_hook(float | str, convert_number_or_string)


def make_channel_structure_fn(cls):
    structure_attrs = make_attrs_structure_fn(cls)

    def structure_fn(obj, cls):
        if isinstance(obj, (str, int)):
            return cls(str(obj))

        return structure_attrs(obj, cls)

    return structure_fn


qwip.converter.register_structure_hook_factory(
    lambda cls: issubclass(cls, Channel), make_channel_structure_fn
)

# ========== Waveform converters ========== #


def make_waveform_structure_fn(cls):
    structure_attrs = make_attrs_structure_fn(cls)

    def structure_fn(val, cls):
        if isinstance(val, cls):
            return val

        subclass = REGISTERED_WAVEFORMS.get(
            val.get("__class__"),
        )

        if subclass is None:
            logger.warning(f"No registered waveform found. Structuring {val} as {cls}.")
            return structure_attrs(val, cls)

        return qwip.converter.structure(val, subclass)

    return structure_fn


def make_waveform_unstructure_fn(cls):
    unstructure_attrs = make_attrs_unstructure_fn(cls)

    def unstructure_fn(obj):
        return {**unstructure_attrs(obj), "__class__": type(obj).__name__}

    return unstructure_fn


qwip.converter.register_structure_hook_factory(
    lambda cls: cls in (BasicWaveform, Waveform), make_waveform_structure_fn
)

qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, Waveform), make_waveform_unstructure_fn
)

__all__ = [
    "register_waveform",
    "Channel",
    "Waveform",
    "BasicWaveform",
    "InfiniteWaveform",
    "Marker",
    "CompositeWidthMarker",
    "TriggerMarker",
    "ReadoutMarker",
    "DCWaveform",
    "CWWaveform",
    "ModulatedWaveform",
    "VirtualZWaveform",
    "SquareWaveform",
    "GaussianWaveform",
    "CosineRampWaveform",
    "DRAG",
]
