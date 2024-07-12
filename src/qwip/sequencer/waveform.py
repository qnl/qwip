import itertools as it
from collections import defaultdict
from functools import lru_cache
from numbers import Number, Real
from typing import TYPE_CHECKING, Any, Literal, Self

import attrs
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from attrs import field, validators
from cattr import Converter
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter
from matplotlib.transforms import ScaledTranslation
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import convolve

import qwip
from qwip._cattr import make_attrs_structure_fn, make_attrs_unstructure_fn
from qwip.attrs import qfrozen
from qwip.attrs.serialization import _TypeConverter
from qwip.defaults import dynamic_default
from qwip.sequencer.phase_tracker import Frame, PhaseJump, PhaseTracker
from qwip.sequencer.utils import (
    LinearExpression,
    NumberOrExpression,
    _to_python_number,
    _variable_substitution,
)
from qwip.typing import is_union_type
from qwip.utils import deprecated

if TYPE_CHECKING:
    from qwip.sequencer.timeline import Timeline

REGISTERED_WAVEFORMS: dict[str, "Waveform"] = dict()


def register_waveform(cls) -> type:
    if not issubclass(cls, Waveform):
        raise TypeError(f"Registered waveforms must subclass {Waveform}.")

    REGISTERED_WAVEFORMS[cls.__name__] = cls

    return cls


@qfrozen
class Operation:
    name: str = field(metadata=dict(allow_override=False))

    @property
    def channel(self) -> str:
        return ""

    @property
    def resolved(self) -> bool:
        """True if an Operation contains no variables.

        Returns:
            A boolean that specifies if a waveform has any variables.
        """
        return not bool(self.variables())

    @name.default
    def _default_name(self):
        return type(self).__name__

    def __getattribute__(self, name: str) -> Any:
        return _to_python_number(object.__getattribute__(self, name))

    def __copy__(self) -> Self:
        """Overrides copy for Operation instances.

        Since Operations are immutable and only contain references
        to other immutable objects we just return self instead of
        unnecessarily creating new objects.

        Returns:
            The Operation object.
        """
        return self

    def __deepcopy__(self, memo) -> Self:
        """Overrides deepcopy for Opoeration instances.

        Since Operations are immutable and only contain references
        to other immutable objects we just return self instead of
        unnecessarily creating new objects.

        Returns:
            The Operation object.
        """
        return self

    @classmethod
    @lru_cache
    def evolvable(cls) -> set[str]:
        return set((f.name for f in attrs.fields(cls) if f.init))

    def resolve(self, **variable_map) -> Self:
        from qwip.sequencer.timeline import Timeline

        variable_map = {k: v for k, v in variable_map.items() if k in self.variables()}

        if not variable_map:
            return self

        to_update = {}

        for f in attrs.fields(type(self)):
            if not f.init:
                continue

            orig = getattr(self, f.name)

            match orig:
                case Operation():
                    to_update[f.name] = orig.resolve(**variable_map)
                case sym.Expr():
                    to_update[f.name] = _variable_substitution(orig, variable_map)
                case LinearExpression():
                    converted = {}
                    for k, v in variable_map.items():
                        v = qwip.converter.unstructure(v)
                        if isinstance(v, str):
                            converted[k] = type(orig).from_string(v)
                        else:
                            converted[k] = v
                    to_update[f.name] = orig.resolve(**converted)
                case Timeline() if set(variable_map) & orig.variables():
                    new = orig.copy()
                    new.resolve(**variable_map, inplace=True)
                    to_update[f.name] = new

        if not to_update:
            return self

        return attrs.evolve(self, **to_update)

    def assign_channel(self, new_channel, /) -> Self:
        """Returns a modified waveform with a new channel."""
        raise NotImplementedError(
            f"Cannot assign channel for object of type {type(self)}"
        )

    def evolve(self, **updates):
        # Separate out fields that are also operations.
        groupby = it.groupby(
            attrs.fields(type(self)),
            key=lambda f: isinstance(getattr(self, f.name), Operation),
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
            if name in updates and name in self.evolvable():
                to_update[name] = updates.pop(name)

        for name in field_names["waveform"]:
            if name in updates and name in self.evolvable():
                to_update[name] = updates[name]
            else:
                old = getattr(self, name)
                updates_to_wave = updates | nested_updates[name]
                to_update[name] = old.evolve(**updates_to_wave)

        return attrs.evolve(self, **to_update)

    @lru_cache
    def variables(self) -> frozenset[str]:
        """Returns the set of variables referenced in the waveform."""
        from qwip.sequencer.timeline import Timeline

        varset = set()

        for f in attrs.fields(type(self)):
            var = getattr(self, f.name)

            match var:
                case Operation():
                    varset.update(var.variables())
                case Timeline():
                    varset.update(var.variables())
                case sym.Expr():
                    varset.update(s.name for s in var.free_symbols)
                case LinearExpression():
                    varset.update(var.variables(return_string=True))
                case _:
                    ...

        return frozenset(varset)

    def __contains__(self, var: str) -> bool:
        """Returns whether a variable is referenced in the waveform."""
        return var in self.variables()


@qfrozen
class Waveform(Operation):
    def _update_fields(self, **kwargs) -> dict[str, Number]:
        fields = {}

        symbols = self.variables()

        for f in attrs.fields(type(self)):
            if not f.metadata.get("allow_override", True):
                continue

            value = getattr(self, f.name)
            match value:
                case sym.Expr():
                    subs = {
                        k: v
                        for k, v in kwargs.items()
                        if isinstance(v, (str, Real, sym.Expr))
                    }
                    fields[f.name] = value = _to_python_number(
                        _variable_substitution(value, subs)
                    )
                case str():
                    fields[f.name] = kwargs.get(value, kwargs.get(f.name, value))
                case _:
                    fields[f.name] = value

        return fields | {k: v for k, v in kwargs.items() if k not in symbols}

    def __call__(self, ts: np.ndarray, **kwargs) -> np.ndarray:
        kwargs = self._update_fields(**kwargs)

        try:
            wave = self.evaluate_timepoints(ts.astype(np.float32), **kwargs)
            return wave
        except TypeError as e:
            variables = set()
            for f in attrs.fields(type(self)):
                v = getattr(self, f.name)

                if not isinstance(v, Number) and f.name not in ("name", "channel"):
                    variables.add((f.name, v))

            text = (
                "The following string variables need to be resolved:\n\t"
                + "\n\t".join(f"{n} = {v}" for n, v in variables)
            )

            if variables:
                raise ValueError(text) from e
            else:
                raise e

    def __mul__(self, other) -> Self:
        return ConvolvedWaveform(a=self, b=other)

    def evaluate_timepoints(self, ts: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError(
            f"Method evaluate_timepoints not defined for {type(self)}!"
        )

    def plot(
        self,
        *,
        ts: np.ndarray | None = None,
        variables: dict[str, float] = {},
        ax: Axes | None = None,
        sample_rate: float | None = None,
        label: str = "",
        fig_kwargs: dict = {},
    ) -> Figure:
        wave = self.resolve(**variables)

        if wvars := wave.variables():
            raise ValueError(
                f"Cannot plot wave without concrete values for: {", ".join(wvars)}"
            )

        if ts is None:
            sample_rate = sample_rate or 101 / wave.width
            N = int(sample_rate * wave.width)
            ts = wave.t0 + np.r_[: N + 1] / sample_rate

        if ax is None:
            fig, ax = plt.subplots(**fig_kwargs)
        else:
            fig = ax.get_figure()

        w_t = wave(ts)
        ax.plot(ts, w_t.real)
        ax.plot(ts, w_t.imag)
        ax.xaxis.set_major_formatter(EngFormatter(unit="s"))

        return fig

    def fft(
        self, ts: np.ndarray, variables: dict[str, float] = {}
    ) -> tuple[np.ndarray, np.ndarray]:
        wave = self(ts, **variables)

        if len(wave.shape) > 1 and wave.shape[0] == 2:
            wave = 1j * wave[1] + wave[0]

        N = len(ts)
        fs = fft(wave)
        ks = fftfreq(N, ts[1] - ts[0])

        return fftshift(ks), fftshift(fs)


@register_waveform
@qfrozen
class TimedWaveform(Waveform):
    t0: NumberOrExpression = 0
    width: NumberOrExpression = 0
    channel: str = field(
        default="",
        metadata=dict(allow_override=False),
    )

    @property
    @deprecated(
        version="24.6.0",
        removed="24.8.0",
        message=(
            "Waveforms are now restricted to a single channel. Use `Waveform.channel` "
            "instead."
        ),
    )
    def channels(self) -> tuple[str]:
        return (self.channel,) if self.channel else tuple()

    def evaluate_timepoints(
        self, ts: np.ndarray, width: float, t0: float, **kwargs
    ) -> np.ndarray:
        return np.zeros((len(self.channels), len(ts)))

    def assign_channel(self, new_channel, /) -> Self:
        """Returns a modified waveform with a new channel."""
        return self.evolve(channel=new_channel)


@register_waveform
@qfrozen
class Delay(TimedWaveform):
    hardware: bool = False


@register_waveform
@qfrozen
class BasicWaveform(TimedWaveform):
    amplitude: NumberOrExpression = 1
    phase: NumberOrExpression = 0

    @property
    def complex_amp(self):
        return self.amplitude * sym.exp(1j * self.phase * sym.pi / 180)


@register_waveform
@qfrozen
class InfiniteWaveform(BasicWaveform):
    width: NumberOrExpression = field(default=sym.oo, init=False)


@register_waveform
@qfrozen
class Marker(TimedWaveform):
    width: NumberOrExpression = field(default=0, init=False)

    def plot(
        self,
        *,
        ts: np.ndarray | None = None,
        variables: dict[str, float] = {},
        ax: Axes | None = None,
        sample_rate: float | None = None,
        label: str = "",
        fig_kwargs: dict = {},
    ) -> Figure:
        wave = self.resolve(**variables)

        if not isinstance(wave.t0, Real):
            raise ValueError(f"Cannot plot Marker with t0 = {self.t0}")

        if ax is None:
            fig, ax = plt.subplots(**fig_kwargs)
        else:
            fig = ax.get_figure()

        label = label or f"{wave.name}"
        offset = ScaledTranslation(10 / 72, 0, fig.dpi_scale_trans)
        text_transform = ax.get_xaxis_transform() + offset

        ax.axvline(wave.t0)
        ax.text(wave.t0, 0.9, label, transform=text_transform)
        ax.autoscale_view()
        ax.xaxis.set_major_formatter(EngFormatter(unit="s"))

        return fig


@qfrozen
class ConvolvedWaveform(Waveform):
    a: Waveform
    b: Waveform = field()

    @b.validator
    def _validate_operands(self, attribute, value):
        a = self.a
        b = value
        if a.channel and b.channel and a.channel != b.channel:
            raise ValueError(
                f"Cannot convolve waveforms on different channels. Got channels "
                f"'{a.channel}' != '{b.channel}'"
            )

    @property
    def width(self) -> NumberOrExpression:
        return self.a.width + self.b.width

    @property
    def t0(self) -> NumberOrExpression:
        return self.a.t0 + self.b.t0

    @property
    def phase(self) -> NumberOrExpression:
        return self.a.phase + self.b.phase

    @property
    def channel(self) -> str:
        return self.a.channel or self.b.channel

    def evaluate_timepoints(self, ts: np.ndarray, **kwargs) -> np.ndarray:
        t0 = kwargs.get("t0", self.t0)

        start = ts[0]
        (N,) = ts.shape

        t0 = ((t0 - start) / 2).astype(np.float32)

        t_eval = ts - start
        t_eval = np.r_[-t_eval[::-1], t_eval[1:]]

        a_t = self.a(t_eval, **(kwargs | dict(t0=t0)))
        b_t = self.b(t_eval, **(kwargs | dict(t0=t0)))

        w_t = convolve(a_t, b_t, mode="full")

        return w_t[2 * (N - 1) : 3 * (N - 1) + 1]

    def assign_channel(self, new_channel, /) -> Self:
        """Returns a modified waveform with a new channel."""
        return self.evolve(a_channel=new_channel, b_channel=new_channel)


@register_waveform
@qfrozen
class TriggeredWaveform(BasicWaveform):
    target: "qwip.sequencer.timeline.Timeline | None" = field(
        default=None, eq=id, metadata=dict(allow_override=False)
    )

    def evaluate_timepoints(
        self, ts: np.ndarray, width: float, amplitude: float, t0: float, **kwargs
    ) -> np.ndarray:
        ts = ts - t0
        wave = ((ts > 0) & (ts < width)).astype(np.int8)

        return wave


@register_waveform
@qfrozen
class ReadoutMarker(Marker): ...


@register_waveform
@qfrozen
class DCWaveform(InfiniteWaveform):
    def evaluate_timepoints(
        self, ts: np.ndarray, amplitude: float, phase: float, t0: float, **kwargs
    ) -> np.ndarray:
        rphi = amplitude * np.exp(1j * phase * np.pi / 180)
        wave = rphi * np.ones_like(ts, dtype=np.complex64)
        return wave

    def plot(
        self,
        *,
        ts: np.ndarray | None = None,
        variables: dict[str, float] = {},
        ax: Axes | None = None,
        sample_rate: float | None = None,
        label: str = "",
        fig_kwargs: dict = {},
    ) -> Figure:
        wave = self.resolve(**variables)

        if isinstance(wave.amplitude, Real):
            raise ValueError(
                f"Cannot plot DCWaveform with amplitude = {self.ampllitude}"
            )

        if ax is None:
            fig, ax = plt.subplots(**fig_kwargs)
        else:
            fig = ax.get_figure()

        ax.axhline(wave.t0)
        ax.autoscale_view()
        ax.xaxis.set_major_formatter(EngFormatter(unit="s"))

        return fig


@register_waveform
@qfrozen
class CWWaveform(InfiniteWaveform):
    frequency: Frame
    phase: NumberOrExpression = 0
    # offset: NumberOrExpression = field(
    #     default=0, converter=lambda v: float(v) if isinstance(v, int) else v
    # )
    frame: Frame | None = None
    hardware_modulation: bool = False

    @dynamic_default(phase_unit="units/phase")
    def evaluate_timepoints(
        self,
        ts: np.ndarray,
        amplitude: float,
        phase: float,
        phase_tracker: PhaseTracker | None = None,
        frames: dict[str, Frame] = {},
        phase_unit: str = None,
        **kwargs,
    ) -> np.ndarray:
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
            phase_unit: Either degrees or radians, specifies the phase units.
                Defaults to the value set in `qsettings['units/phase']`.

        Returns:
            A ndarray containing the function w(t) evaluated at the specified
            times.
        """
        if phase_tracker:
            # phase tracking frame and drive frequency are different and neither is None
            if len({self.frequency, self.frame, None}) == 3:
                df = self.frequency - self.frame
            else:
                df = Frame()

            frame = self.frame or self.frequency
            phis = phase_tracker.compute_integrated_phase(frame, ts)
            software_oscillator = phase_tracker.compute_oscillator_phase(
                frame, ts, frames, df
            )
        else:
            phis = np.zeros_like(ts)
            freq = 2 * np.pi * self.frequency.resolve(**frames).offset
            software_oscillator = freq * ts

        if phase_unit.lower() == "degrees":
            phase *= np.pi / 180
            phis *= np.pi / 180

        # Add base modulation at the relevant frequency if doing software modulation
        oscillator = 0 if self.hardware_modulation else software_oscillator
        amplitude = 1 if self.hardware_modulation else amplitude
        phase = 0 if self.hardware_modulation else phase
        wave = amplitude * np.exp(1j * (oscillator + phis + phase), dtype=np.complex64)

        return wave


@register_waveform
@qfrozen
class ModulatedWaveform(Waveform):
    envelope: Waveform
    modulation: CWWaveform

    @property
    def width(self) -> NumberOrExpression:
        return self.envelope.width

    @property
    def t0(self) -> NumberOrExpression:
        return self.envelope.t0

    @property
    def amplitude(self) -> NumberOrExpression:
        A_e = self.envelope.amplitude
        A_m = self.modulation.amplitude

        return A_e * A_m

    @property
    def phase(self) -> NumberOrExpression:
        phi_e = self.envelope.phase
        phi_m = self.modulation.phase

        return phi_e + phi_m

    @property
    def channel(self) -> str:
        return self.modulation.channel

    def evaluate_timepoints(
        self, ts: np.ndarray, complex_out: bool = False, **kwargs
    ) -> np.ndarray:
        modulation = self.modulation(ts, **kwargs)
        envelope = self.envelope(ts, **kwargs)

        wave = envelope * modulation

        return wave

    def update_phase_tracker(
        self,
        time: float,
        phase_tracker: PhaseTracker,
        frames: dict[str, Frame] = {},
    ) -> None:
        modulation = self.modulation
        if len({modulation.frequency, modulation.frame, None}) == 3:
            df = (modulation.frequency - modulation.frame).resolve(**frames).offset
            tau = self.width.offset

            phi = df * tau * 360
            t0 = time + tau
            phis = modulation.frame.distribute_phase(phi)

            for frame, phase in phis.items():
                phase_tracker.append(frame, PhaseJump(t0, phase))

    def assign_channel(self, new_channel, /) -> Self:
        """Returns a modified waveform with a new channel."""
        return self.evolve(modulation_channel=new_channel)


@register_waveform
@qfrozen
class VirtualZWaveform(Marker):
    """A Virtual-Z operation.

    Virtual-Z gates are implemented by applying a phase update to a particular reference
    frame that tracks the qubit phase evolution.
    """

    frame: Frame
    phase: NumberOrExpression = 0

    def update_phase_tracker(
        self,
        time: float,
        phase_tracker: PhaseTracker,
        frames: dict[str, Frame] = {},
    ) -> None:
        phase_tracker.append(self.frame, PhaseJump(time, self.phase))

    def plot(
        self,
        *,
        ts: np.ndarray | None = None,
        variables: dict[str, float] = {},
        ax: Axes | None = None,
        sample_rate: float | None = None,
        label: str = "",
        fig_kwargs: dict = {},
    ) -> Figure:
        return super().plot(
            ts=ts,
            variables=variables,
            ax=ax,
            sample_rate=sample_rate,
            label=label or f"VZ[{self.frame}, {self.phase}]",
            fig_kwargs=fig_kwargs,
        )


@register_waveform
@qfrozen
class PhaseResetWaveform(Marker):
    frame: Frame

    def update_phase_tracker(
        self, time, phase_tracker: PhaseTracker, frames: dict[str, Frame] = {}
    ) -> None:
        phase_tracker.reset(self.frame, time)


@register_waveform
@qfrozen
class SquareWaveform(BasicWaveform):
    def evaluate_timepoints(
        self,
        ts: np.ndarray,
        width: float,
        amplitude: float,
        phase: float,
        t0: float,
        **kwargs,
    ) -> np.ndarray:
        """Square waveform.

        Args:
            ts: Time values at which to evaluate the pulse.
            width: The width of the square pulse.
            amplitude: The amplitude of the pulse.
            phase: The phase of the pulse.
            t0: The starting time of the pulse.

        Returns:
            A ndarray containing the function w(t) evaluated at the specified
            times.
        """
        ts = ts - t0
        rphi = amplitude * np.exp(1j * phase * np.pi / 180)
        wave = rphi * ((ts > 0) & (ts < width)).astype(np.complex64)

        return wave


@register_waveform
@qfrozen
class GaussianWaveform(BasicWaveform):
    cutoff: NumberOrExpression = 3

    def evaluate_timepoints(
        self,
        ts: np.ndarray,
        width: float,
        amplitude: float,
        phase: float,
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
            phase: The phase of the pulse.
            t0: The starting time of the pulse.
            cutoff: How many standard deviations to include in the pulse width.
                This defines the standard deviation as sigma = width / (2 * cutoff).

        Returns:
            A ndarray containing the function w(t) evaluated at the specified
            times.
        """
        ts = ts - t0
        sigma = width / (2 * cutoff)
        rphi = amplitude * np.exp(1j * phase * np.pi / 180)
        wave = rphi * np.exp(
            -0.5 * (ts - width / 2) ** 2 / sigma**2, dtype=np.complex64
        )

        wave[~((0 <= ts) & (ts <= width))] = 0

        return wave


@register_waveform
@qfrozen
class CosineRampWaveform(BasicWaveform):
    ramp: NumberOrExpression | None = None
    ramp_fraction: NumberOrExpression | None = field(
        default=0.1, validator=[validators.le(0.5), validators.gt(0)]
    )

    def evaluate_timepoints(
        self,
        ts: np.ndarray,
        width: float,
        amplitude: float,
        phase: float,
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
            phase: The phase of the pulse.
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

        wave = np.zeros_like(ts, dtype=np.complex64)

        rphi = amplitude * np.exp(1j * phase * np.pi / 180)

        ramp_up = (0 <= ts) & (ts < rlen)
        wave[ramp_up] = 0.5 * rphi * (1 - np.cos(np.pi / rlen * ts[ramp_up]))

        const = (rlen <= ts) & (ts < width - rlen)
        wave[const] = rphi

        ramp_down = (width - rlen <= ts) & (ts < width)
        wave[ramp_down] = (
            0.5 * rphi * (1 + np.cos(np.pi / rlen * (ts[ramp_down] - width + rlen)))
        )

        return wave


@register_waveform
@qfrozen
class DRAG(Waveform):
    envelope: Waveform
    lmbda: NumberOrExpression = 0
    lmbda2: NumberOrExpression = 0

    @property
    def width(self) -> NumberOrExpression:
        return self.envelope.width

    @property
    def t0(self) -> NumberOrExpression:
        return self.envelope.t0

    @property
    def amplitude(self) -> NumberOrExpression:
        self.envelope.amplitude

    @property
    def phase(self) -> NumberOrExpression:
        return self.envelope.phase

    @property
    def channel(self) -> str:
        return self.envelope.channel

    def evaluate_timepoints(
        self, ts: np.ndarray, lmbda: float, lmbda2: float, **kwargs: Any
    ) -> np.ndarray:
        """Numerical DRAG pulse.

        Args:
            ts: Time values at which to evaluate the pulse.
            lmbda: The DRAG parameter used to control the amplitude of the
                quadrature correction.
            **kwargs: All keyword arguments are passed to the envelope function.

        Returns:
            A ndarray containing the function w(t) evaluated at the specified
            times.
        """
        envelope = self.envelope(ts, **kwargs)

        d1 = np.gradient(envelope, ts)
        d2 = np.gradient(d1, ts)

        return envelope + 1j * lmbda * d1 + lmbda2 * d2

    def assign_channel(self, new_channel, /) -> Self:
        """Returns a modified waveform with a new channel."""
        return self.evolve(envelope_channel=new_channel)


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

        if subclass is VirtualZWaveform and "mod_key" in val:
            logger.warning(
                "VirtualZWavefrom 'mod_key' has been renamed to 'frame' and is now "
                "deprecated. Update all unstructured waveforms accordingly."
            )
            val["frame"] = val["mod_key"]
            del val["mod_key"]

        if "channels" in val:
            logger.warning(
                "Waveform 'channels' has been renamed to 'channel' and is now "
                "deprecated. Waveforms can only have a single string channel now."
            )
            val["channel"] = "".join(sorted(val["channels"]))
            del val["channels"]

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
    "Waveform",
    "BasicWaveform",
    "ConvolvedWaveform",
    "InfiniteWaveform",
    "Marker",
    "TriggeredWaveform",
    "ReadoutMarker",
    "DCWaveform",
    "CWWaveform",
    "ModulatedWaveform",
    "VirtualZWaveform",
    "SquareWaveform",
    "GaussianWaveform",
    "CosineRampWaveform",
    "PhaseResetWaveform",
    "DRAG",
]
