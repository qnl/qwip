import attrs
from attrs import validators
import numpy as np
from numbers import Number
from typing import get_args
from typing_extensions import Self
from functools import lru_cache

from scipy.fft import fft, fftfreq, fftshift
from loguru import logger

from attrs import field

import qwip
from qwip._cattr import make_attrs_structure_fn, make_attrs_unstructure_fn
from qwip.defaults import dynamic_default
from qwip.settings.settings import qdefine, qfrozen
from qwip.sequencer.utils import LinearExpression
from qwip.typing import is_union_type

REGISTERED_WAVEFORMS: dict[str, 'Waveform'] = dict()

def register_waveform(cls) -> type:
    if not issubclass(cls, Waveform):
        raise TypeError(f'Registered waveforms must subclass {Waveform}.')

    REGISTERED_WAVEFORMS[cls.__name__] = cls

    return cls

@qfrozen(kw_only=False)
class ModulationFrequency(LinearExpression):
    ...

@qfrozen(kw_only=False)
class Channel:
    name: str

def update_fields(inst, /, **kwargs) -> dict:
    fields = {}

    for field in attrs.fields(type(inst)):
        if not field.metadata.get('allow_override', True):
            continue
        
        value = getattr(inst, field.name)
        fields[field.name] = kwargs.pop(
            value,
            kwargs.pop(
                field.name,
                value
            )
        )

    return fields | {k: v for k, v in kwargs.items() if k not in fields}

@qfrozen
class Waveform:
    name: str = field(metadata=dict(allow_override=False))

    evolve = attrs.evolve

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

                if not isinstance(v, Number) and f.name not in ('name', 'channels'):
                    variables.add((f.name, v))

            text = (
                f'The following string variables need to be resolved:\n\t' +
                '\n\t'.join(f'{n} = {v}' for n, v in variables)
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
            f'Method evaluate_timepoints not defined for {type(self)}!'
        )

    def plot(self):
        ...
    
    def fft(self, ts, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        wave = self(ts, **kwargs)

        if len(wave.shape) > 1 and wave.shape[0] == 2:
            wave = 1j*wave[1] + wave[0]

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
            elif (
                isinstance(var, str) and
                is_union_type(f.type)
            ):
                varset.add(var)
            
        return frozenset(varset)

    def resolve(self, **variable_map) -> Self:
        variable_map = {
            k: v for k, v in variable_map.items() if k in self.variables()
        }

        if not variable_map:
            return self

        to_update = {}

        for f in attrs.fields(type(self)):
            orig = getattr(self, f.name)

            if isinstance(orig, Waveform):
                to_update[f.name] = orig.resolve(**variable_map)
            elif (
                isinstance(orig, str) and
                is_union_type(f.type) and
                (updated := variable_map.get(orig)) is not None
            ):
                to_update[f.name] = updated

        return self.evolve(**to_update)

    def __contains__(self, var: str) -> bool:
        """Returns whether a variable is referenced in the waveform."""
        return var in self.variables()

@register_waveform
@qfrozen
class BasicWaveform(Waveform):
    channels: tuple[Channel, ...] = field(
        factory=tuple,
        metadata=dict(allow_override=False)
    )
    width: float | str = 0
    amplitude: float | str = 1
    t0: float | str = 0

@register_waveform
@qfrozen
class InfiniteWaveform(BasicWaveform):
    width: float | str = np.inf

@register_waveform
@qfrozen
class Marker(Waveform):
    @property
    def channels(self):
        return set()

    @property
    def width(self):
        return 0

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
        self,
        ts: np.ndarray,
        amplitude: float,
        t0: float,
        **kwargs
    ) -> np.ndarray:
        return amplitude * np.ones_like(ts, dtype=np.float32)

@register_waveform
@qfrozen
class CWWaveform(InfiniteWaveform):
    frequency: ModulationFrequency
    phase: float | str = 0

    @dynamic_default(phase_unit='units/phase')
    def evaluate_timepoints(
        self,
        ts: np.ndarray,
        amplitude: float,
        phase: float,
        phase_tracker: dict[ModulationFrequency, np.ndarray] | None = None,
        modulations: dict[str, ModulationFrequency] = {},
        phase_unit: str = None,
        complex_out: str = False,
        **kwargs
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
            phase_unit: Either degrees or radians, specifies the phase units.
                Defaults to the value set in `qsettings['units/phase']`.

        Returns:
            A ndarray containing the function w(t) evaluated at the specified
            times.
        """
        freq = 2*np.pi*self.frequency.resolve(**modulations).offset

        if phase_tracker:
            phis = self.compute_integrated_phase(ts, phase_tracker)
        else:
            phis = np.zeros_like(ts)

        if phase_unit.lower() == 'degrees':
            phase *= np.pi / 180
            phis *= np.pi / 180

        wave = amplitude * np.exp(1j*(freq*ts + phis + phase), dtype=np.complex64)

        if complex_out:
            return wave
        elif len(self.channels) <= 1:
            return wave.real
        elif len(self.channels) == 2:
            return wave.view(np.float32).reshape(-1, 2).T
        else:
            shape = (max(1, len(self.channels)), len(wave))
            return np.broadcast_to(wave.real, shape)

    def compute_integrated_phase(
        self,
        ts: np.ndarray,
        phase_tracker: dict[ModulationFrequency, np.ndarray],
    ):
        phase_jumps = phase_tracker[self.frequency]

        # Find phase_jumps that are relevant for the time slice
        s = np.searchsorted(phase_jumps[:, 0], ts[0])
        e = np.searchsorted(phase_jumps[:, 0], ts[-1], side='right')

        idx = np.searchsorted(ts, phase_jumps[s:e, 0])

        # Set phis equal to last phase before or equal to ts[0]
        phis = phase_jumps[max(s - 1, 0), 1]*np.ones_like(ts)

        N = phis.shape[0]

        for i, (left, right) in enumerate(zip(idx, idx[1:])):
            phis[left:right] = phase_jumps[s + i, 1]

            if right >= N:
                break
        else:
            # Handle any remaining bit
            if len(idx):
                phis[idx[-1]:] = phase_jumps[s + len(idx) - 1, 1]

        return phis

@register_waveform
@qfrozen
class ModulatedWaveform(Waveform):
    envelope: Waveform
    mod_freq: CWWaveform

    @property
    def width(self) -> float | str:
        return self.envelope.width

    @property
    def t0(self) -> float | str:
        return self.envelope.t0

    @property
    def amplitude(self) -> float | str:
        A_e = self.envelope.amplitude
        A_f = self.mod_freq.amplitude
        try:
            return A_e * A_f
        except TypeError:
            return f'{A_e} * {A_f}'

    @property
    def channels(self) -> tuple[Channel]:
        return self.mod_freq.channels

    def evaluate_timepoints(self, ts, **kwargs) -> np.ndarray:
        modulation = self.mod_freq(ts, complex_out=True, **kwargs)
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
    mod_freq: ModulationFrequency
    phase: float | str = 0

    @dynamic_default(phase_unit='units/phase')
    def get_phase_jump(self, phase_unit: str = None, **kwargs):
        phase = kwargs.get(self.phase, self.phase)

        if phase_unit.lower() == 'degrees':
            phase *= np.pi / 180

        return phase

    def update_phase_tracker(
        self,
        time: float,
        phase_tracker: dict[ModulationFrequency, list[tuple[float, float]]],
    ) -> None:
        try:
            previous_time, phi = phase_tracker[self.mod_freq][-1]
        except KeyError as e:
            raise KeyError(f'Modulation {self.mod_freq} not found!') from e

        phase_entry = (time, phi + self.get_phase_jump())

        if time == previous_time:
            phase_tracker[self.mod_freq][-1] = phase_entry
        else:
            phase_tracker[self.mod_freq].append(phase_entry)

@register_waveform
@qfrozen
class SquareWaveform(BasicWaveform):
    def evaluate_timepoints(
        self,
        ts: np.ndarray,
        width: float,
        amplitude: float,
        t0: float,
        **kwargs
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

    def evaluate_timepoints(self,
        ts: np.ndarray,
        width: float,
        amplitude: float,
        t0: float,
        cutoff: float,
        **kwargs
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
        wave = amplitude * np.exp(-0.5 * (ts - width / 2)**2 / sigma**2)

        wave[~((0 <= ts) & (ts <= width))] = 0

        return wave

@register_waveform
@qfrozen
class CosineRampWaveform(BasicWaveform):
    ramp: float = field(
        default=0.1,
        validator=[validators.le(0.5), validators.gt(0)]
    )

    def evaluate_timepoints(
        self,
        ts: np.ndarray,
        width: float,
        amplitude: float,
        t0: float,
        ramp: float,
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
        rlen = ramp * width
        wave = np.zeros_like(ts, dtype=np.float32)

        ramp_up = (0 <= ts) & (ts < rlen)
        wave[ramp_up] = 0.5 * amplitude * (1 - np.cos(np.pi / rlen * ts[ramp_up]))

        const = (rlen <= ts) & (ts < width - rlen)
        wave[const] = amplitude

        ramp_down = (width - rlen <= ts) & (ts < width)
        wave[ramp_down] = 0.5 * amplitude * (
            1 + np.cos(np.pi / rlen * (ts[ramp_down] - width + rlen))
        )

        return wave


@register_waveform
@qfrozen
class DRAG(Waveform):
    lmbda: float | str
    envelope: Waveform

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
            lmbda: The DRAG parameter used to control the amplitude of the 
                quadrature correction.
            envelope: The envelope to apply the DRAG correction to.
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

qwip.converter.register_structure_hook(
    float | str,
    convert_number_or_string
)

def make_channel_structure_fn(cls):
    structure_attrs = make_attrs_structure_fn(cls)

    def structure_fn(obj, cls):
        if isinstance(obj, (str, int)):
            return cls(str(obj))

        return structure_attrs(obj, cls)

    return structure_fn

qwip.converter.register_structure_hook_factory(
    lambda cls: issubclass(cls, Channel),
    make_channel_structure_fn
)

# ========== Waveform converters ========== #

def make_waveform_structure_fn(cls):
    structure_attrs = make_attrs_structure_fn(cls)
    
    def structure_fn(val, cls):
        if isinstance(val, cls):
            return val

        subclass = REGISTERED_WAVEFORMS.get(
            val.get('__class__'),
        )

        if subclass is None:
            logger.warning(
                f'No registered waveform found. Structuring {val} as {cls}.'
            )
            return structure_attrs(val, cls)

        return qwip.converter.structure(val, subclass)

    return structure_fn

def make_waveform_unstructure_fn(cls):
    unstructure_attrs = make_attrs_unstructure_fn(cls)

    def unstructure_fn(obj):
        return {**unstructure_attrs(obj), '__class__': type(obj).__name__}

    return unstructure_fn

qwip.converter.register_structure_hook_factory(
    lambda cls: cls in (BasicWaveform, Waveform),
    make_waveform_structure_fn
)

qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, Waveform),
    make_waveform_unstructure_fn
)