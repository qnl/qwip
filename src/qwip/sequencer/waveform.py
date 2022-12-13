import attrs
import numpy as np
from numbers import Number
from typing import get_args

from scipy.fft import fft, fftfreq

from attrs import field

import qwip
from qwip._cattr import make_attrs_structure_fn
from qwip.defaults import dynamic_default
from qwip.settings.settings import qdefine, qfrozen
from qwip.sequencer.utils import LinearExpression

@qfrozen(kw_only=False)
class ModulationFrequency(LinearExpression):
    ...

@qfrozen(kw_only=False)
class Channel:
    name: str

@qfrozen
class Waveform:
    name: str = field(metadata=dict(allow_overwrite=False))
    channels: tuple[Channel, ...] = field(
        factory=tuple,
        metadata=dict(allow_overwrite=False)
    )
    width: float | str = 0
    amplitude: float | str = 1
    t0: float | str = 0

    evolve = attrs.evolve

    @name.default
    def _default_name(self):
        return type(self).__name__

    def update_fields(self, **kwargs) -> dict:
        fields = {}
        for field in attrs.fields(type(self)):
            if not field.metadata.get('allow_overwrite', True):
                continue

            value = getattr(self, field.name)
            fields[field.name] = kwargs.pop(
                value,
                kwargs.pop(
                    field.name,
                    value
                )
            )

        return fields | {k: v for k, v in kwargs.items() if k not in fields}

    def __call__(self, ts: np.ndarray, **kwargs) -> np.ndarray:
        kwargs = self.update_fields(**kwargs)
        try:
            return self.get_waveform(ts.astype(np.float64), **kwargs)
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
            raise ValueError(text) from e

    def get_waveform(self, ts: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError(f'No waveform defined for {type(self)}!')

    def plot(self):
        ...
    
    def fft(self, ts, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        wave = self.get_waveform(ts, **kwargs)

        N = len(ts)
        fs = fft(wave)
        ks = fftfreq(N, ts[1] - ts[0])

        return ks, fs
    
@qfrozen
class DCWaveform(Waveform):
    def get_waveform(
        self,
        ts: np.ndarray,
        amplitude: float,
        t0: float,
        **kwargs
    ) -> np.ndarray:
        return amplitude * np.ones_like(ts, dtype=np.float32)

@qfrozen
class SquareWaveform(Waveform):
    def get_waveform(
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

@qfrozen
class GaussianWaveform(Waveform):
    cutoff: float | str = 3

    def get_waveform(self,
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

        return wave

@qfrozen
class CosineRamp(Waveform):
    ramp: float = 0.1

@qfrozen
class ModulatedWaveform(Waveform):
    envelope: Waveform = field(metadata=dict(allow_overwrite=False))
    width: float | str = field()
    amplitude: float | str = field()
    t0: float | str = field()
    mod_freq: ModulationFrequency
    phase: float | str = 0

    @width.default
    def _get_width_from_envelope(self):
        return self.envelope.width

    @t0.default
    def _get_t0_from_envelope(self):
        return self.envelope.t0

    @amplitude.default
    def _get_amplitude_from_envelope(self):
        return self.envelope.amplitude

    @dynamic_default(phase_unit='units/phase')
    def get_waveform(self,
        ts: np.ndarray,
        phase: float,
        phase_tracker: dict[ModulationFrequency, np.ndarray] | None = None,
        modulations: dict[str, ModulationFrequency] = {},
        phase_unit: str = None,
        **kwargs
    ):
        """Modulated waveform.
        
        A modulated waveform takes a waveform envelope and convolves it with a
        single frequency valued waveform. If a phase tracker is supplied, the
        relevant phases will also be added to the 

        Args:
            ts: Time values at which to evaluate the pulse.
            phase: The starting phase of the modulation tone.
            phase_tracker: A phase tracking dictionary mapping modulation channels
                to a ndarray of times and discrete phase jumps.
            phase_unit: Either degrees or radians, specifies the phase units.
                Defaults to the value set in `qsettings['units/phase']`.
            **kwargs: Remaining keyword arguments are passed to `get_waveform` for
                the envelope Waveform.

        Returns:
            A ndarray containing the function w(t) evaluated at the specified
            times.
        """
        freq = 2*np.pi*self.mod_freq.resolve(**modulations).offset

        if phase_tracker:
            phis = self.compute_integrated_phase(ts, phase_tracker)
        else:
            phis = np.zeros_like(ts)

        if phase_unit.lower() == 'degrees':
            phase *= np.pi / 180
            phis *= np.pi / 180

        envelope = self.envelope(ts, **kwargs).astype(np.float32)
        modulation = np.exp(1j*(freq*ts + phis + phase), dtype=np.complex64)

        wave = envelope * modulation

        if len(self.channels) == 1:
            return wave.real
        if len(self.channels) == 2:
            return wave.view(np.float32).reshape(-1, 2).T
        else:
            return np.broadcast_to(wave.real, (len(self.channels), len(wave)))

    def compute_integrated_phase(
        self,
        ts: np.ndarray,
        phase_tracker: dict[ModulationFrequency, np.ndarray],
    ):
        phase_jumps = phase_tracker[self.mod_freq]

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

@qfrozen
class MarkerWaveform(Waveform):
    ...

@qfrozen
class VirtualZWaveform(MarkerWaveform):
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