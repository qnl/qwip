"""Fitting routines and helpers.

All models should have guess functions implemented.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from lmfit import Model, Parameters

from qwip.analysis.frequency import get_frequency_phase, simple_fft


class GuessModel(Model):
    """A lmfit.Model that auto-initializes parameters before fitting."""

    def __init__(
        self,
        model: Callable,
        independent_vars: list[str] = ["x"],
        prefix: str = "",
        nan_policy: str = "raise",
        **kwargs,
    ):
        super().__init__(
            model,
            independent_vars=independent_vars,
            prefix=prefix,
            nan_policy=nan_policy,
            **kwargs,
        )

    def fit(self, data: np.ndarray, params: Parameters = None, **kwargs: Any):
        """Calls `Model.fit` with initialized parameters from the guess function.

        Guess parameters can be overriden by explicitly specifying an initial
        parameter value.

        Args:
            data: The data to fit.
            params: Initial parameter values.
            **kwargs: Remaining keyword arguments are passed to `Model.fit`
        """
        independent_vars = dict()

        if params:
            independent_vars.update(
                {
                    name: params[name].value
                    for name in self.independent_vars
                    if name in params
                }
            )

        independent_vars.update(
            {name: kwargs[name] for name in self.independent_vars if name in kwargs}
        )

        guess = self.guess(data, **independent_vars)

        if params:
            guess.update(**params)

        return super().fit(data, params=guess, **kwargs)

    def guess(self, data: np.ndarray, x: np.ndarray, **kwargs):
        """Default guess function simply initializes the parameters."""

        return self.make_params(**kwargs)


class FrequencyModel(GuessModel):
    """A model for fitting single-frequency data.

    $$ S(t; A, B, f, \\phi) = A \\sin(2\\pi f t + \\phi) + B$$


    Attributes:
        offset: If False, `B` is constrained to be 0 upon initialization.
    """

    def __init__(
        self,
        offset: bool = True,
        prefix="",
        nan_policy="raise",
        name="A*cos(2*pi*f*t + phi) + B",
        **kwargs,
    ):
        self.offset = offset
        super().__init__(
            type(self).func,
            independent_vars=["t"],
            prefix=prefix,
            nan_policy=nan_policy,
            name=name,
            **kwargs,
        )

        self.set_param_hint("frequency", min=0)
        self.set_param_hint("A", min=0)
        self.set_param_hint("phi", min=-np.pi - 1e-5, max=np.pi + 1e-5)
        self.set_param_hint("B", vary=self.offset)

    @classmethod
    def func(cls, t, A=1, B=0, frequency=1, phi=0):
        """Cos function."""

        return A * np.cos(2 * np.pi * frequency * t + phi) + B

    def guess(self, data: np.ndarray, t: np.ndarray) -> Parameters:
        """Determines initial fit parameters.

        The frequency is estimated via an fft.

        Returns:
            An initialized `Parameters` dictionary.
        """

        A0 = 0.5 * (np.max(data) - np.min(data))
        B0 = np.mean(data) if self.offset else 0

        ## FFT phase is referenced to cosine
        fs, yfs = simple_fft(t, data, subtract_mean=True)
        f0, phi0 = get_frequency_phase(fs, yfs)

        resid = []

        for sgn in (1, -1):
            yguess = A0 * np.cos(2 * np.pi * f0 * t + sgn * phi0)
            resid.append(np.sum((yguess - data) ** 2))

        if np.argmin(resid):
            phi0 *= -1

        return self.make_params(frequency=f0, A=A0, B=B0, phi=phi0)


class TriangularWaveModel(GuessModel):
    """A model for fitting a triangular wave.

    $$ S(t; A, B, p, o) = \\frac{4A}{p} \\left|(t + op) \\mod p - \\frac{p}{2}\\right| + (B - A)$$

    Attributes:
        offset: If False, `B` is constrained to be 0 upon initialization.
    """

    def __init__(
        self,
        offset: bool = True,
        prefix="",
        nan_policy="raise",
        name="4 * A / period * abs((t + offset * period) % period - period / 2) + (B - A)",
        **kwargs,
    ):
        self.offset = offset
        super().__init__(
            type(self).func,
            independent_vars=["t"],
            prefix=prefix,
            nan_policy=nan_policy,
            name=name,
            **kwargs,
        )

        self.set_param_hint("period", min=0)
        self.set_param_hint("A", min=0)
        self.set_param_hint("offset", min=-0.5 - 1e-5, max=0.5 + 1e-5)
        self.set_param_hint("B", vary=self.offset)

    @classmethod
    def func(cls, t, A=1, B=0, period=1, offset=0):
        """Triangular wave function."""

        return (
            4 * A / period * np.abs((t + offset * period) % period - period / 2) - A + B
        )

    def guess(self, data: np.ndarray, t: np.ndarray) -> Parameters:
        """Determines initial fit parameters.

        The frequency is estimated via an fft.

        Returns:
            An initialized `Parameters` dictionary.
        """

        A0 = 0.5 * (np.max(data) - np.min(data))
        B0 = np.mean(data) if self.offset else 0

        ## FFT phase is referenced to cosine
        fs, yfs = simple_fft(t, data, subtract_mean=True)
        f0, phi0 = get_frequency_phase(fs, yfs)

        offset0 = phi0 / (2 * np.pi)
        period0 = 1 / f0

        resid = []
        for sgn in (1, -1):
            yguess = type(self).func(
                t=t, A=A0, B=B0, period=period0, offset=sgn * offset0
            )
            resid.append(np.sum((yguess - data) ** 2))

        if np.argmin(resid):
            offset0 *= -1

        return self.make_params(period=period0, A=A0, B=B0, offset=offset0)


class MobiusModel(GuessModel):
    """A model for fitting a generalized resonator response.

    $$ S(f;A,\\phi,f_0,f_p,\\tau) = Ae^{i\\phi} \\left(\\frac{f - f_0}{f - f_p}\\right) e^{2\\pi if\\tau}$$

    """

    def __init__(
        self,
        prefix="",
        nan_policy="raise",
        name="A*exp(1j*phi)*(f-f0)/(f-fp)*exp(2*pi*(f - Re(f0))*tau)",
        **kwargs,
    ):
        super().__init__(
            type(self).func,
            independent_vars=["f"],
            prefix=prefix,
            nan_policy=nan_policy,
            name=name,
            **kwargs,
        )

        self.set_param_hint("phi", min=-np.pi - 1e-5, max=np.pi + 1e-5)

    @classmethod
    def func(cls, f, A, phi, f0_r, f0_i, fp_r, fp_i, tau=0):
        """Mobius transform.

        This maps a real-valued frequency to a circle in the complex plane. `f0` is the
        point in the frequency domain mapping to zero in the complex plane. `fp` is the
        point in the frequency domain mapping to infinity in the complex plane, aka a
        pole.

        Args:
            f: A frequency.
            A: An overall amplitude scaling factor.
            phi: An overall phase factor.
            f0_r: The real component of `f0`.
            f0_i: The imaginary component of `f0`.
            fp_r: The real component of `fp`.
            fp_i: The complex component of `fp`.
            tau: An electrical delay.

        Returnss:
            A point in the complex plane representing the scattering response of a
            resonator.
        """
        f0 = f0_r + 1j * f0_i
        fp = fp_r + 1j * fp_i

        delay = np.exp(2 * np.pi * 1j * (f - f0_r) * tau)
        S = A * np.exp(1j * phi) * (f - f0) / (f - fp) * delay

        return S

    def guess(self, data: np.ndarray, f: np.ndarray):
        """Determines initial fit parameters.

        Initial parameters are determined via a linear least squares fit assuming zero
        electrical delay. Frequencies are normalized before performing the fit to
        prevent singularities from large values.

        Returns:
            An initialized `Parameters` dictionary.
        """
        S = data

        fmin = np.min(f)
        fmax = np.max(f)
        df = (fmax - fmin) / len(f)
        f_normalized = (f - fmin) / df

        ## Do a linear least squares fit to get initial guess
        ## a@x = b
        b = S * f_normalized
        a = np.stack([f_normalized, S, np.ones_like(f_normalized)]).T

        x = np.linalg.lstsq(a, b, rcond=None)[0]

        A = np.abs(x[0])
        phi = np.angle(x[0])
        fp = x[1] * df + fmin
        f0 = -x[2] / x[0] * df + fmin

        return self.make_params(
            A=A, phi=phi, f0_r=f0.real, f0_i=f0.imag, fp_r=fp.real, fp_i=fp.imag, tau=0
        )


class ReflectionResonatorModel(MobiusModel):
    """A model for fitting a reflection resonator response."""

    def __init__(
        self,
        prefix="",
        nan_policy="raise",
        **kwargs,
    ):
        super().__init__(
            prefix=prefix,
            nan_policy=nan_policy,
            **kwargs,
        )

        self.set_param_hint("fr", expr="(f0_r + fp_r) / 2")
        self.set_param_hint("kappa_i", min=1e-12, expr="fp_i + f0_i")
        self.set_param_hint("kappa_e", min=1e-12, expr="fp_i - f0_i")
        self.set_param_hint("kappa", min=1e-12, expr="kappa_i + kappa_e")
        self.set_param_hint("Q_i", expr="fr / kappa_i")
        self.set_param_hint("Q_e", expr="fr / kappa_e")
        self.set_param_hint("Q", expr="fr / kappa")


class HangerResonatorModel(MobiusModel):
    """A model for fitting a two port hanger resonator response."""

    def __init__(
        self,
        prefix="",
        nan_policy="raise",
        **kwargs,
    ):
        super().__init__(
            prefix=prefix,
            nan_policy=nan_policy,
            **kwargs,
        )

        self.set_param_hint("fr", expr="(f0_r + fp_r) / 2")
        self.set_param_hint("kappa_i", min=1e-12, expr="2*f0_i")
        self.set_param_hint("kappa_e", min=1e-12, expr="2*(fp_i - f0_i)")
        self.set_param_hint("kappa", min=1e-12, expr="kappa_i + kappa_e")
        self.set_param_hint("Q_i", expr="fr / kappa_i")
        self.set_param_hint("Q_e", expr="fr / kappa_e")
        self.set_param_hint("Q", expr="fr / kappa")
