"""Fitting routines and helpers.

All models should have guess functions implemented.
"""
from collections.abc import Callable

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

    def fit(self, data: np.ndarray, params: Parameters = None, **kwargs):
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

    def guess(self, data, t):
        """Determines initial fit parameters.

        The frequency is estimated via an fft.

        Returns:
            An initialized `Parameters` dictionary.
        """
        import matplotlib.pyplot as plt

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
