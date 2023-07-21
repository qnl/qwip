import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.random import default_rng

from qwip.analysis.fitting import FrequencyModel, TriangularWaveModel


class TestFrequencyModel:
    def test_override_guess(self):
        ...

    CASES_PHASE_EST = [
        (1, 4, phi, 0.5, np.linspace(0, 1), 0)
        for phi in np.linspace(-np.pi / 2, np.pi / 2, 11)
    ]

    CASES_NOISY = [
        (0.5, 2, 0, -0.5, np.linspace(0, 3, 51), 0.1),
        (0.5, 2, 0, -0.5, np.linspace(0, 3, 51), 0.2),
    ]

    @pytest.mark.parametrize("A,f,phi,B,ts,noise", CASES_PHASE_EST + CASES_NOISY)
    def test_fit(self, A, f, phi, B, ts, noise, seed):
        rng = default_rng(seed)

        ys = A * (np.sin(2 * np.pi * f * ts + phi) + rng.random(ts.shape) * noise) + B
        result = FrequencyModel().fit(ys, t=ts)

        assert result.chisqr / len(ys) < 1e-2


class TestTriangularWaveModel:
    @pytest.mark.parametrize(
        "A,period,offset,B",
        [
            (1, 2, 0, 0),
            (2, 0.25, 0, 0),
            (1, 5, -0.1, 2),
        ],
    )
    def test_fit(self, A, period, offset, B, seed):
        rng = default_rng(seed)

        ts = np.linspace(-2, 2)

        ys = TriangularWaveModel.func(ts, A=A, B=B, period=period, offset=offset)
        ys += rng.random(ts.shape) * A * 0.1

        result = TriangularWaveModel().fit(ys, t=ts)

        assert result.chisqr / len(ys) < 1e-2
