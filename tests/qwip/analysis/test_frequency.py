import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from qwip.analysis.frequency import get_frequency_phase, simple_fft


@pytest.mark.parametrize(
    "ys,axis",
    [
        (np.zeros(21), -1),
        (np.zeros(10), -1),
        (np.zeros((15, 5)), 1),
        (np.zeros((5, 21), dtype=np.complex128), -1),
    ],
)
def test_simple_fft(ys, axis):
    """Tests that simple_fft works on a variety of argument types.

    We can assume that scipy tests their fft implementation so we will not check
    correctness here.
    """
    ts = np.linspace(0, 1, ys.shape[axis])
    fs, yfs = simple_fft(ts, ys)

    assert fs.shape[0] == yfs.shape[axis]
    assert yfs.shape == ys.shape
    # Check fftshift
    assert_array_equal(fs, np.sort(fs))


@pytest.mark.parametrize(
    "vals,sgn,expect",
    [
        ([(-1, 1), (1, 0.5)], 1, 1),
        ([(-0.2, -1)], -1, -0.2),
    ],
)
def test_get_frequency_phase(vals, sgn, expect):
    fs = np.round(np.arange(-50, 51, 1) / 50, 3)
    yfs = np.zeros_like(fs)

    for f, v in vals:
        yfs[fs == f] = v

    assert get_frequency_phase(fs, yfs, sgn)[0] == expect
