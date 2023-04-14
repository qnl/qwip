import pytest
import numpy as np
import pandas as pd
from numpy.random import default_rng
from numpy.testing import assert_array_equal, assert_array_almost_equal

from qwip.processing.processors import (
    ClassifiedResult,
    FormatLegacyIQ,
    GMMClassification,
    HistogramResult,
    IQResult,
    IQRotation,
    PopulationResult,
    ReadoutBitstring,
    ReadoutHistogram,
    StatePopulations,
)

class TestFormatLegacyIQ:
    def test_reorder(self):
        meas = np.arange(2 * 3 * 4 * 5).astype(float).reshape(2, 3, 4, 5)

        iqdata = FormatLegacyIQ()(meas)

        assert iqdata.shape == (4 * 5, 3)
        assert iqdata.data.index.levshape == (4, 5)

    def test_float32(self):
        meas = np.arange(2 * 3 * 4 * 5).astype(np.float32).reshape(2, 3, 4, 5)

        iqdata = FormatLegacyIQ()(meas)

        assert iqdata.shape == (4 * 5, 3)
        assert iqdata.data.index.levshape == (4, 5)


class TestIQRotation:
    def test_rotate(self):
        angles = np.exp(1j*np.arange(8) * np.pi / 4)
        meas = IQResult(
            name="R0",
            data=pd.DataFrame(angles.reshape(4, 2))
        )

        res = IQRotation(angle=0)(meas)
        assert_array_almost_equal(res.data.to_numpy(), angles.reshape(4, 2))

        res = IQRotation(angle=np.pi/4)(meas)
        assert_array_almost_equal(res.data.to_numpy(), np.roll(angles, -1).reshape(4, 2))
        
        res = IQRotation(angle=-np.pi/2)(meas)
        assert_array_almost_equal(res.data.to_numpy(), np.roll(angles, 2).reshape(4, 2))


class TestGMMClassification:
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((2, 3, 4), np.float64),
            ((5,), np.float32),
        ]
    )
    def test_get_real_IQ_from_domplex(self, shape, dtype, seed):
        rng = default_rng(seed + np.product(shape))

        real = rng.random(size=shape, dtype=dtype)
        imag = rng.random(size=shape, dtype=dtype)

        IQ_data = real + 1j*imag

        result = GMMClassification._get_real_IQ_from_complex(IQ_data)

        assert result.shape == (*shape, 2)
        assert_array_equal(result[..., 0], real)
        assert_array_equal(result[..., 1], imag)


    @pytest.mark.parametrize(
        "means,shape",
        [
            (np.array([[-1, 0], [1, 0]], dtype=float), (10 * 2, 5)),
            (np.array([[-1, 0], [1, 0]], dtype=float), (1, 1)),
            (np.array([[-1, 0], [1, 0], [0, 1]], dtype=float), (20, 1)),
            (np.array([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=float), (1, 50))
        ]
    )
    def test_classification_qubit(self, means, shape, seed):
        rng = default_rng(seed)

        N = means.shape[0]

        gmm = GMMClassification(
            num_states=N,
            means=means,
            covariances=0.2*np.ones(means.shape[0])
        )

        expect = rng.choice(N, size=shape)
        iqdata = gmm.means[expect].view(np.complex128)[..., 0]

        iq = IQResult(
            name="R0",
            data=pd.DataFrame(iqdata)
        )

        classified = gmm(iq)

        assert_array_equal(classified.data.to_numpy(), np.char.mod("%d", expect))