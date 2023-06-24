import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

import qwip
from qwip.backends.qutip import QutipBackend
from qwip.processing.processors import (
    ClassifiedResult,
    FormatLegacyIQ,
    GMMClassification,
    HeterodyneDemodulation,
    HistogramResult,
    IQResult,
    IQRotation,
    IQTraceResult,
    PopulationResult,
    ReadoutBitstring,
    ReadoutHistogram,
    StatePopulations,
)
from qwip.sequencer import ReadoutMarker, Sequence, SequenceElement


class TestIQTraceResult:
    def test_create_from_numpy(self):
        IQ_data = np.ones((10, 2, 512, 1000))
        IQ_default = IQTraceResult.from_numpy(name="Q0", IQ_raw=IQ_data)
        assert IQ_default.data.to_numpy().shape == (10 * 2 * 512, 1000)
        assert IQ_default.data.index.names == ["element", "readout", "shot"]

        IQ_data_no_readout = np.ones((10, 512, 1000))
        IQ_no_readout = IQTraceResult.from_numpy(
            name="Q1", IQ_raw=IQ_data_no_readout, labels=["element", "shot"]
        )
        assert IQ_no_readout.data.to_numpy().shape == (10 * 512, 1000)
        assert IQ_no_readout.data.index.names == ["element", "shot"]


class TestHeterodyneDemodulation:
    @pytest.fixture(scope="function")
    def signal_generator(self, seed):
        rng = default_rng(seed=seed)

        def generate(IQ, freqs, ts, shape):
            IQ = IQ.reshape(-1, 1)
            freqs = freqs.reshape(-1, 1)

            V_t = np.sum(IQ * np.exp(1j * 2 * np.pi * freqs * ts), axis=0)

            noise = rng.random((*shape, ts.shape[0], 2)).view(np.complex128)
            noise = noise.reshape(*noise.shape[:-1])

            return V_t + noise

        return generate

    @pytest.mark.parametrize(
        "IQ,freq",
        [
            (10 + 5j, 0.5),
            (10 - 5j, 0.3),
        ],
    )
    def test_single(self, IQ, freq, signal_generator):
        ts = np.arange(4096) / 1.8
        freqs = np.array([freq])
        IQ = np.array([IQ])

        shape = (20, 1, 1000)

        data = signal_generator(IQ, freqs, ts, shape)

        res = IQTraceResult.from_numpy("raw", data)

        weight = np.exp(-1j * 2 * np.pi * freqs[0] * ts)

        processor = HeterodyneDemodulation(weights={"Q0": weight})
        processed = processor(res)

        demod_IQ = processed["Q0"].data.mean(axis=1).mean()

        assert_allclose(IQ[0], demod_IQ, atol=1e-3)

    @pytest.mark.parametrize(
        "IQ,freqs",
        [
            (np.array([10 + 5j, 10 - 5j]), np.array([0.5, 0.6])),
        ],
    )
    def test_multiplexed(self, IQ, freqs, signal_generator):
        ts = np.arange(4096) / 1.8

        shape = (20, 1, 1000)

        data = signal_generator(IQ, freqs, ts, shape)

        res = IQTraceResult.from_numpy("raw", data)

        weights = {
            f"Q{i}": np.exp(-1j * 2 * np.pi * freqs[i] * ts) for i in range(len(freqs))
        }

        processor = HeterodyneDemodulation(weights=weights)
        processed = processor(res)

        demod_IQ = np.array(
            [res.data.mean(axis=1).mean() for res in processed.values()]
        )

        assert_allclose(IQ, demod_IQ, atol=5e-2)


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

    def test_unstructure(self):
        meas = np.arange(2 * 3 * 4 * 5).astype(float).reshape(2, 3, 4, 5)
        iqdata = FormatLegacyIQ()(meas)

        unstructured = qwip.converter.unstructure(iqdata)
        df = iqdata.data
        structured = qwip.converter.structure(unstructured | dict(data=df), IQResult)

        assert unstructured == dict(
            name="IQResult",
            processors=[dict(measurement_key=None, __class__="FormatLegacyIQ")],
            __class__="IQResult"
        )
        assert structured == iqdata

class TestIQRotation:
    def test_rotate(self):
        angles = np.exp(1j * np.arange(8) * np.pi / 4)
        meas = IQResult(name="R0", data=pd.DataFrame(angles.reshape(4, 2)))

        res = IQRotation(angle=0)(meas)
        assert_array_almost_equal(res.data.to_numpy(), angles.reshape(4, 2))

        res = IQRotation(angle=np.pi / 4)(meas)
        assert_array_almost_equal(
            res.data.to_numpy(), np.roll(angles, -1).reshape(4, 2)
        )

        res = IQRotation(angle=-np.pi / 2)(meas)
        assert_array_almost_equal(res.data.to_numpy(), np.roll(angles, 2).reshape(4, 2))


class TestGMMClassification:
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((2, 3, 4), np.float64),
            ((5,), np.float32),
        ],
    )
    def test_get_real_IQ_from_domplex(self, shape, dtype, seed):
        rng = default_rng(seed + np.product(shape))

        real = rng.random(size=shape, dtype=dtype)
        imag = rng.random(size=shape, dtype=dtype)

        IQ_data = real + 1j * imag

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
            (np.array([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=float), (1, 50)),
        ],
    )
    def test_classification_qubit(self, means, shape, seed):
        rng = default_rng(seed)

        N = means.shape[0]

        gmm = GMMClassification(
            num_states=N, means=means, covariances=0.2 * np.ones(means.shape[0])
        )

        expect = rng.choice(N, size=shape)
        iqdata = gmm.means[expect].view(np.complex128)[..., 0]

        iq = IQResult(name="R0", data=pd.DataFrame(iqdata))

        classified = gmm(iq)

        assert_array_equal(classified.data.to_numpy(), np.char.mod("%d", expect))
