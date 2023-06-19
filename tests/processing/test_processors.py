import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

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
        ...

    def test_create_df_ts_not_equal(self):
        IQ_obj = IQTraceResult(name="Q0", data=pd.DataFrame())

        IQ_obj.create_df(np.ones((2, 50, 100)))
        assert IQ_obj.data.index.levshape == (2, 1, 50)
        assert IQ_obj.data.shape == (2 * 50, 100)


class TestHeterodyneDemodulation:
    @pytest.fixture
    def compile_Q0X90(self, qpu_01):
        db = qpu_01.db
        Q0_X = db.load_pulse("Q0_X90", variables=dict(width="rabi_width"))

        rabi_se = SequenceElement()
        rabi_se.append(Q0_X)
        rabi_se.add_waveform(ReadoutMarker(), location=Q0_X.width)
        ro_se = SequenceElement()

        ts = np.linspace(0, 34.8e-9, 21)
        seq = Sequence.sweep(rabi_se, rabi_width=ts)
        cseq = qpu_01.sequencer.compile(seq, readout=ro_se)

        return cseq

    # def test_run_empty(self):
    #     freqs = {f"Q{t}": 0 for t in range(3)}
    #     demodulator = HeterodyneDemodulation(frequencies=freqs)

    #     IQ_empty = IQTraceResult(name="Q0", data=pd.DataFrame())
    #     with pytest.raises(ValueError):
    #         demodulator.run(IQ_empty, np.arange(10))

    # def test_run_ts_not_equal(self):
    #     freqs = {f"Q{t}": 0 for t in range(3)}
    #     demodulator = HeterodyneDemodulation(frequencies=freqs)

    #     IQ_100 = IQTraceResult(name="Q3", data=pd.DataFrame())
    #     IQ_100.create_df(np.arange(2 * 50 * 100).reshape((2, 50, 100)))
    #     ts_50 = np.arange(50)

    #     with pytest.raises(ValueError):
    #         demodulator.run(IQ_100, ts_50)

    # def test_run_no_freq(self):
    #     freqs = {f"Q{t}": 0 for t in range(3)}
    #     demodulator = HeterodyneDemodulation(frequencies=freqs)

    #     IQ_wrong_key = IQTraceResult(name="Q3", data=pd.DataFrame())
    #     IQ_wrong_key.create_df(np.arange(2 * 50 * 100).reshape((2, 50, 100)))

    #     with pytest.raises(KeyError):
    #         demodulator.run(IQ_wrong_key, np.arange(100))
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

    def test_run_no_noise(self):
        f1, f2 = 0, 0
        demodulator = HeterodyneDemodulation(
            frequencies={f"Q{i}": f1 for i in range(0, 8)}
        )

        e, shot, N = 2, 3, 5
        results_test = np.arange(e * shot * N).reshape((e, shot, N))
        ts = np.arange(N)

        freq_weight = 0.5 * np.exp(-1j * 2 * np.pi * f1 * ts) + 0.5 * np.exp(
            -1j * 2 * np.pi * f2 * ts
        )

        IQ_raw = IQTraceResult(name="Q0", data=pd.DataFrame())
        IQ_raw.create_df(results_test * np.tile(freq_weight, (e, shot, 1)))
        IQ_processed = demodulator.run(IQ_raw, ts)

        assert_allclose(
            IQ_processed.data.to_numpy(),
            np.array(
                [
                    [2.0 + 0.0j, 7.0 + 0.0j, 12.0 + 0.0j],
                    [17.0 + 0.0j, 22.0 + 0.0j, 27.0 + 0.0j],
                ]
            ),
        )

    def test_run_Q0X90(self, qpu_01, compile_Q0X90):
        sim_backend = QutipBackend()
        sim_backend.update_parameters(qpu_01)
        sim_backend.upload(compile_Q0X90)

        results = sim_backend.acquire(compile_Q0X90)
        demodulator = HeterodyneDemodulation(
            frequencies={f"Q{i}": 0 for i in range(0, 8)}
        )

        IQ_results = dict()
        ts = np.linspace(0, 1e-5, 1000)

        for key in results.keys():
            IQ_raw = IQTraceResult(name=key, data=pd.DataFrame())
            IQ_raw.create_df(results[key])

            IQ_process = demodulator.run(IQ_raw, ts)
            IQ_results[key] = IQ_process

        # Should observe two blobs for "Q0" and one blob for all other qubits because
        # only a pi/2 pulse on Q0 was applied
        IQ_Q0 = IQ_results["Q0"].data.to_numpy()[0]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(np.real(IQ_Q0), np.imag(IQ_Q0))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Test: Should see two distinct blobs")
        plt.show()


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
