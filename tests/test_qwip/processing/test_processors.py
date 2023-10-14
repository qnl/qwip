import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

import qwip
from qwip.backends.qutip import QutipBackend
from qwip.backends.software import HeterodyneProgram
from qwip.processing.data_processor import DataProcessor
from qwip.processing.processors import (
    Averaged,
    ClassifiedResult,
    GMMClassification,
    HeterodyneDemodulation,
    IQResult,
    IQRotation,
    IQTraceResult,
    Labeled,
    ReadoutBitstring,
    ReadoutHistogram,
    StatePopulations,
    array_complex_to_real,
    dataframe_complex_to_real,
    dataframe_real_to_complex,
)
from qwip.sequencer import Sequence
from qwip.sequencer.compilation import QuantumExecutable, QWiPExecutable


@pytest.mark.parametrize(
    "shape,dtype,order",
    [
        ((2, 3, 4), np.float64, "C"),
        ((5,), np.float32, "C"),
        ((5, 1, 3), np.float64, "F"),
    ],
)
def test_array_complex_to_real(shape, dtype, order, seed):
    rng = default_rng(seed)

    real = rng.random(size=shape, dtype=dtype)
    imag = rng.random(size=shape, dtype=dtype)

    IQ_data = np.array(real + 1j * imag, order=order)

    result = array_complex_to_real(IQ_data)

    assert result.shape == (*shape, 2)
    assert_array_equal(result[..., 0], real)
    assert_array_equal(result[..., 1], imag)


def test_dataframe_complex_to_real(seed):
    shape = (10, 128, 2)
    rng = default_rng(seed)

    arr = rng.random(size=shape) + 1j * rng.random(size=shape)
    data = IQResult.from_numpy(arr).data

    real_data = dataframe_complex_to_real(data)

    assert real_data.index.equals(data.index)
    assert real_data.columns.equals(
        pd.MultiIndex.from_tuples([("IQ", "real"), ("IQ", "imag")])
    )
    assert_array_equal(
        real_data["IQ", "real"].to_numpy().flatten(), np.real(data).flatten()
    )
    assert_array_equal(
        real_data["IQ", "imag"].to_numpy().flatten(), np.imag(data).flatten()
    )
    assert dataframe_real_to_complex(real_data).equals(data)


def test_dataframe_complex_to_real_multicolumn(seed):
    shape = (10, 128, 2)
    rng = default_rng(seed)

    arr = rng.random(size=shape) + 1j * rng.random(size=shape)
    data = IQResult.from_numpy(arr).data
    data["IQ_conj"] = np.conj(data)

    real_data = dataframe_complex_to_real(data, names=("I", "Q"))

    assert real_data.index.equals(data.index)
    assert real_data.columns.equals(
        pd.MultiIndex.from_tuples(
            [("IQ", "I"), ("IQ", "Q"), ("IQ_conj", "I"), ("IQ_conj", "Q")]
        )
    )

    for col in data.columns:
        assert_array_equal(
            real_data[col, "I"].to_numpy().flatten(), np.real(data[col]).flatten()
        )
        assert_array_equal(
            real_data[col, "Q"].to_numpy().flatten(), np.imag(data[col]).flatten()
        )

    assert dataframe_real_to_complex(real_data).equals(data)


def test_dataframe_complex_to_real_multiindex(seed):
    shape = (10, 128, 2)
    rng = default_rng(seed)

    arr = rng.random(size=shape) + 1j * rng.random(size=shape)
    data = IQResult.from_numpy(arr).data
    data["IQ_conj"] = np.conj(data)
    data[["rotated", "rotated_conj"]] = data * np.exp(1j * np.pi / 4)
    data.columns = pd.MultiIndex.from_tuples(
        it.product(["orig", "rotated"], ["IQ", "IQ_conj"])
    )

    real_data = dataframe_complex_to_real(data, names=("I", "Q"))

    assert real_data.index.equals(data.index)
    assert real_data.columns.equals(
        pd.MultiIndex.from_tuples(
            it.product(["orig", "rotated"], ["IQ", "IQ_conj"], ["I", "Q"])
        )
    )

    for col in data.columns:
        assert_array_equal(
            real_data[(*col, "I")].to_numpy().flatten(), np.real(data[col]).flatten()
        )
        assert_array_equal(
            real_data[(*col, "Q")].to_numpy().flatten(), np.imag(data[col]).flatten()
        )

    assert dataframe_real_to_complex(real_data).equals(data)


class TestIQTraceResult:
    def test_from_numpy(self):
        IQ_data = np.ones((512, 10, 2, 1000))
        IQ_default = IQTraceResult.from_numpy(IQ_data, name="Q0")
        assert IQ_default.data.to_numpy().shape == (10 * 2 * 512 * 1000, 1)
        assert IQ_default.data.index.names == ["shot", "element", "readout", "time"]

        IQ_data_no_readout = np.ones((512, 10, 1000))
        IQ_no_readout = IQTraceResult.from_numpy(
            IQ_data_no_readout, name="Q1", labels=["shot", "element", "time"]
        )
        assert IQ_no_readout.data.to_numpy().shape == (512 * 10 * 1000, 1)
        assert IQ_no_readout.data.index.names == ["shot", "element", "time"]


class TestHeterodyneDemodulation:
    @pytest.fixture(scope="class")
    def signal_generator(self, seed):
        rng = default_rng(seed=seed)

        def generate(IQ, freqs, ts, shape):
            IQ = IQ.reshape(-1, 1)
            freqs = freqs.reshape(-1, 1)

            V_t = np.sum(IQ * np.exp(1j * 2 * np.pi * freqs * ts), axis=0)

            noise = rng.random((*shape, ts.shape[0], 2)).view(np.complex128)
            noise = noise.reshape(*noise.shape[:-1])

            data = (V_t + noise).astype(np.complex64)
            return IQTraceResult.from_numpy(data)

        return generate

    @pytest.mark.parametrize(
        "IQ,signal_freq,demod_freq,expect",
        [
            ([10 + 5j], [500e6], [500e6], [10 + 5j]),
            ([10 - 5j], [300e6], [-300e6], [0]),
            ([8 + 4j, 20 + 3j], [250e6, 300e6], [250e6, 300e6], [8 + 4j, 20 + 3j]),
            ([10j, 20], [100e6, 150e6], [100e6, 150e6, 250e6], [10j, 20, 0]),
        ],
    )
    def test_single_slice(self, IQ, signal_freq, demod_freq, expect, signal_generator):
        ts = np.arange(1024) / 1.8e9
        freqs = np.array(signal_freq)
        demod_freq = np.array(demod_freq).reshape(-1, 1)
        IQ = np.array(IQ)

        shape = (100, 20, 1)
        result = signal_generator(IQ, freqs, ts, shape)

        weight = np.exp(2 * np.pi * 1j * demod_freq * ts).astype(np.complex64)

        demods = np.zeros(20, dtype=int)
        keys = [f"D{i}" for i in range(demod_freq.shape[0])]

        program = HeterodyneProgram(
            device="demod", demods=demods, weights=(weight,), keys=keys
        )
        exe = QWiPExecutable(num_reads=[1] * 20, programs=dict(demod=program))
        processor = HeterodyneDemodulation(device="demod")
        processed = processor(result, exe=exe)
        demod_IQ = [p.data.values.mean() for p in processed]

        assert_allclose(demod_IQ, expect, atol=0.05, rtol=0.05)

    @pytest.mark.parametrize(
        "IQ,signal_freq,demod_freq,expect",
        [
            ([10 + 5j], [500e6], [500e6], [10 + 5j]),
            ([10 - 5j], [300e6], [-300e6], [0]),
            ([8 + 4j, 20 + 3j], [250e6, 300e6], [250e6, 300e6], [8 + 4j, 20 + 3j]),
            ([10j, 20], [100e6, 150e6], [100e6, 150e6, 400e6], [10j, 20, 0]),
        ],
    )
    def test_multi_slice(self, IQ, signal_freq, demod_freq, expect, signal_generator):
        ts = np.arange(1024) / 1.0e9
        freqs = np.array(signal_freq)
        IQ = np.array(IQ)

        shape = (512, len(demod_freq), 1)
        result = signal_generator(IQ, freqs, ts, shape)

        weights = [
            np.exp(2*np.pi * 1j * f * ts).reshape(1, -1) for f in demod_freq
        ]
        demods = np.r_[:len(demod_freq)]
        keys = ["D0"]

        program = HeterodyneProgram(device="demod", demods=demods, weights=weights, keys=keys)
        exe = QWiPExecutable(num_reads=[1]*len(freqs), programs=dict(demod=program))
        processor = HeterodyneDemodulation(device="demod")
        processed = processor(result, exe=exe)[0]
        demod_IQ = processed.groupby(["element", "readout"]).mean().to_numpy().flatten()

        assert_allclose(demod_IQ, expect, atol=0.05, rtol=0.05)


class TestIQResult:
    def test_unstructure(self, fixed_time):
        arr = np.arange(2 * 3 * 4 * 5, dtype=np.float32).view(np.complex64)
        iqdata = IQResult.from_numpy(arr.reshape(3, 4, 5))

        unstructured = qwip.converter.unstructure(iqdata)
        df = iqdata.data
        structured = qwip.converter.structure(unstructured | dict(data=df), IQResult)

        assert unstructured == dict(
            name="IQResult",
            timestamp=fixed_time.isoformat(),
            processors=[],
            __class__="IQResult",
        )
        assert structured == iqdata

    def test_amplitude(self, seed):
        rng = default_rng(seed=seed)
        result = IQResult.random(shape=(100, 200, 2), num_states=1, rng=rng)

        mag = result.amplitude()
        log_mag = result.amplitude(log=True)

        assert_array_almost_equal(mag, np.sqrt(10 ** (log_mag / 10)))
        assert np.all(mag.index == result.data.index)
        assert list(mag.columns) == ["amplitude"]

    def test_phase(self, seed):
        rng = default_rng(seed=seed)
        result = IQResult.random(shape=(100, 200, 2), num_states=1, rng=rng)

        phase = result.phase()

        assert_array_almost_equal(phase, np.angle(result.data))
        assert list(phase.columns) == ["phase"]


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

    def test_unstructure(self):
        processor = IQRotation(angle=np.pi / 4, measurement_key="R0")
        unstructured = qwip.converter.unstructure(processor)

        assert unstructured == dict(
            measurement_key="R0", angle=np.pi / 4, __class__="IQRotation"
        )

        structured = qwip.converter.structure(unstructured, DataProcessor)
        assert processor == structured


class TestGMMClassification:
    @pytest.mark.parametrize(
        "means,shape",
        [
            (np.array([[-1, 0], [1, 0]], dtype=float), (10, 5, 2)),
            (np.array([[-1, 0], [1, 0]], dtype=float), (1, 1, 1)),
            (np.array([[-1, 0], [1, 0], [0, 1]], dtype=float), (4, 1, 5)),
            (np.array([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=float), (1, 5, 10)),
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

        iq = IQResult.from_numpy(name="R0", arr=iqdata)

        classified = gmm(iq)

        assert_array_equal(
            classified.data.to_numpy(), np.char.mod("%d", expect.reshape(-1, 1))
        )


class TestReadoutHistogram:
    @pytest.fixture
    def rng(self, seed):
        return default_rng(seed)

    def test_single_qubit(self, rng):
        shape = (1024, 10, 2)
        result = ClassifiedResult.from_numpy(
            rng.choice(2, size=np.prod(shape)).reshape(shape),
            name="R0",
        )

        counts = ReadoutHistogram()(result)

        assert counts.columns.equals(pd.Index(["0", "1"], name="state"))
        assert counts.shape == (shape[1] * shape[2], 2)
        assert counts.index.equals(
            pd.MultiIndex.from_tuples(
                it.product(*(range(d) for d in (shape[1], shape[2])))
            )
        )

    @pytest.fixture
    def multi_qubit_result(self, rng):
        shape = (512, 5, 1)
        result = ReadoutBitstring()(
            [
                ClassifiedResult.from_numpy(
                    rng.choice(3, size=np.prod(shape)).reshape(shape),
                    name="R0",
                    num_states=3,
                ),
                ClassifiedResult.from_numpy(
                    rng.choice(2, size=np.prod(shape)).reshape(shape), name="R1"
                ),
            ]
        )

        return result

    def test_multi_qubit(self, multi_qubit_result):
        result = multi_qubit_result
        shape = result.index.levshape

        counts = ReadoutHistogram()(result)
        assert counts.columns.equals(
            pd.Index(
                ["".join(map(str, s)) for s in it.product(range(3), range(2))],
                name="state",
            )
        )
        assert counts.shape == (shape[1] * shape[2], 6)
        assert counts.index.equals(
            pd.MultiIndex.from_tuples(
                it.product(*(range(d) for d in (shape[1], shape[2])))
            )
        )

    def test_fill_missing(self, multi_qubit_result):
        result = multi_qubit_result
        shape = result.index.levshape

        counts = ReadoutHistogram()(result, fill_missing=True)
        assert counts.columns.equals(
            pd.Index(
                ["".join(map(str, s)) for s in it.product(range(3), repeat=2)],
                name="state",
            )
        )
        assert counts.shape == (shape[1] * shape[2], 9)
        assert counts.index.equals(
            pd.MultiIndex.from_tuples(
                it.product(*(range(d) for d in (shape[1], shape[2])))
            )
        )

    def test_multiindex(self, rng):
        shape = (1024, 10, 2)
        result = ClassifiedResult.from_numpy(
            rng.choice(2, size=np.prod(shape)).reshape(shape),
            name="readout",
        )

        result.data = pd.concat(
            [result.data, result.data],
            axis="columns",
            keys=["R0", "R1"],
            names=["qubit"],
        )

        counts = ReadoutHistogram(fill_missing=False)(result)

        assert counts.columns.equals(
            pd.MultiIndex.from_tuples(
                it.product(["R0", "R1"], ["0", "1"]), names=["qubit", "state"]
            )
        )
        assert counts.shape == (shape[1] * shape[2], 4)
        assert counts.index.equals(
            pd.MultiIndex.from_tuples(
                it.product(*(range(d) for d in (shape[1], shape[2])))
            )
        )


class TestStatePopulations:
    @pytest.fixture
    def rng(self, seed):
        return default_rng(seed)

    def test_single_qubit(self, rng):
        shape = (10, 1024, 2)
        result = ClassifiedResult.from_numpy(
            rng.choice(2, size=np.prod(shape)).reshape(shape),
            name="R0",
        )

        counts = ReadoutHistogram()(result)
        populations = StatePopulations()(counts)

        assert populations.data.index.equals(counts.data.index)
        assert populations.data.columns.equals(counts.data.columns)
        assert np.all(populations.sum(axis="columns") == 1)

    def test_multi_index(self, rng):
        shape = (10, 1024, 2)
        result = ClassifiedResult.from_numpy(
            rng.choice(2, size=np.prod(shape)).reshape(shape),
            name="readout",
        )

        result.data = pd.concat(
            [result.data, result.data],
            axis="columns",
            keys=["R0", "R1"],
            names=["qubit"],
        )

        counts = ReadoutHistogram()(result, fill_missing=False)
        populations = StatePopulations()(counts)

        assert populations.data.index.equals(counts.data.index)
        assert populations.data.columns.equals(counts.data.columns)
        assert np.all(populations.groupby(level=[0], axis="columns").sum() == 1)


class TestAveraged:
    def test_copy(self):
        arr = np.arange(2 * 3 * 4 * 5, dtype=np.float32).view(np.complex64)
        iqdata = IQResult.from_numpy(arr.reshape(3, 4, 5))

        avgiq = Averaged()(iqdata)

        assert avgiq is not iqdata
        assert avgiq.data is not iqdata.data


class TestLabeled:
    def test_copy(self):
        arr = np.arange(2 * 3 * 4 * 5, dtype=np.float32).view(np.complex64)
        iqdata = IQResult.from_numpy(arr.reshape(4, 3, 5))
        orig_index = iqdata.index.names

        seq = Sequence.empty(
            (2, 2), names=("prep", "measure"), prep=np.arange(2), measure=np.arange(2)
        )
        labeled = Labeled()(iqdata, exe=QuantumExecutable(sequence=seq))

        assert labeled is not iqdata
        assert labeled.data is not iqdata.data
        assert iqdata.index.names == orig_index

    def test_label(self):
        arr = np.arange(2 * 3 * 4 * 5, dtype=np.float32).view(np.complex64)
        iqdata = IQResult.from_numpy(arr.reshape(5, 4, 3))

        seq = Sequence.empty(
            (2, 2), names=("prep", "measure"), prep=np.arange(2), measure=np.arange(2)
        )
        labeled = Labeled()(iqdata, exe=QuantumExecutable(sequence=seq))

        expected = pd.MultiIndex.from_tuples(
            it.product(np.arange(5), np.arange(2), np.arange(2), np.arange(3)),
            names=["shot", "prep", "measure", "readout"],
        )

        assert labeled.data.index.equals(expected)
        assert_array_equal(labeled.data.values, iqdata.data.values)
