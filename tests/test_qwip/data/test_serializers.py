import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from qwip.data.serializers import (
    DataFrameSerializer,
    DefaultSerializer,
    MatplotlibSerializer,
    ResultSerializer,
    TrueQSerializer,
    detect_serializer,
    from_arrow_table,
    to_arrow_table,
)
from qwip.processing.data_processor import MeasurementResult
from qwip.processing.processors import (
    GMMClassification,
    IQResult,
    ReadoutBitstring,
    ReadoutHistogram,
    StatePopulations,
)

DEFAULT_CASES = [
    (1, int),
    (1, float),
    (True, bool),
    (dict(a=1, b=2, c=3), dict[str, int]),
    (ResultSerializer, type),
]


class TestDefaultSerializer:
    @pytest.fixture
    def serializer(self):
        return DefaultSerializer()

    @pytest.mark.parametrize("obj,cls", DEFAULT_CASES)
    def test_json(self, serializer, obj, cls):
        stream = serializer.to_stream(obj, fmt="json")
        reloaded = serializer.from_stream(stream, fmt="json", cls=cls)

        assert obj == reloaded
        assert stream.closed

    @pytest.mark.parametrize(
        "obj,name",
        [
            (1, "int"),
            (2.0, "float"),
            ([1, 2, 3], "list"),
        ],
    )
    def test_get_name(self, serializer, obj, name):
        assert serializer.get_name(obj) == name


PANDAS_ARROW_CASES = [
    (np.complex64, dict(date="2006-01-02"), b'{"date": "2006-01-02"}'),
    (np.complex128, dict(counts=20), b'{"counts": 20}'),
    (np.float32, dict(average=3.1415), b'{"average": 3.1415}'),
    (np.int64, dict(important=False), b'{"important": false}'),
    (np.float64, dict(extra=None), b'{"extra": null}'),
    (np.int32, dict(array=[1, 2, 3]), b'{"array": [1, 2, 3]}'),
    (
        np.int8,
        dict(nested=dict(counts=10, average=1.5)),
        b'{"nested": {"counts": 10, "average": 1.5}}',
    ),
]


@pytest.mark.parametrize("dtype,metadata,metadata_str", PANDAS_ARROW_CASES)
def test_to_arrow_table(dtype, metadata, metadata_str):
    data = pd.DataFrame(
        np.arange(100, dtype=dtype).reshape(20, 5),
        columns=list("ABCDE"),
        index=pd.RangeIndex(20, name="Index"),
    )

    table = to_arrow_table(metadata, data)

    assert (
        table.schema.metadata[b"qwip_complex"]
        == str(issubclass(dtype, np.complexfloating)).encode()
    )
    assert table.schema.metadata[b"qwip"] == metadata_str

    if issubclass(dtype, np.complexfloating):
        expected_names = [
            f"('{col}', '{iq}')" for col, iq in it.product(data.columns, "IQ")
        ]
    else:
        expected_names = list(data.columns)

    assert table.schema.names == expected_names


@pytest.mark.parametrize("dtype,metadata,metadata_str", PANDAS_ARROW_CASES)
def test_arrow_roundtrip(dtype, metadata, metadata_str):
    data = pd.DataFrame(
        np.arange(100, dtype=dtype).reshape(20, 5),
        columns=list("ABCDE"),
        index=pd.RangeIndex(20, name="Index"),
    )

    table = to_arrow_table(metadata, data)
    reloaded_metadata, reloaded_data = from_arrow_table(table)

    assert_frame_equal(data, reloaded_data)
    assert metadata == reloaded_metadata


class TestDataFrameSerializer:
    @pytest.fixture
    def serializer(self):
        return DataFrameSerializer()

    @pytest.mark.parametrize("dtype,metadata,metadata_str", PANDAS_ARROW_CASES)
    def test_parquet(self, serializer, dtype, metadata, metadata_str):
        data = pd.DataFrame(
            np.arange(100, dtype=dtype).reshape(20, 5),
            columns=list("ABCDE"),
            index=pd.RangeIndex(20, name="Index"),
        )

        stream = serializer.to_stream(data, fmt="parquet", metadata=metadata)
        reloaded_data, reloaded_metadata = serializer.from_stream(stream, fmt="parquet")

        assert_frame_equal(data, reloaded_data)
        assert metadata == reloaded_metadata
        assert stream.closed

    @pytest.mark.parametrize("dtype,metadata,metadata_str", PANDAS_ARROW_CASES)
    def test_feather(self, serializer, dtype, metadata, metadata_str):
        data = pd.DataFrame(
            np.arange(100, dtype=dtype).reshape(20, 5),
            columns=list("ABCDE"),
            index=pd.RangeIndex(20, name="Index"),
        )

        stream = serializer.to_stream(data, fmt="feather", metadata=metadata)
        reloaded_data, reloaded_metadata = serializer.from_stream(stream, fmt="feather")

        assert_frame_equal(data, reloaded_data)
        assert metadata == reloaded_metadata
        assert stream.closed


class TestResultSerializer:
    @pytest.fixture
    def serializer(self):
        return ResultSerializer()

    @pytest.mark.parametrize(
        "processors",
        [
            tuple(),
            (ReadoutHistogram,),
            (ReadoutBitstring, ReadoutHistogram, StatePopulations),
        ],
    )
    def test_get_result_processor(self, processors):
        result = MeasurementResult(
            name="result",
            data=pd.DataFrame(),
            processors=tuple(p() for p in processors),
        )

        expected = processors[-1] if processors else None
        assert ResultSerializer.get_result_processor(result) == expected

    def test_get_result_processor_exception(self):
        result = dict(
            R0=MeasurementResult(name="R0", data=pd.DataFrame()),
            R1=MeasurementResult(
                name="R1",
                data=pd.DataFrame(),
                processors=(ReadoutBitstring(), StatePopulations()),
            ),
        )

        with pytest.raises(ValueError):
            ResultSerializer.get_result_processor(result)

    def test_split_result(self, fixed_time, rng):
        shape = (21, 2048, 2)

        result = {
            k: IQResult.random(shape, rng=rng, name=k) for k in ("R0", "R1", "R2")
        }

        metadata, data = ResultSerializer.split_result(result)

        assert metadata == {
            k: {
                "name": k,
                "timestamp": fixed_time.isoformat(),
                "processors": [],
                "__class__": "IQResult",
            }
            for k in result
        }
        assert data.columns.equals(
            pd.MultiIndex.from_tuples(((k, "IQ") for k in result), names=["key", None])
        )

    @pytest.mark.parametrize(
        "processor,name",
        [
            (GMMClassification, "gmm-classification"),
            (None, "raw"),
            (StatePopulations, "state-populations"),
        ],
    )
    def test_name_from_processor(self, processor, name, serializer):
        assert serializer.name_from_processor(processor) == name

    @pytest.mark.parametrize(
        "name,processor",
        [
            ("gmm-classification", GMMClassification),
            ("raw", None),
            ("state-populations", StatePopulations),
            ("random", ValueError),
        ],
    )
    def test_processor_from_name(self, name, processor, serializer):
        if processor is ValueError:
            with pytest.raises(processor):
                serializer.processor_from_name(name)

        else:
            cache_hits = serializer.processor_from_name.cache_info().hits
            assert serializer.processor_from_name(name) is processor
            serializer.processor_from_name(name)
            assert serializer.processor_from_name.cache_info().hits > cache_hits


class TestMatplotlibSerializer:
    @pytest.fixture
    def serializer(self):
        return MatplotlibSerializer()

    @pytest.fixture
    def figure(self):
        fig, ax = plt.subplots()
        ts = np.linspace(0, 1)
        ax.plot(ts, np.sin(2 * np.pi * ts))
        ax.grid(True)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y-Axis")

        return fig

    @pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
    def test_round_trip(self, serializer, figure, fmt):
        stream = serializer.to_stream(figure, fmt=fmt)
        reloaded = serializer.from_stream(stream, fmt=fmt)


class TestTrueQSerializer:
    @pytest.fixture
    def serializer(self):
        return TrueQSerializer()

    @pytest.fixture(scope="class")
    def circuits(self):
        tq = pytest.importorskip("trueq")
        circuits = tq.make_srb(0, [4, 16, 64])
        sim = tq.Simulator().add_overrotation(0.01).add_depolarizing(0.01)
        sim.run(circuits)

        return circuits

    def test_circuits(self, serializer, circuits):
        stream = serializer.to_stream(circuits, fmt="tq")
        reloaded = serializer.from_stream(stream, fmt="tq")

        assert circuits == reloaded


DETECT_SERIALIZER_CASES = [
    ({"R0": IQResult.random((256, 10, 2))}, "result"),
    (pd.DataFrame(), "dataframe"),
]


@pytest.mark.parametrize("obj,expect", DETECT_SERIALIZER_CASES)
def test_detect_serializer(obj, expect):
    assert detect_serializer(obj).key == expect
