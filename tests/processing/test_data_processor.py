from collections.abc import Collection

import numpy as np
import pandas as pd
import pytest
import rustworkx as rx

import qwip
from qwip.processing.data_processor import (
    DATA_PROCESSORS,
    GRAPH_IN,
    GRAPH_OUT,
    DataProcessorGraph,
    MeasurementResult,
    ReadoutPipeline,
)
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


class TestMeasurementResult:
    def test_equal(self):
        x = pd.DataFrame([1, 2, 3])
        y = pd.DataFrame([1, 2, 3])

        assert MeasurementResult(name="name", data=x) == MeasurementResult(
            name="name", data=y
        )

    def test_not_equal(self):
        x = pd.DataFrame([1, 2, 3])
        y = pd.DataFrame([1, 2, 3])
        z = pd.DataFrame([3, 2, 1])

        assert MeasurementResult(name="name", data=x) != MeasurementResult(
            name="name", data=z
        )
        assert MeasurementResult(name="name", data=x) != MeasurementResult(
            name="NAME", data=y
        )

    def test_repr(self):
        res = MeasurementResult(name="name", data=pd.DataFrame([1, 2, 3, 4]))

        assert (
            repr(res)
            == "MeasurementResult(name='name', data=DataFrame [4 rows x 1 columns], "
            "processors=())"
        )

    def test_unstructure(self):
        res = MeasurementResult(name="name", data=pd.DataFrame([1, 2, 3, 4]))

        assert qwip.converter.unstructure(res) == dict(name="name", processors=[])


class TestDataProcessor:
    def test_unstructure(self):
        IQ = FormatLegacyIQ()
        rot = IQRotation(measurement_key="R0", angle=90)
        gmm = GMMClassification(measurement_key="R0")
        print(qwip.converter.unstructure(gmm))
        gmmd = qwip.converter.unstructure(gmm)
        print(type(gmmd['means']))

class TestProcessingGraph:
    def test_init(self):
        assert DATA_PROCESSORS.measurement_results() == {
            np.ndarray,
            IQResult,
            ClassifiedResult,
            HistogramResult,
            PopulationResult,
        }

        assert DATA_PROCESSORS.data_processors() == {
            FormatLegacyIQ,
            IQRotation,
            GMMClassification,
            ReadoutBitstring,
            ReadoutHistogram,
            StatePopulations,
        }

    @pytest.mark.parametrize(
        "processor,in_type,out_type",
        [
            (FormatLegacyIQ, np.ndarray, IQResult),
            (IQRotation, IQResult, IQResult),
            (GMMClassification, IQResult, ClassifiedResult),
            (ReadoutBitstring, Collection[ClassifiedResult], ClassifiedResult),
            (ReadoutHistogram, ClassifiedResult, HistogramResult),
            (StatePopulations, HistogramResult, PopulationResult),
        ],
    )
    def test_get_input_type(self, processor, in_type, out_type):
        assert DATA_PROCESSORS.get_input_type(processor) == in_type
        assert DATA_PROCESSORS.get_result_type(processor) == out_type

    @pytest.mark.parametrize(
        "processor,replace,expect",
        [
            (FormatLegacyIQ, False, (np.ndarray, IQResult)),
            (GMMClassification, False, (IQResult, ClassifiedResult)),
            (ReadoutBitstring, False, (Collection[ClassifiedResult], ClassifiedResult)),
            (ReadoutBitstring, True, (ClassifiedResult, ClassifiedResult)),
        ],
    )
    def test_get_types_from_signature(self, processor, replace, expect):
        assert (
            DataProcessorGraph.get_types_from_signature(
                processor, replace_generic=replace
            )
            == expect
        )

    def test_get_dependencies(self):
        deps = DATA_PROCESSORS.get_dependencies(StatePopulations, input_type=np.ndarray)
        assert deps == [
            FormatLegacyIQ,
            IQRotation,
            GMMClassification,
            ReadoutBitstring,
            ReadoutHistogram,
            StatePopulations,
        ]

        deps = DATA_PROCESSORS.get_dependencies(GMMClassification, input_type=IQResult)
        assert deps == [GMMClassification]

        with pytest.raises(ValueError):
            deps = DATA_PROCESSORS.get_dependencies(
                GMMClassification, input_type=ClassifiedResult
            )


class TestPipeline:
    @pytest.fixture
    def single_qubit(self):
        processors = [
            FormatLegacyIQ(),
            IQRotation(measurement_key="R0", angle=np.pi / 2),
            GMMClassification(
                measurement_key=None, means=np.zeros((2, 2)), covariances=np.zeros(2)
            ),
            GMMClassification(
                measurement_key="R0",
                means=np.array([[0, 1], [1, 0]]),
                covariances=np.array([0.2, 0.2]),
            ),
            GMMClassification(
                measurement_key="R1",
                means=np.array([[1, 0], [-1, 0]]),
                covariances=np.array([0.2, 0.2]),
            ),
            ReadoutHistogram(),
            StatePopulations(),
        ]
        return processors

    def test_get_processor(self, single_qubit):
        processors = single_qubit
        pipeline = ReadoutPipeline(processors=processors)

        assert pipeline.get_processor("GMMClassification", "R0") == processors[3]
        assert pipeline.get_processor("StatePopulations", "R1") == processors[6]
        assert pipeline.get_processor("IQRotation", "R1") is None

    def test_get_processor_cache(self, single_qubit):
        processors = single_qubit
        pipeline = ReadoutPipeline(processors=processors)

        pipeline.get_processor.cache_clear()

        assert pipeline.get_processor("FormatLegacyIQ", "R0") == processors[0]
        assert pipeline.get_processor("FormatLegacyIQ", "R0") == processors[0]
        assert pipeline.get_processor("FormatLegacyIQ", "R0") == processors[0]

        cinfo = pipeline.get_processor.cache_info()
        assert cinfo.hits == 2
        assert cinfo.misses == 1

    def test_add_processor(self, single_qubit):
        pipeline = ReadoutPipeline(processors=single_qubit)

        bitstring_proc = ReadoutBitstring()

        assert pipeline.get_processor("ReadoutBitstring", "R0") is None
        pipeline.add_processor(bitstring_proc)
        assert pipeline.get_processor("ReadoutBitstring", "R0") == bitstring_proc

    def test_remove_processor(self, single_qubit):
        pipeline = ReadoutPipeline(processors=single_qubit)

        p = pipeline.remove_processor(StatePopulations)
        assert p not in pipeline.processors

        with pytest.raises(KeyError):
            pipeline.remove_processor(ReadoutHistogram, "R3")

    def test_result_types(self, single_qubit):
        pipeline = ReadoutPipeline(processors=single_qubit)

        assert pipeline.result_types() == {
            IQResult,
            ClassifiedResult,
            HistogramResult,
            PopulationResult,
        }

    def test_dependency_resolution(self, single_qubit):
        pipeline = ReadoutPipeline(processors=single_qubit)
        pipeline.add_processor(ReadoutBitstring())

        resolved = pipeline.resolve_dependencies(
            {"R0": GMMClassification, "R1": ReadoutHistogram, "R0,R1": StatePopulations}
        )

        order = [(k, type(p).__name__) for k, p, pred in resolved]

        assert order.index(("R0", "FormatLegacyIQ")) < order.index(
            ("R0", "GMMClassification")
        )
        assert order.index(("R1", "FormatLegacyIQ")) < order.index(
            ("R1", "GMMClassification")
        )
        assert order.index(("R1", "GMMClassification")) < order.index(
            ("R1", "ReadoutHistogram")
        )
        assert order.index(("R0", "GMMClassification")) < order.index(
            ("R0,R1", "ReadoutBitstring")
        )
        assert order.index(("R1", "GMMClassification")) < order.index(
            ("R0,R1", "ReadoutBitstring")
        )
        assert order.index(("R0,R1", "ReadoutBitstring")) < order.index(
            ("R0,R1", "ReadoutHistogram")
        )
        assert order.index(("R0,R1", "ReadoutHistogram")) < order.index(
            ("R0,R1", "StatePopulations")
        )
