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

        assert qwip.converter.unstructure(res) == dict(
            name="name", processors=[], __class__="MeasurementResult"
        )


class TestDataProcessor:
    def test_unstructure(self):
        ...
        # IQ = FormatLegacyIQ()
        # rot = IQRotation(measurement_key="R0", angle=90)
        # gmm = GMMClassification(measurement_key="R0")
        # print(qwip.converter.unstructure(gmm))
        # gmmd = qwip.converter.unstructure(gmm)
        # print(type(gmmd['means']))


class TestProcessingGraph:
    def test_init(self):
        assert DATA_PROCESSORS.measurement_results() == {
            IQTraceResult,
            IQResult,
            ClassifiedResult,
            HistogramResult,
            PopulationResult,
        }

        assert DATA_PROCESSORS.data_processors() == {
            HeterodyneDemodulation,
            IQRotation,
            GMMClassification,
            ReadoutBitstring,
            ReadoutHistogram,
            StatePopulations,
        }

    @pytest.mark.parametrize(
        "processor,in_type,out_type",
        [
            (HeterodyneDemodulation, IQTraceResult, list[IQResult]),
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

    def test_get_root_node(self):
        assert DATA_PROCESSORS.get_root_node() == IQTraceResult

    def test_get_dependencies(self):
        deps = DATA_PROCESSORS.get_dependencies(StatePopulations)
        assert deps == [
            HeterodyneDemodulation,
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
            HeterodyneDemodulation(
                measurement_key="ADC", weights=dict(R0=np.zeros(10), R1=np.zeros(20))
            ),
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

    @pytest.mark.parametrize(
        "processor,input,output,expect",
        [
            ("GMMClassification", "R0", None, 3),
            ("StatePopulations", "R1", None, 6),
            ("IQRotation", "R1", None, None),
            ("StatePopulations", None, "R2", 6),
            ("HeterodyneDemodulation", None, "R0", 0),
        ],
    )
    def test_get_processor(self, processor, input, output, expect, single_qubit):
        processors = single_qubit
        pipeline = ReadoutPipeline(processors=processors)

        match expect:
            case int():
                expect = processors[expect]

        assert pipeline.get_processor(processor, input, output) == expect

    def test_get_processor_cache(self, single_qubit):
        processors = single_qubit
        pipeline = ReadoutPipeline(processors=processors)

        pipeline.get_processor.cache_clear()

        assert pipeline.get_processor("GMMClassification", "R0") == processors[3]
        assert pipeline.get_processor("GMMClassification", "R0") == processors[3]
        assert pipeline.get_processor("GMMClassification", "R0") == processors[3]

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

        assert order.index(("ADC", "HeterodyneDemodulation")) < order.index(
            ("R0", "GMMClassification")
        )
        assert order.index(("ADC", "HeterodyneDemodulation")) < order.index(
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

    def test_get_inputs(self, cache, single_qubit):
        pipeline = ReadoutPipeline(processors=single_qubit)

        res = IQResult.from_numpy(np.zeros((4, 3, 2), dtype=complex))
        pipeline.dependency_cache.update({("R0", None): res})

        gmm = pipeline.get_processor(GMMClassification, "R0")
        bitstrings = pipeline.get_processor(ReadoutHistogram)

        assert pipeline._get_inputs("R0", None, gmm) == res
        assert pipeline._get_inputs("R0", HeterodyneDemodulation, gmm) == res
        assert pipeline._get_inputs("R0", GMMClassification, bitstrings) is None
