from collections.abc import Collection

import numpy as np
import pandas as pd
import pytest
import rustworkx as rx

from qwip.processing.data_processor import (
    DATA_PROCESSORS,
    GRAPH_IN,
    GRAPH_OUT,
    DataProcessorGraph,
    MeasurementResult,
    ReadoutPipeline,
)
from qwip.processing.processors import *


class TestMeasurementResult:
    def test_equal(self):
        x = pd.DataFrame([1, 2, 3])
        y = pd.DataFrame([1, 2, 3])

        assert MeasurementResult(name="name", data=x) == MeasurementResult(
            name="name", data=y
        )

    def test_repr(self):
        res = MeasurementResult(name="name", data=pd.DataFrame([1, 2, 3, 4]))


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


class TestPipeline:
    @pytest.fixture
    def single_qubit(self):
        processors = [
            FormatLegacyIQ(),
            IQRotation(measurement_key="R0", angle=np.pi / 2),
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

    @pytest.mark.skip
    def test_get_processor(self, single_qubit):
        processors = single_qubit
        pipeline = ReadoutPipeline(processors=processors)

        assert pipeline.get_processor("GMMClassification", "R0") == processors[2]
        assert pipeline.get_processor("StatePopulations", "R1") == processors[5]
        assert pipeline.get_processor("IQRotation", "R1") == None

    @pytest.mark.skip
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

    @pytest.mark.skip
    def test_result_types(self, single_qubit):
        pipeline = ReadoutPipeline(processors=single_qubit)

        assert pipeline.result_types() == {
            IQResult,
            ClassifiedResult,
            HistogramResult,
            PopulationResult,
        }

    def test_dependencies(self):
        ...
        # deps = resolve_class_dependencies('StatePopulations')

    @pytest.mark.skip
    def test_dependency_resolution(self):
        from collections import defaultdict

        n1 = FormatLegacyIQ()
        n2_R1 = IQRotation(angle=np.pi / 2)
        n3_R0 = GMMClassification(
            means=np.array([[0, 1], [1, 0]], dtype=float),
            covariances=np.array([0.2, 0.2]),
        )
        n3_R1 = GMMClassification(
            means=np.array([[-1, 0], [-1, -2]], dtype=float),
            covariances=np.array([0.2, 0.2]),
        )
        n4 = ReadoutBitstring()

        n5 = ReadoutHistogram()
        n6 = StatePopulations()

        processes = [
            (None, n1),
            ("R1", n2_R1),
            ("R0", n3_R0),
            ("R1", n3_R1),
            (None, n4),
            (None, n5),
            (None, n6),
        ]

        for cls, dependencies in DATA_PROCESSOR_LOOKUP.items():
            print(cls)
            print(dependencies)

        print(resolve_class_dependencies("StatePopulations"))

        pipeline = ReadoutPipeline(
            processors={(k, type(p).__name__): p for k, p in processes}
        )

        key = "R0,R1"
        processor = "ReadoutBitstring"

        data = np.zeros((2, 10, 15, 1))

        result = pipeline.process_key(key, data, processor)
        new_result = pipeline.process_key(key, result, processor)

        assert result is new_result

        print(result)
        print()

        # for cls, (indata, outdata) in DATA_PROCESSOR_LOOKUP.items():
        #     print(f'{cls}:', indata, outdata)
        #     LOOKUP[outdata].append((indata, cls))

        # print(LOOKUP)

        # key = 'R1'
        # result_type = 'PopulationResult'

        # process_chain = []

        # while result_type in LOOKUP:
        #     LOOKUP[result_type]
