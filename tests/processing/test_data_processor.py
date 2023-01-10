import pytest

import pandas as pd
import numpy as np

from qwip.processing.data_processor import (
    MeasurementResult,
    ReadoutPipeline,
    resolve_class_dependencies,
    DATA_PROCESSOR_LOOKUP
)
from qwip.processing.processors import *

class TestMeasurementResult:
    def test_equal(self):
        x = pd.DataFrame([1, 2, 3])
        y = pd.DataFrame([1, 2, 3])

        assert (
            MeasurementResult(name='name', data=x) == 
            MeasurementResult(name='name', data=y)
        )

    def test_repr(self):
        res = MeasurementResult(
            name='name',
            data=pd.DataFrame([1, 2, 3, 4])
        )
    


class TestPipeline:

    def test_dependency_resolution(self):
        from collections import defaultdict

        n1 = FormatLegacyIQ()
        n2_R1 = IQRotation(angle=np.pi/2)
        n3_R0 = GMMClassification(
            means=np.array([[0, 1], [1, 0]], dtype=float),
            covariances=np.array([0.2, 0.2])
        )
        n3_R1 = GMMClassification(
            means=np.array([[-1, 0], [-1, -2]], dtype=float),
            covariances=np.array([0.2, 0.2])
        )
        n4 = ReadoutBitstring()

        n5 = ReadoutHistogram()
        n6 = StatePopulations()

        processes = [
            (None, n1),
            ('R1', n2_R1),
            ('R0', n3_R0),
            ('R1', n3_R1),
            (None, n4)
            (None, n5),
            (None, n6)
        ]

        for cls, dependencies in DATA_PROCESSOR_LOOKUP.items():
            print(cls)
            print(dependencies)

        print(resolve_class_dependencies('StatePopulations'))

        pipeline = ReadoutPipeline(
            processors={(k, type(p).__name__): p for k, p in processes}
        )

        key = 'R0,R1'
        processor = 'ReadoutBitstring'

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

