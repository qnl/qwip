import unittest

import numpy as np
import pandas as pd

from qwip.data.experiment_metadata import (
    make_experiment_data,
    make_measurement_data,
    run_with_recipe,
)


class ExperimentMetadataTests(unittest.TestCase):
    def test_builds_qwip_data_mapping(self):
        data = make_experiment_data(
            protocol="cryoscope",
            comments="Q1 test",
            fixed={
                "qubit": "Q1",
                "amplitude_au": np.float64(-0.2),
                "delay_s": 1e-6,
            },
            sweeps={"time_s": np.array([5e-9, 10e-9])},
        )

        self.assertEqual(data["protocol"], "cryoscope")
        self.assertEqual(data["comments"], "Q1 test")
        self.assertEqual(data["measurement_recipe"]["schema_version"], 1)
        pd.testing.assert_frame_equal(
            data["experiment_parameters"],
            pd.DataFrame(
                [{"qubit": "Q1", "amplitude_au": -0.2, "delay_s": 1e-6}]
            ),
        )
        pd.testing.assert_frame_equal(
            data["sweep_parameters"],
            pd.DataFrame({"time_s": [5e-9, 10e-9]}),
        )

    def test_supports_different_length_sweep_axes(self):
        data = make_experiment_data(
            protocol="test",
            fixed={"envelope_weights": [0.1, 0.5, 0.1]},
            sweeps={"time_s": [1, 2], "amplitude_au": [0.1]},
        )
        self.assertEqual(
            data["experiment_parameters"].iloc[0]["envelope_weights"],
            "[0.1, 0.5, 0.1]",
        )
        self.assertEqual(data["sweep_parameters"].shape, (2, 2))
        self.assertTrue(np.isnan(data["sweep_parameters"].iloc[1]["amplitude_au"]))

    def test_builds_general_measurement_recipe(self):
        data = make_measurement_data(
            protocol="rabi",
            builder="make_rabi_timeline",
            parameters={"qubit": "Q1", "pulse_width_s": 20e-9},
            sweeps={"amplitude_au": np.array([0.1, 0.2])},
            repetitions=1000,
            processor=str,
            compilation={"reset_delay": np.float64(50e-6)},
            source_reference={"notebook": "IQM/rabi.ipynb", "cell": "timeline"},
        )

        recipe = data["measurement_recipe"]
        self.assertEqual(recipe["builder"]["callable"], "make_rabi_timeline")
        self.assertEqual(recipe["acquisition"]["repetitions"], 1000)
        self.assertEqual(recipe["compilation"]["reset_delay"], 50e-6)
        self.assertEqual(recipe["sweep_parameters"]["amplitude_au"], [0.1, 0.2])

    def test_run_wrapper_records_the_values_it_uses(self):
        class FakeQPU:
            def run(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                return "result"

        qpu = FakeQPU()
        sequence = object()
        result = run_with_recipe(
            qpu,
            sequence,
            str,
            protocol="ramsey",
            parameters={"delay_s": 1e-6},
            repetitions=200,
            compilation={"batch_size": 20},
        )

        self.assertEqual(result, "result")
        self.assertEqual(qpu.kwargs["repetitions"], 200)
        self.assertEqual(qpu.kwargs["data"]["sequence"], sequence)
        recipe = qpu.kwargs["data"]["measurement_recipe"]
        self.assertEqual(recipe["acquisition"]["repetitions"], 200)
        self.assertEqual(recipe["compilation"], {"batch_size": 20})


if __name__ == "__main__":
    unittest.main()
