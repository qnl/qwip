"""Tests for qwip.processing.classification"""
import numpy as np
import pytest
from loguru import logger
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_array_equal

import qwip
from qwip.processing.classification import GMMData
from qwip.processing.process import ProcessSettings

logger.enable("qwip")


@pytest.fixture
def rng():
    return default_rng(12345)


class TestIQRotation:
    def get_sample(self, rng, size):
        return rng.random(size=size) * 50 + 120 * rng.random(size=2)

    def get_data(self, rng, sizes):
        return {f"R{i}": self.get_sample(rng, size) for i, size in enumerate(sizes)}

    @pytest.fixture
    def process_settings(self):
        return ProcessSettings(
            name="rotate",
            process_type="classification.IQRotation",
        )

    @pytest.fixture
    def iq_process(self, process_settings):
        return process_settings.get_process()

    def test_identity(self, rng, iq_process):
        sizes = ((3, 1, 4, 2), (1, 3, 4, 2), (3, 4, 1, 2))

        data = self.get_data(rng, sizes)

        output = iq_process(data)

        for actual, desired in zip(output.values(), data.values()):
            assert_allclose(actual, desired)

    def test_rotate(self, rng, iq_process):
        sizes = ((3, 1, 2, 2), (3, 2, 3, 2))

        data = self.get_data(rng, sizes)
        iq_process.angles.update(dict(R0=15, R1=-71))

        outputs = iq_process(data)

        desired = {
            "R0": np.array(
                [
                    [81.24213335, 50.07908405],
                    [104.12021614, 74.81814727],
                    [88.94401277, 52.97389481],
                    [100.84137681, 48.60014002],
                    [94.6655971, 86.03057576],
                    [74.0717228, 80.87886285],
                ]
            ),
            "R1": np.array(
                [
                    [67.85916681, -35.83760845],
                    [45.54529936, -57.03801303],
                    [41.11182542, -60.49339845],
                    [27.64500177, -30.6343227],
                    [46.28628693, -37.88565773],
                    [61.66105633, -28.69547008],
                    [28.02485476, -36.4103308],
                    [48.54757403, -23.96951152],
                    [61.11359924, -59.99499096],
                    [68.19355759, -61.64203831],
                    [76.70127198, -54.93493257],
                    [71.97795762, -39.93733756],
                    [39.75805603, -48.32413254],
                    [57.55212983, -39.91219267],
                    [66.85556381, -30.3160928],
                    [39.00801758, -36.00126559],
                    [39.77534269, -35.83114154],
                    [48.5571783, -19.38446386],
                ]
            ),
        }

        for key, actual in outputs.items():
            assert_allclose(actual, desired[key].reshape(actual.shape))

    def test_radians(self, rng, iq_process):
        sizes = ((2, 1, 5, 2), (1, 2, 2, 2))

        data = self.get_data(rng, sizes)
        iq_process.angles.update(dict(R0=np.pi / 2, R1=-np.pi))

        with qwip.qsettings.context():
            qwip.qsettings["units/phase"] = "radians"
            outputs = iq_process(data)

        desired = {
            "R0": np.array(
                [
                    [-35.02538911, 21.15814947],
                    [-53.00020567, 49.65962121],
                    [-35.82816852, 29.34682588],
                    [-28.52418141, 39.70678602],
                    [-66.27761539, 43.42915055],
                    [-66.63152972, 22.20363408],
                    [-23.98236891, 43.153221],
                    [-63.5114681, 31.88333165],
                    [-35.51111533, 44.66402334],
                    [-30.19421991, 46.48775651],
                ]
            ),
            "R1": np.array(
                [
                    [-119.57403777, -95.45420669],
                    [-115.89007994, -112.98336917],
                    [-112.23374799, -78.66800281],
                    [-107.1522661, -102.12294969],
                ]
            ),
        }

        for key, actual in outputs.items():
            assert_allclose(actual, desired[key].reshape(actual.shape))


class TestGMM:
    def generate_IQ_data(self, rng, shape, means, covariances):
        assert means.shape == (covariances.shape[0], 2)

        n_states = means.shape[0]

        states = rng.choice(n_states, size=shape)
        IQ = np.empty(shape=(*shape, 2), dtype=np.float64)

        for i in range(n_states):
            idx = states == i
            IQ[idx] = rng.multivariate_normal(
                mean=means[i],
                cov=covariances[i] * np.identity(2),
                size=np.count_nonzero(idx),
            )

        return IQ, states

    def generate_data(self, rng, sizes, gmms):
        data = {}
        answer = {}
        for i, (k, gmm) in enumerate(gmms.items()):
            data[k], answer[k] = self.generate_IQ_data(
                rng, sizes[i], gmm.means, gmm.covariances
            )

        return data, answer

    @pytest.fixture
    def psettings(self):
        return ProcessSettings(
            name="gmm",
            process_type="classification.GMM",
        )

    @pytest.fixture
    def process_three_state(self, psettings):
        gmms = {
            "R1": {
                "means": np.array([30.0, 32.0, 31.0]),
                "covariances": np.array([[94.0, 20.0], [72.0, -45.2], [-62.1, 55.0]]),
            }
        }

        psettings.parameters.update(gmms=gmms)

        return psettings.get_process()

    def test_init(self, psettings):
        gmm = psettings.get_process(
            gmms={"R1": {"means": np.zeros((2, 2)), "covariances": np.zeros(2)}}
        )

        assert isinstance(gmm.gmms["R1"], GMMData)
        assert "R1" in gmm.mixes

    CASES = [
        (  # Two state classification
            dict(
                R1=dict(
                    means=np.array([[94.0, 20.0], [72.0, -45.2]]),
                    covariances=np.array([30.0, 32.0]),
                )
            ),
            [(512, 20, 1)],
        ),
        (  # Three state classification
            dict(
                R2=dict(
                    means=np.array([[94.0, 20.0], [72.0, -45.2], [-62.1, 55.0]]),
                    covariances=np.array([30.0, 32.0, 31.0]),
                )
            ),
            [(5, 4, 3, 2)],
        ),
        (
            dict(
                R1=dict(
                    means=np.array([[94.0, 20.0], [72.0, -45.2]]),
                    covariances=np.array([30.0, 32.0]),
                ),
                R2=dict(
                    means=np.array([[94.0, 20.0], [72.0, -45.2], [-62.1, 55.0]]),
                    covariances=np.array([30.0, 32.0, 31.0]),
                ),
            ),
            [(2, 3, 4), (5, 4, 3, 2)],
        ),
    ]

    @pytest.mark.parametrize("gmms,sizes", CASES)
    def test_classify_states(self, rng, psettings, gmms, sizes):
        gmm = psettings.get_process(gmms=gmms)
        IQ, states = self.generate_data(rng, sizes, gmm.gmms)

        output = gmm(IQ)

        for k in states:
            assert_array_equal(output[k], states[k])

    def test_missing_means_covariances(self, psettings):
        with pytest.raises(TypeError):
            psettings.get_process()

        with pytest.raises(TypeError):
            psettings.get_process(means=dict(R2=np.zeros(2, 2)))

        with pytest.raises(TypeError):
            psettings.get_process(covariances=dict(R2=np.zeros(2)))

    def test_update_settings(self, psettings):
        gmmdata = {"R1": {"means": np.zeros((2, 2)), "covariances": np.ones(2)}}
        gmm = psettings.get_process(gmms=gmmdata)

        assert psettings.parameters == {
            "gmms": gmmdata,
        }


class TestStatePopulations:
    @pytest.fixture
    def process_settings(self):
        return ProcessSettings(
            name="populations",
            process_type="classification.StatePopulations",
        )

    def test_simple(self, process_settings):
        state_pops = process_settings.get_process()

        # All zeros

        classified = np.zeros((5, 4, 3, 2))
        expected = np.zeros((2, 4, 3, 2))
        expected[0] = 1

        outputs = state_pops(dict(R0=classified))
        assert_allclose(outputs["R0"], expected)

        # All ones

        classified = np.ones((2, 2))
        expected = np.zeros((2, 2))
        expected[1] = 1

        outputs = state_pops(dict(R0=classified))
        assert_allclose(outputs["R0"], expected)

        # Simple case

        classified = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1]])
        expected = np.zeros((2, 3))
        expected[0] = [0.25, 0.75, 0.0]
        expected[1] = np.ones(3) - expected[0]

        outputs = state_pops(dict(R0=classified))

        assert_allclose(outputs["R0"], expected)

    def test_three_state(self, process_settings):
        state_pops = process_settings.get_process(states=3)

        classified = np.zeros((5, 4, 3, 2))
        expected = np.zeros((3, 4, 3, 2))
        expected[0] = 1

        outputs = state_pops(dict(R0=classified))

        assert_allclose(outputs["R0"], expected)

    def test_different_num_states(self, process_settings):
        state_pops = process_settings.get_process(states={"R0": 2, "R1": 3})

        classified = np.zeros((5, 4, 3, 2))
        expected = np.zeros((2, 4, 3, 2))
        expected[0] = 1

        outputs = state_pops(dict(R0=classified))
        assert_allclose(outputs["R0"], expected)

        expected = np.zeros((3, 4, 3, 2))
        expected[0] = 1

        outputs = state_pops(dict(R1=classified))
        assert_allclose(outputs["R1"], expected)

    def test_num_states_default(self, process_settings):
        state_pops = process_settings.get_process(states={"default": 3})

        classified = np.ones((1, 4, 3, 2))
        expected = np.zeros((3, 4, 3, 2))
        expected[1] = 1

        outputs = state_pops(dict(R0=classified))
        assert_allclose(outputs["R0"], expected)
