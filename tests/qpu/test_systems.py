import numpy as np
import pytest
from numpy.testing import assert_allclose

from qwip.qpu.systems import QuantumSystem, ReadoutResonator, Transmon


class TestQuantumSystem:
    def test_get_modulations(self):
        assert QuantumSystem(name="Q").get_modulations() == dict()


class TestTransmon:
    def test_get_modulations(self):
        t = Transmon(name="Q0", frequency=5e9)
        assert t.get_modulations() == {"Q0.mod_GE": 5e9}

        t = Transmon(name="Q1", frequency=5.1e9, local_oscillator="qubit_LO")
        with pytest.raises(KeyError):
            t.get_modulations()
        assert t.get_modulations(dict(qubit_LO=5.2e9)) == {"Q1.mod_GE": -100e6}

        t.anharmonicity = -200e6
        assert t.get_modulations(dict(qubit_LO=5e9)) == {
            "Q1.mod_GE": 100e6,
            "Q1.mod_EF": -100e6,
        }

    def test_modulation_name(self):
        t = Transmon(name="Q2", frequency=5e9, modulation_name="mod_name")
        assert t.get_modulations() == {"mod_name": 5e9}

        t.modulation_name = "{name}"
        assert t.get_modulations() == {"Q2": 5e9}

        t.modulation_name = "{mod_key}_for_{name}"
        assert t.get_modulations() == {"GE_for_Q2": 5e9}


class TestReadoutResonator:
    @pytest.fixture
    def resonator(self):
        r = ReadoutResonator(
            frequency=7e9 * 2 * np.pi,
            kappa=0.5e6 * 2 * np.pi,
            chi=(-0.5e6 * 2 * np.pi, 0.5e6 * 2 * np.pi, 0.5e6 * 2 * np.pi),
            eta=1.0,
            name="R0",
        )  # Implement eta later
        return r

    def test_get_modulations(self):
        r = ReadoutResonator(name="R0", frequency=6e9)
        assert r.get_modulations() == {"R0.mod": 6e9}

    @pytest.mark.parametrize(
        "drive,chis",
        [
            (1e7, (-0.5e6 * 2 * np.pi, 0.5e6 * 2 * np.pi, 0.5e6 * 2 * np.pi)),
            (0, (0, 0, 0)),
            (1, (0, 0, 0)),
        ],
    )
    def test_get_cavity_field_equation(self, drive, chis, resonator, data_file):
        expected = np.loadtxt(str(data_file), delimiter=",", dtype=complex)
        resonator.chi = chis

        def drive_env(t):
            return drive

        alpha = resonator.get_cavity_field_equation(drive_envelope=drive_env)

        assert_allclose(
            expected, alpha(np.linspace(0, 1, 100), np.linspace(0, 0.5, 100))
        )

    def test_solve_cavity_field_equation_empty(self, resonator):
        with pytest.raises(ValueError):
            resonator.solve_cavity_field_equation(ts=[], drive_envelope=lambda t: 1e7)

    def test_solve_cavity_field_equation(self, resonator, data_file):
        expected = np.loadtxt(str(data_file), delimiter=",", dtype=complex)

        alphas = resonator.solve_cavity_field_equation(
            ts=np.linspace(0, 1e-5, 1000), drive_envelope=lambda t: 1e7
        )
        assert_allclose(alphas, expected)

    def test_solve_cavity_field_equation_decay(
        self,
        resonator,
    ):
        resonator.chi = (0, 0, 0)

        ts = np.linspace(0, 1e-6, 1000)
        # When drive = 0, field equation should be an exponential function
        alphas = resonator.solve_cavity_field_equation(
            ts=ts, drive_envelope=lambda t: 0, alpha_0=[1.0, 1.0, 1.0]
        )

        expected_amplitudes = np.exp(-2 * np.pi * resonator.kappa / 2 * ts)
        assert_allclose(alphas, np.tile(expected_amplitudes, (3, 1)), atol=1e-3)

    def test_solve_cavity_field_equation_alpha(self, resonator):
        with pytest.raises(ValueError):
            resonator.solve_cavity_field_equation(
                ts=np.array([1, 2]), drive_envelope=lambda t: 0, alpha_0=[0.0, 0.0]
            )

    @pytest.mark.parametrize("drive,chi,kappa", [(1, 0.5, 1), (1, 5, 0.5), (1, 0.5, 5)])
    def test_solve_cavity_field_equation_steady_state(
        self,
        drive,
        chi,
        kappa,
        resonator,
    ):
        ts = np.linspace(0, 50 / kappa, 201)
        resonator.kappa = kappa
        resonator.chi = (-chi, chi)

        alphas = resonator.solve_cavity_field_equation(
            ts=ts, drive_envelope=lambda t: drive
        )

        # expected steady state value
        alpha_s0 = -drive / (-chi - 1j * kappa / 2)
        alpha_s1 = -drive / (chi - 1j * kappa / 2)

        assert_allclose([alphas[0][-1], alphas[1][-1]], [alpha_s0, alpha_s1], atol=2e-3)
