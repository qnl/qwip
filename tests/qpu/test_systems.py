import pytest
import numpy as np

from qwip.qpu.systems import QuantumSystem, ReadoutResonator, Transmon
from numpy.testing import assert_allclose

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
    def compile_resonator(self):
        r = ReadoutResonator(
            frequency=7e9*2*np.pi, 
            kappa=0.5e6*2*np.pi, 
            chi=(-0.5e6*2*np.pi, +1*0.5e6*2*np.pi, 0.5e6*2*np.pi), 
            eta=1.0,
            name="R0")   # Implement eta later
        return r

    @pytest.fixture
    def compile_resonator_kappa(self):
        r = ReadoutResonator(
            frequency=7e9*2*np.pi, 
            kappa=0.5e6*2*np.pi, 
            chi=(0, 0, 0), 
            eta=1.0,
            name="R0")   
        return r
    
    def test_get_modulations(self):
        r = ReadoutResonator(name="R0", frequency=6e9)
        assert r.get_modulations() == {"R0.mod": 6e9}
    
    def test_get_cavity_field_equation(
            self,
            compile_resonator,
            data_file
    ):
        expected = np.loadtxt(str(data_file), delimiter=',', dtype=complex)
        drive_envelope=lambda t: 1e7
        r = compile_resonator

        # Check None returned when no chi value exists for state
        assert (
            r.get_cavity_field_equation(qubit_state=3, drive_envelope=drive_envelope)
                is None
        )

        # Check ground |0> state
        field_equation_0 = r.get_cavity_field_equation(
            drive_envelope=drive_envelope)
        
        # Check excited |1> state
        field_equation_1 = r.get_cavity_field_equation(
            qubit_state=1,
            drive_envelope=drive_envelope)
        
        assert_allclose(expected, [
            field_equation_0(np.linspace(0, 1, 100), np.linspace(0, 0.5, 100)),
            field_equation_1(np.linspace(0, 1, 100), np.linspace(0, 0.5, 100))
        ])
    
    def test_get_cavity_field_equation_kappa(
            self, 
            compile_resonator_kappa,
            data_file
    ):
        ## Test: Kappa is the only nonzero parameter
        expected = np.loadtxt(str(data_file), delimiter=',', dtype=complex)
        r = compile_resonator_kappa

        # Check zero drive -- no imag component
        drive = lambda t: 0
        field_equation_0 = r.get_cavity_field_equation(
            drive_envelope=drive,
            qubit_state=0
        )
        field_equation_1 = r.get_cavity_field_equation(
            drive_envelope=drive,
            qubit_state=1
        )
        
        # Check nonzero drive -- has imag component
        drive = lambda t: 1
        field_equation_2 = r.get_cavity_field_equation(
            drive_envelope=drive,
            qubit_state=0
        )
        field_equation_3 = r.get_cavity_field_equation(
            drive_envelope=drive,
            qubit_state=1
        )

        assert_allclose(expected, [
            field_equation_0(np.linspace(0, 1, 100), np.linspace(0, 0.5, 100)),
            field_equation_1(np.linspace(0, 1, 100), np.linspace(0, 0.5, 100)),
            field_equation_2(np.linspace(0, 1, 100), np.linspace(0, 0.5, 100)),
            field_equation_3(np.linspace(0, 1, 100), np.linspace(0, 0.5, 100))
        ])

    def test_solve_cavity_field_equation(self, compile_resonator, data_file):
        r = compile_resonator
        expected = np.loadtxt(str(data_file), delimiter=",", dtype=complex)

        with pytest.raises(ValueError):
            r.solve_cavity_field_equation(ts=np.array([]),
                                          drive_envelope=lambda t: 1e7)
        
        # Check returns array of NaN values when no chi value for qubit state
        assert np.isnan(r.solve_cavity_field_equation(
            ts=np.array([1, 2, 3]),
            qubit_state=3,
            drive_envelope=lambda t: 1e7)
        ).all()
        
        # General case with all nonzero parameters
        field_amplitudes = r.solve_cavity_field_equation(
            ts = np.linspace(0, 1e-5, 1000),
            drive_envelope = lambda t: 1e7,
            qubit_state = 1
        )
        assert_allclose(expected, field_amplitudes[0])

        # Check limit when t becomes very large (if drive envelope = 0, then the field
        # equation is exponential and approaches 0)
        field_amplitudes = r.solve_cavity_field_equation(
            ts = np.linspace(0, 2e-5, 1000),
            drive_envelope = lambda t: 0,
            qubit_state = 1,
            alpha_0 = [1]
        )

        assert abs(field_amplitudes[0][-1]) < 1e-4
    
    def test_solve_cavity_field_equation_exp(
            self, 
            compile_resonator_kappa,
    ):
        r = compile_resonator_kappa

        # When drive = 0, field equation should be an exponential function
        field_amplitudes = r.solve_cavity_field_equation(
            ts = np.linspace(0, 1e-6, 1000),
            drive_envelope = lambda t: 0,
            qubit_state = 1,
            alpha_0 = [1.0]
        )

        expected_amplitudes = np.exp(np.linspace(0, 1e-6, 1000)*-r.kappa/2)
        assert_allclose(field_amplitudes[0], expected_amplitudes, atol=1e-3)


