import pytest

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
    def test_get_modulations(self):
        r = ReadoutResonator(name="R0", frequency=6e9)
        assert r.get_modulations() == {"R0.mod": 6e9}
