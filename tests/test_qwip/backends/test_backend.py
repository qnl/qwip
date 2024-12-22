import numpy as np
import pytest
from numpy.random import default_rng

from qwip.backends.backend import DummyBackend
from qwip.processing.processors import GMMClassification, StatePopulations
from qwip.sequencer import ReadoutMarker, Sequence, Timeline
from qwip.sequencer.compilation import IntermediateProgram, QWiPExecutable


class TestQuantumBackend:
    @pytest.fixture
    def backend(self, qpu_01, seed):
        backend = DummyBackend(rng=default_rng(seed))
        backend.update_parameters(qpu_01)

        return backend

    @pytest.fixture
    def exe(self):
        def make_exe(shape):
            exe = QWiPExecutable(
                programs=dict(
                    dac=IntermediateProgram(device="dac", read_registers={0, 1, 2}),
                ),
                num_reads=[1] * shape,
            )

            return exe

        return make_exe

    def test_upload(self, backend, exe):
        exe = exe(15)

        def zeros(readout_key, element_index, readout_index, repetitions):
            return np.zeros(repetitions)

        backend.upload(exe, data_func=zeros)

        assert backend.uploaded is exe
        assert backend.data_func is zeros

    def test_update_parameters(self, qpu_01, backend):
        qpu = qpu_01
        gmm0 = GMMClassification(
            measurement_key="R0",
            means=np.array([[5, 0], [-5, 0]], dtype=float),
            covariances=np.array([1, 1], dtype=float),
        )

        qpu.pipeline.add_processor(gmm0)

        backend.update_parameters(qpu)
        assert backend.gmms["R0"] is gmm0

    def test_acquire_zeros(self, qpu_01, backend, exe):
        qpu = qpu_01

        def data_func(key, element, readout, repetitions):
            return np.zeros(repetitions)

        exe = exe(15)
        backend.upload(exe, data_func)
        data = backend.acquire()

        results = qpu.pipeline.process_results(
            data, dict(R0=StatePopulations, R1=StatePopulations)
        )

        assert (results["R0"].data["0"] > 0.99).all()
        assert (results["R1"].data["0"] > 0.99).all()

    @pytest.mark.xfail
    def test_acquire_sin(self, qpu_01, backend, exe):
        qpu = qpu_01

        def data_func(key, element, readout, repetitions):
            if key == "R0":
                populations = (np.sin(element / 20 * 2 * np.pi) + 1) / 2
                ones = np.round(populations * repetitions).astype(int)

                states = np.zeros(repetitions)

                states[:ones] = 1
                return states
            else:
                return np.zeros(repetitions)

        exe = exe(20)
        backend.upload(exe, data_func)
        data = backend.acquire()

        results = qpu.pipeline.process_results(
            data, dict(R0=StatePopulations, R1=StatePopulations)
        )

        R0_expect = (np.sin(np.r_[:20] / 20 * 2 * np.pi) + 1) / 2

        assert (np.abs(results["R0"].data["1"] - R0_expect) < 1e-2).all()
        assert (results["R1"].data["1"] < 1e-2).all()
