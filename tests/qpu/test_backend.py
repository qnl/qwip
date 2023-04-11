import numpy as np
import pytest

from numpy.random import default_rng

import qwip
from qwip.processing.processors import GMMClassification, StatePopulations
from qwip.qpu.backend import FakeBackend
from qwip.sequencer import ReadoutMarker, Sequence, SequenceElement


class TestQuantumBackend:
    @pytest.fixture
    def backend(self, qpu, seed):
        backend = FakeBackend(rng=default_rng(seed))
        backend.update_parameters(qpu)

        return backend

    @pytest.fixture
    def compiled(self, sequencer):
        def generate_cseq(shape):
            ro = SequenceElement()
            seq = Sequence.empty(shape)
            seq = seq + SequenceElement().add_waveform(ReadoutMarker(), 10e-9)

            return sequencer.compile(seq, readout=ro)

        return generate_cseq

    def test_upload(self, backend, compiled):
        cseq = compiled(15)

        def zeros(readout_key, element_index, readout_index, repetitions):
            return np.zeros(repetitions)

        backend.upload(cseq, data_func=zeros)

        assert backend.uploaded is cseq
        assert backend.data_func is zeros

    def test_update_parameters(self, qpu, backend):
        gmm0 = GMMClassification(
            measurement_key="R0",
            means=np.array([[5, 0], [-5, 0]], dtype=float),
            covariances=np.array([1, 1], dtype=float)
        )

        qpu.pipeline.add_processor(gmm0)
        
        backend.update_parameters(qpu)
        assert backend.gmms["R0"] is gmm0

    def test_acquire_zeros(self, qpu, backend, compiled):
        def data_func(key, element, readout, repetitions):
            return np.zeros(repetitions)

        cseq = compiled(15)
        backend.upload(cseq, data_func)
        data = backend.acquire(cseq)

        results = qpu.pipeline.process_results(
            data, dict(R0=StatePopulations, R1=StatePopulations)
        )

        assert (results["R0"].data["0"] > 0.99).all()
        assert (results["R1"].data["0"] > 0.99).all()

    def test_acquire_sin(self, qpu, backend, compiled):
        def data_func(key, element, readout, repetitions):
            if key == "R0":
                populations = (np.sin(element / 20 * 2 * np.pi) + 1) / 2
                ones = np.round(populations* repetitions).astype(int)

                states = np.zeros(repetitions)

                states[:ones] = 1
                return states
            else:
                return np.zeros(repetitions)

        cseq = compiled(20)
        backend.upload(cseq, data_func)
        data = backend.acquire(cseq)

        results = qpu.pipeline.process_results(
            data, dict(R0=StatePopulations, R1=StatePopulations)
        )

        R0_expect = (np.sin(np.r_[:20] / 20 * 2 * np.pi) + 1) / 2

        assert (np.abs(results["R0"].data["1"] - R0_expect) < 1e-2).all()
        assert (results["R1"].data["1"] < 1e-2).all()
