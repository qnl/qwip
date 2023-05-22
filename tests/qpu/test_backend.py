import numpy as np
import pytest
from numpy.random import default_rng
import matplotlib.pyplot as plt

import qwip
from qwip.processing.processors import GMMClassification, StatePopulations
from qwip.qpu.backend import FakeBackend
from qwip.qpu.backend import SimulatorBackend
from qwip.sequencer import ReadoutMarker, Sequence, SequenceElement


class TestQuantumBackend:
    @pytest.fixture
    def backend(self, qpu_01, seed):
        backend = FakeBackend(rng=default_rng(seed))
        backend.update_parameters(qpu_01)

        return backend

    @pytest.fixture
    def sequencer(self, qpu_01):
        return qpu_01.sequencer

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

    def test_acquire_zeros(self, qpu_01, backend, compiled):
        qpu = qpu_01

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

    def test_acquire_sin(self, qpu_01, backend, compiled):
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

        cseq = compiled(20)
        backend.upload(cseq, data_func)
        data = backend.acquire(cseq)

        results = qpu.pipeline.process_results(
            data, dict(R0=StatePopulations, R1=StatePopulations)
        )

        R0_expect = (np.sin(np.r_[:20] / 20 * 2 * np.pi) + 1) / 2

        assert (np.abs(results["R0"].data["1"] - R0_expect) < 1e-2).all()
        assert (results["R1"].data["1"] < 1e-2).all()
    

class TestSimulatorBackend:

    @pytest.fixture
    def sequencer(self, qpu_01):
        return qpu_01.sequencer

    @pytest.fixture
    def compiled(self, sequencer):
        def generate_cseq(shape):
            ro = SequenceElement()
            seq = Sequence.empty(shape)
            seq = seq + SequenceElement().add_waveform(ReadoutMarker(), 10e-9)

            return sequencer.compile(seq, readout=ro)

        return generate_cseq

    @pytest.fixture
    def sim_backend(self, qpu_01):
        backend = SimulatorBackend()
        backend.update_parameters(qpu_01)

        return backend

    @pytest.fixture
    def compile_Q0_X(self, qpu_01):
        db = qpu_01.db
        Q0_X = db.load_pulse('Q0_X90', variables=dict(width="rabi_width"))

        rabi_se = SequenceElement()
        rabi_se.append(Q0_X)
        rabi_se.add_waveform(ReadoutMarker(), location=Q0_X.width)

        ro_se = SequenceElement()

        ts = np.linspace(0, 12.54e-9, 21)
        seq = Sequence.sweep(rabi_se, rabi_width=ts)

        cseq = qpu_01.sequencer.compile(seq, readout=ro_se)      

        return cseq
    
    def check_fft(self, ts, drive):
        from qwip.analysis.frequency import simple_fft

        fs, yfs = simple_fft(ts, drive)

        fig, ax = plt.subplots()
        ax.plot(fs, np.abs(yfs))
        ax.set_xlim(4.6e9, 6e9)
        plt.show()
    

    def plot_states(self, result):
        states = np.array(result.expect)
        ts = list(range(states.shape[1]))
        N = states.shape[0]

        fig, ax = plt.subplots()
        for level in range(N):
            ax.plot(ts, states[level], label=f'$|{level}⟩$')
        ax.legend()
        plt.show()


    def test_sim_upload_db(self, qpu_01, sim_backend):
        assert sim_backend.qpu is qpu_01


    def test_sim_identify_on_channels(self, qpu_01, sim_backend, compile_Q0_X):
        cseq = compile_Q0_X

        sim_backend.update_parameters(qpu_01)
        sim_backend.upload(cseq)
        
        assert sim_backend.on_channels() == [(0, 1)]


    def test_sim_upload_Q0X_seq(self, qpu_01, sim_backend, compile_Q0_X):
        cseq = compile_Q0_X
        sim_backend.update_parameters(qpu_01)
        sim_backend.upload(cseq)
        
        assert len(sim_backend.uploaded.array[0]) == 21

    
    def test_sim_acquire_Q0X_seq(self, sim_backend, compile_Q0_X):
        cseq = compile_Q0_X
        H_list = sim_backend.upload(cseq)

        results = sim_backend.acquire(cseq)
        assert len(results["Q0"]["results"]) == 21


    def test_sim_acquire_Q0X_seq_fft_correct(self, sim_backend, compile_Q0_X):
        cseq = compile_Q0_X
        data = sim_backend.acquire(cseq)
        self.check_fft(sim_backend.ts, sim_backend.drive)
        assert True
    
    def test_sim_acquire_Q0X_seq_states_correct(self, sim_backend, compile_Q0_X):
        cseq = compile_Q0_X
        data = sim_backend.acquire(cseq)

        full_seq = data["Q0"]["results"][-1]
        self.plot_states(full_seq)

        assert True

        
        


        


