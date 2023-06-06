import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.random import default_rng
import qutip as qt

import qwip
from qwip.processing.processors import GMMClassification, StatePopulations
from qwip.qpu.backend import FakeBackend, SimulatorBackend, OperatorChannelMap
from qwip.sequencer import ReadoutMarker, Sequence, SequenceElement
from qwip.qpu.backend import on_channels, simulate_H
from qwip.analysis.frequency import simple_fft

import tests.qpu.backend_test_data as tdata


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


def check_fft(ts, drive):
    fs, yfs = simple_fft(ts, drive)
    return fs[np.argmax(yfs)]

def plot_states(result, ts, N):
        states = np.array(result.expect)
        ts = list(range(states.shape[1]))
        N = states.shape[0]

        fig, ax = plt.subplots()
        for level in range(N):
            ax.plot(ts, states[level], label=f"$|{level}⟩$")
        ax.legend()
        plt.show()

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
    def sim_backend(self):
        backend = SimulatorBackend()
        return backend

    @pytest.fixture
    def compile_Q0_X(self, qpu_01):
        db = qpu_01.db
        Q0_X = db.load_pulse("Q0_X90", variables=dict(width="rabi_width"))

        rabi_se = SequenceElement()
        rabi_se.append(Q0_X)
        rabi_se.add_waveform(ReadoutMarker(), location=Q0_X.width)
        ro_se = SequenceElement()

        ts = np.linspace(0, 12.54e-9, 21)
        seq = Sequence.sweep(rabi_se, rabi_width=ts)
        cseq = qpu_01.sequencer.compile(seq, readout=ro_se)

        return cseq
    

    def test_update_parameters(self, qpu_01, sim_backend):
        sim_backend.update_parameters(qpu_01)
        a, adag = qt.destroy(4), qt.create(4)

        assert sim_backend.num_levels == 4
        for i in range(8):
            assert sim_backend.channel_map[i] == OperatorChannelMap(operator=a+adag, channels=(2*i, 2*i+1), amplitude_factor=40e6, LO_frequency=5.8e9, name=f"Q{i}")

            assert list(sim_backend.static_hamiltonian.keys()) == [f"Q{i}" for i in range(8)]

            #assert np.allclose(np.array(sim_backend.static_hamiltonian[f"Q{i}"].data, dtype=float), np.array((2*np.pi*(50+i)*1e8*adag*a + 2*np.pi*(-200e6/2)*adag*adag*a*a).data, dtype=float))

        # test error thrown correctly


    def test_on_channels(self, compile_Q0_X):
        assert (on_channels(compile_Q0_X.array) == [0, 1]).all()
    

    def test_upload(self, compiled, qpu_01, sim_backend):
        cseq = compiled(20)
        sim_backend.update_parameters(qpu_01)
        sim_backend.upload(cseq)

        assert sim_backend.drive_hamiltonian == {}


    def test_upload_Q0X_seq(self, qpu_01, sim_backend, compile_Q0_X):
        sim_backend.update_parameters(qpu_01)
        sim_backend.upload(compile_Q0_X)

        assert abs(np.max(check_fft(sim_backend.ts, sim_backend.drive)) - 5e9)/5e9 < 0.01
        
        # Check drive
        #assert (2*np.pi*sim_backend.channel_map[0].amplitude_factor*sim_backend.drive == tdata.drive_X90)
        
        H = [[sim_backend.static_hamiltonian["Q0"], np.ones_like(sim_backend.ts)],
             [sim_backend.drive_hamiltonian["Q0"][0],sim_backend.drive_hamiltonian["Q0"][1]]
            ]
        result = simulate_H(H, sim_backend.ts, 4)
        plot_states(result, sim_backend.ts, 4)

    
    # test for subsequent X90 pulses
    # test for change in axis of rotation
    

