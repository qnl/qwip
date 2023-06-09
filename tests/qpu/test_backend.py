import matplotlib.pyplot as plt
import numpy as np
import pytest
import qutip as qt
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_array_equal

import qwip
from qwip.analysis.frequency import simple_fft
from qwip.processing.processors import GMMClassification, StatePopulations
from qwip.qpu.backend import (
    FakeBackend,
    OperatorChannelMap,
    SimulatorBackend,
    TimeDependentHamiltonian,
    get_active_channels,
)
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


def check_fft(ts, drive):
    fs, yfs = simple_fft(ts, drive)
    return fs[np.argmax(yfs)]


def plot_states(result, ts, labels):
    states = np.array(result.expect)
    N = states.shape[0]

    fig, ax = plt.subplots()
    for level in range(N):
        ax.plot(ts, states[level], label=f"$|{labels[level]}⟩$")
    ax.legend()
    plt.show()


class TestTimeDependentHamiltonian:
    @pytest.mark.parametrize(
        "H,shape",
        [
            ([], None),
            ([qt.sigmax()], (2, 2)),
            ([qt.tensor(qt.sigmax(), qt.destroy(5))], (10, 10))
        ]
    )
    def test_shape(self, H, shape):
        H_list = [(Hi, np.ones(10)) for Hi in H]

        H_t = TimeDependentHamiltonian(H=H_list, ts=np.arange(10))

        assert H_t.shape == shape

    @pytest.mark.parametrize(
        "H,dims",
        [
            ([], None),
            ([qt.sigmax()], [[2], [2]]),
            ([qt.tensor(qt.sigmax(), qt.destroy(5))], [[2, 5], [2, 5]])
        ]
    )
    def test_dims(self, H, dims):
        H_list = [(Hi, np.ones(10)) for Hi in H]

        H_t = TimeDependentHamiltonian(H=H_list, ts=np.arange(10))

        assert H_t.dims == dims

    def test_simulate_single_qubit(self):
        f = 1e9
        ts = np.linspace(0, 5e-9, 201)

        # Single qubit case
        H = 2 * np.pi * f * qt.sigmax() / 2
        H_t = TimeDependentHamiltonian(H=[(H, np.ones_like(ts))], ts=ts)

        result = H_t.simulate()
        psis = H_t.get_basis()

        labeled = dict(zip(psis.keys(), result.expect))

        for state, populations in labeled.items():
            match state:
                case (0,):
                    expected = 0.5 * np.cos(2*np.pi*1e9*H_t.ts) + 0.5
                case (1,):
                    expected = -0.5 * np.cos(2*np.pi*1e9*H_t.ts) + 0.5
                case _:
                    expected = np.zeros_like(H_t.ts)

            assert_allclose(populations, expected, atol=2e-5)

    def test_simulate_multi_qubit(self):
        f = 1e9
        ts = np.linspace(0, 5e-9, 201)

        # Three qubit case
        H_1 = 2 * np.pi * f * qt.tensor(qt.sigmax(), qt.qeye(2), qt.qeye(2)) / 2
        H_2 = 2 * np.pi * f * qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(2)) / 2
        H_3 = 2 * np.pi * f * qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2)) / 2
        H_t = TimeDependentHamiltonian(
            H=[
                (H_1, np.ones_like(ts)),
                (H_2, np.ones_like(ts)),
                (H_3, np.ones_like(ts)),
            ],
            ts=ts,
        )

        result = H_t.simulate()
        psis = H_t.get_basis()
        labeled = dict(zip(psis.keys(), result.expect))

        for state, populations in labeled.items():
            match state:
                case (0, 0, 0):
                    expected = 0.5 * np.cos(2*np.pi*1e9*H_t.ts) + 0.5
                case (1, 0, 0):
                    expected = -0.5 * np.cos(2*np.pi*1e9*H_t.ts) + 0.5
                case _:
                    expected = np.zeros_like(H_t.ts)

            assert_allclose(populations, expected, atol=3e-5)

    def test_tensor_different_times(self):
        H1 = qt.sigmax()
        H2 = qt.sigmaz()

        ## ts not the same
        ts1 = np.linspace(0, 20e-9, 21)
        ts2 = np.linspace(0, 10e-9, 21)

        H1_t = TimeDependentHamiltonian(H=[(H1, np.ones_like(ts1))], ts=ts1)
        H2_t = TimeDependentHamiltonian(H=[(H2, np.ones_like(ts2))], ts=ts2)

        with pytest.raises(ValueError):
            TimeDependentHamiltonian.tensor(H1_t, H2_t)

    def test_tensor_empty(self):
        H0 = TimeDependentHamiltonian(H=[], ts=None)

        assert TimeDependentHamiltonian.tensor(H0, H0) == H0

        ts = np.linspace(0, 10)
        H1 = TimeDependentHamiltonian(
            H=[(qt.sigmax(), np.zeros_like(ts))],
            ts=ts
        )

        assert TimeDependentHamiltonian.tensor(H0, H1) == H1

    def test_tensor(self):
        ts = np.linspace(0, 10)
        H0 = TimeDependentHamiltonian(
            H=[
                (qt.sigmaz(), np.ones_like(ts)),
                (qt.sigmax(), np.cos(2*np.pi*ts))
            ],
            ts=ts
        )

        H1 = TimeDependentHamiltonian(
            H=[
                (qt.create(3)*qt.destroy(3), 2*np.ones_like(ts))
            ],
            ts=ts
        )

        expect = TimeDependentHamiltonian(
            H=[
                (qt.tensor(qt.sigmaz(), qt.qeye(3)), np.ones_like(ts)),
                (qt.tensor(qt.sigmax(), qt.qeye(3)), np.cos(2*np.pi*ts)),
                (qt.tensor(qt.qeye(2), qt.create(3)*qt.destroy(3)), 2*np.ones_like(ts))
            ],
            ts=ts
        )

        assert TimeDependentHamiltonian.tensor(H0, H1) == expect

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

    ## Pulse sequence fixtures

    # One Channel
    @pytest.fixture
    def compile_Q0X90(self, qpu_01):
        db = qpu_01.db
        Q0_X = db.load_pulse("Q0_X90", variables=dict(width="rabi_width"))

        rabi_se = SequenceElement()
        rabi_se.append(Q0_X)
        rabi_se.add_waveform(ReadoutMarker(), location=Q0_X.width)
        ro_se = SequenceElement()

        ts = np.linspace(0, 34.8e-9, 21)
        seq = Sequence.sweep(rabi_se, rabi_width=ts)
        cseq = qpu_01.sequencer.compile(seq, readout=ro_se)

        return cseq

    @pytest.fixture
    def compile_Q0_X180(self, qpu_01):
        db = qpu_01.db
        Q0_X = db.load_pulse("Q0_X90", variables=dict(width="rabi_width"))

        rabi_se = SequenceElement()
        rabi_se.append(Q0_X)
        rabi_se.append(Q0_X, self_loc=Q0_X.width)
        rabi_se.add_waveform(ReadoutMarker(), location=2 * Q0_X.width)
        ro_se = SequenceElement()

        ts = np.linspace(0, 34.8e-9, 21)
        seq = Sequence.sweep(rabi_se, rabi_width=ts)
        cseq = qpu_01.sequencer.compile(seq, readout=ro_se)

        return cseq

    # Two channels
    @pytest.fixture
    def compile_Q0X90_Q1X90(self, qpu_01):
        db = qpu_01.db
        Q0_X = db.load_pulse("Q0_X90", variables=dict(width="rabi_width"))
        Q1_X = db.load_pulse("Q1_X90", variables=dict(width="rabi_width"))

        rabi_se = SequenceElement()
        rabi_se.append(Q0_X)
        rabi_se.append(Q1_X)
        rabi_se.add_waveform(ReadoutMarker(), location=Q0_X.width)
        ro_se = SequenceElement()

        ts = np.linspace(0, 34.8e-9, 21)
        seq = Sequence.sweep(rabi_se, rabi_width=ts)
        cseq = qpu_01.sequencer.compile(seq, readout=ro_se)

        return cseq

    # Three channels
    @pytest.fixture
    def compile_Q0X90_Q1X90_Q2X90(self, qpu_01):
        db = qpu_01.db
        Q0_X = db.load_pulse("Q0_X90", variables=dict(width="rabi_width"))
        Q1_X = db.load_pulse("Q1_X90", variables=dict(width="rabi_width"))
        Q2_X = db.load_pulse("Q2_X90", variables=dict(width="rabi_width"))

        rabi_se = SequenceElement()
        rabi_se.append(Q0_X)
        rabi_se.append(Q1_X)
        rabi_se.append(Q2_X)
        rabi_se.add_waveform(ReadoutMarker(), location=Q0_X.width)
        ro_se = SequenceElement()

        ts = np.linspace(0, 34.8e-9, 21)
        seq = Sequence.sweep(rabi_se, rabi_width=ts)
        cseq = qpu_01.sequencer.compile(seq, readout=ro_se)

        return cseq

    ## Tests

    def test_update_parameters(self, qpu_01, sim_backend):
        sim_backend.update_parameters(qpu_01)
        a, adag = qt.destroy(4), qt.create(4)

        assert sim_backend.num_levels == 4
        for i in range(8):
            assert sim_backend.channel_map[i] == OperatorChannelMap(
                target=f"Q{i}",
                operator=a + adag,
                channels=(2 * i, 2 * i + 1),
                amplitude_factor=40e6,
                LO_frequency=5.8e9,
            )

            assert list(sim_backend.static_hamiltonian.keys()) == [
                f"Q{i}" for i in range(8)
            ]

        # TODO: test error thrown correctly

    @pytest.mark.parametrize(
        "data,expect",
        [
            (np.zeros((6, 5, 25, 4)), np.array([])),
            (np.ones((4, 5, 3)), np.array([0, 1, 2, 3])),
        ],
    )
    def test_get_active_channels(
        self,
        data,
        expect,
        compile_Q0X90,
        compile_Q0X90_Q1X90,
        compile_Q0X90_Q1X90_Q2X90,
    ):
        assert_array_equal(get_active_channels(data), expect)
        assert_array_equal(get_active_channels(compile_Q0X90.array[:, 1, ...]), [0, 1])
        assert_array_equal(
            get_active_channels(compile_Q0X90_Q1X90.array[:, 1, ...]), [0, 1, 2, 3]
        )
        assert_array_equal(
            get_active_channels(compile_Q0X90_Q1X90_Q2X90.array[:, 1, ...]),
            [0, 1, 2, 3, 4, 5],
        )

    def test_upload_empty_seq(self, compiled, qpu_01, sim_backend):
        cseq = compiled(20)
        sim_backend.update_parameters(qpu_01)

        with pytest.raises(ValueError):
            sim_backend.upload(cseq)

    @pytest.mark.skip
    def test_upload(
        self,
        qpu_01,
        sim_backend,
        compile_Q0X90,
        compile_Q0X90_Q1X90,
        compile_Q0X90_Q1X90_Q2X90,
    ):
        sim_backend.update_parameters(qpu_01)

        sim_backend.upload(compile_Q0X90)
        assert len(sim_backend.H) == 20
        assert len(sim_backend.H[0].H) == 2

        sim_backend.upload(compile_Q0X90_Q1X90)
        assert len(sim_backend.H[0].H) == 4

        sim_backend.upload(compile_Q0X90_Q1X90_Q2X90)
        assert len(sim_backend.H[0].H) == 6

    def test_aqcuire_Q0X90_seq(self, qpu_01, sim_backend, compile_Q0X90, data_file):
        expected = np.loadtxt(str(data_file), delimiter=",")

        sim_backend.update_parameters(qpu_01)
        sim_backend.upload(compile_Q0X90)
        results = sim_backend.acquire(compile_Q0X90, False)[0]

        assert_allclose(expected, results.expect)

    def test_acquire_Q0X180_seq(self, qpu_01, sim_backend, compile_Q0_X180, data_file):
        expected = np.loadtxt(str(data_file), delimiter=",")

        sim_backend.update_parameters(qpu_01)
        sim_backend.upload(compile_Q0_X180)
        results = sim_backend.acquire(compile_Q0_X180, False)[0]

        assert_allclose(expected, results.expect)

    # Multiple channel pairs on
    @pytest.mark.skip
    def test_acquire_Q0X90_Q1X90_seq(self, qpu_01, sim_backend, compile_Q0X90_Q1X90):
        sim_backend.update_parameters(qpu_01)
        sim_backend.upload(compile_Q0X90_Q1X90)
        results = sim_backend.acquire(compile_Q0X90_Q1X90, False)[0]

        H_full_seq = sim_backend.H[-1]
        psis = H_full_seq.get_basis()
        plot_states(results, H_full_seq.ts, list(psis.keys()))

    @pytest.mark.skip
    def test_acquire_Q0X90_Q1X90_Q2X90_seq(
        self, qpu_01, sim_backend, compile_Q0X90_Q1X90_Q2X90
    ):
        sim_backend.update_parameters(qpu_01)
        sim_backend.upload(compile_Q0X90_Q1X90_Q2X90)
        results = sim_backend.acquire(compile_Q0X90_Q1X90_Q2X90, False)[0]

        H_full_seq = sim_backend.H[-1]
        psis = H_full_seq.get_basis()
        plot_states(results, H_full_seq.ts, list(psis.keys()))
