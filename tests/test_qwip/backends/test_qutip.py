import matplotlib.pyplot as plt
import numpy as np
import pytest
import qutip as qt
from numpy.testing import assert_allclose, assert_array_equal

from qwip.analysis.frequency import simple_fft
from qwip.backends.qutip import (
    OperatorChannelMap,
    QutipBackend,
    TimeDependentHamiltonian,
    get_active_channels,
)
from qwip.processing.processors import HeterodyneDemodulation, IQTraceResult
from qwip.sequencer import Sequence, Timeline, TriggeredWaveform


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
    return fig


def plot_multi_qubit_states(results, ts, labels):
    labeled = dict(zip(labels, results.expect))

    N_sys = len(labels[0])

    qdicts = [dict() for _ in range(N_sys)]
    for q in range(N_sys):
        for state, populations in labeled.items():
            if state[q] not in qdicts[q]:
                qdicts[q][state[q]] = np.array(populations)
            else:
                qdicts[q][state[q]] += populations

    fig, axes = plt.subplots(N_sys, 1)

    for q, ax in enumerate(axes):
        for s, pops in qdicts[q].items():
            ax.plot(ts, pops, label=f"$|{s}⟩$")
        ax.legend()
        ax.set_ylabel(f"Q{q}")

    return fig


@pytest.mark.xfail
class TestTimeDependentHamiltonian:
    @pytest.mark.parametrize(
        "H,shape",
        [
            ([], None),
            ([qt.sigmax()], (2, 2)),
            ([qt.tensor(qt.sigmax(), qt.destroy(5))], (10, 10)),
        ],
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
            ([qt.tensor(qt.sigmax(), qt.destroy(5))], [[2, 5], [2, 5]]),
        ],
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
                    expected = 0.5 * np.cos(2 * np.pi * 1e9 * H_t.ts) + 0.5
                case (1,):
                    expected = -0.5 * np.cos(2 * np.pi * 1e9 * H_t.ts) + 0.5
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
                    expected = 0.5 * np.cos(2 * np.pi * 1e9 * H_t.ts) + 0.5
                case (1, 0, 0):
                    expected = -0.5 * np.cos(2 * np.pi * 1e9 * H_t.ts) + 0.5
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
        H1 = TimeDependentHamiltonian(H=[(qt.sigmax(), np.zeros_like(ts))], ts=ts)

        assert TimeDependentHamiltonian.tensor(H0, H1) == H1

    def test_tensor(self):
        ts = np.linspace(0, 10)
        H0 = TimeDependentHamiltonian(
            H=[(qt.sigmaz(), np.ones_like(ts)), (qt.sigmax(), np.cos(2 * np.pi * ts))],
            ts=ts,
            targets=("Q0",),
        )

        H1 = TimeDependentHamiltonian(
            H=[(qt.create(3) * qt.destroy(3), 2 * np.ones_like(ts))],
            ts=ts,
            targets=("Q1",),
        )

        expect = TimeDependentHamiltonian(
            H=[
                (qt.tensor(qt.sigmaz(), qt.qeye(3)), np.ones_like(ts)),
                (qt.tensor(qt.sigmax(), qt.qeye(3)), np.cos(2 * np.pi * ts)),
                (
                    qt.tensor(qt.qeye(2), qt.create(3) * qt.destroy(3)),
                    2 * np.ones_like(ts),
                ),
            ],
            ts=ts,
            targets=("Q0", "Q1"),
        )

        assert TimeDependentHamiltonian.tensor(H0, H1) == expect


@pytest.mark.xfail
class TestQutipBackend:
    @pytest.fixture
    def compiler(self, qpu_01):
        return qpu_01.compiler

    @pytest.fixture
    def exe(self, compiler):
        def make_exe(shape):
            seq = Sequence.empty(shape)
            readout = TriggeredWaveform(
                target=Timeline(), width=50e-9, channel="RO_marker"
            )
            return compiler.compile(seq + Timeline().add_waveform(readout))

        return make_exe

    @pytest.fixture
    def sim_backend(self):
        backend = QutipBackend()
        return backend

    @pytest.fixture
    def readout_fields(self, sim_backend, qpu_01):
        sim_backend.update_parameters(qpu_01)

        def drive_envelope(t):
            return 1e7

        ts = np.linspace(0, 1e-5, 1000)

        readout_fields = {
            target: R.solve_cavity_field_equation(ts, drive_envelope)
            for target, R in sim_backend.readouts.items()
        }
        return readout_fields

    ## Pulse sequence fixtures

    # One Channel
    @pytest.fixture
    def exe_Q0X90(self, qpu_01):
        db = qpu_01.db
        Q0_X = db.load_pulse("Q0_X90", variables=dict(width="rabi_width"))

        readout = TriggeredWaveform(target=Timeline(), width=50e-9, channel="RO_marker")
        rabi_se = Timeline()
        rabi_se.add(Q0_X)
        rabi_se.add_waveform(readout, location=Q0_X.width)

        ts = np.linspace(0, 34.8e-9, 21)
        seq = Sequence.sweep(rabi_se, rabi_width=ts)
        exe = qpu_01.compiler.compile(seq)

        return exe

    @pytest.fixture
    def exe_Q0_X180(self, qpu_01):
        db = qpu_01.db
        Q0_X = db.load_pulse("Q0_X90", variables=dict(width="rabi_width"))

        readout = TriggeredWaveform(target=Timeline(), width=50e-9, channel="RO_marker")
        rabi_se = Timeline()
        rabi_se.add(Q0_X)
        rabi_se.add(Q0_X, self_loc=Q0_X.width)
        rabi_se.add_waveform(readout, location=2 * Q0_X.width)

        ts = np.linspace(0, 34.8e-9, 21)
        seq = Sequence.sweep(rabi_se, rabi_width=ts)
        exe = qpu_01.compiler.compile(seq)

        return exe

    # Two channels
    @pytest.fixture
    def exe_Q0X90_Q1X90(self, qpu_01):
        db = qpu_01.db
        Q0_X = db.load_pulse("Q0_X90", variables=dict(width="rabi_width"))
        Q1_X = db.load_pulse("Q1_X90", variables=dict(width="rabi_width"))

        readout = TriggeredWaveform(target=Timeline(), width=50e-9, channel="RO_marker")
        rabi_se = Timeline()
        rabi_se.add(Q0_X)
        rabi_se.add(Q1_X)
        rabi_se.add_waveform(readout, location=Q0_X.width)

        ts = np.linspace(0, 34.8e-9, 21)
        seq = Sequence.sweep(rabi_se, rabi_width=ts)
        exe = qpu_01.compiler.compile(seq)

        return exe

    # Three channels
    @pytest.fixture
    def exe_Q0X90_Q1X90_Q2X90(self, qpu_01):
        db = qpu_01.db
        Q0_X = db.load_pulse("Q0_X90", variables=dict(width="rabi_width"))
        Q1_X = db.load_pulse("Q1_X90", variables=dict(width="rabi_width"))
        Q2_X = db.load_pulse("Q2_X90", variables=dict(width="rabi_width"))

        readout = TriggeredWaveform(target=Timeline(), width=50e-9, channel="RO_marker")
        rabi_se = Timeline()
        rabi_se.add(Q0_X)
        rabi_se.add(Q1_X)
        rabi_se.add(Q2_X)
        rabi_se.add_waveform(readout, location=Q0_X.width)

        ts = np.linspace(0, 34.8e-9, 21)
        seq = Sequence.sweep(rabi_se, rabi_width=ts)
        exe = qpu_01.compiler.compile(seq)

        return exe

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
        exe_Q0X90,
        exe_Q0X90_Q1X90,
        exe_Q0X90_Q1X90_Q2X90,
    ):
        assert_array_equal(get_active_channels(data), expect)
        assert_array_equal(get_active_channels(exe_Q0X90.array[:, 1, ...]), [0, 1])
        assert_array_equal(
            get_active_channels(exe_Q0X90_Q1X90.array[:, 1, ...]), [0, 1, 2, 3]
        )
        assert_array_equal(
            get_active_channels(exe_Q0X90_Q1X90_Q2X90.array[:, 1, ...]),
            [0, 1, 2, 3, 4, 5],
        )

    def test_upload_empty_seq(self, exe, qpu_01, sim_backend):
        exe = exe(20)
        sim_backend.update_parameters(qpu_01)

        with pytest.raises(ValueError):
            sim_backend.upload(exe)

    def test_upload(
        self,
        qpu_01,
        sim_backend,
        exe_Q0X90,
        exe_Q0X90_Q1X90,
        exe_Q0X90_Q1X90_Q2X90,
    ):
        sim_backend.update_parameters(qpu_01)

        sim_backend.upload(exe_Q0X90)
        assert len(sim_backend.H) == 21
        assert len(sim_backend.H[1].H) == 2
        assert sim_backend.H[1].targets == ("Q0",)

        sim_backend.upload(exe_Q0X90_Q1X90)
        assert len(sim_backend.H[1].H) == 4
        assert sim_backend.H[1].targets == ("Q0", "Q1")

        sim_backend.upload(exe_Q0X90_Q1X90_Q2X90)
        assert len(sim_backend.H[1].H) == 6
        assert sim_backend.H[1].targets == ("Q0", "Q1", "Q2")

    def test_upload_readouts(self, qpu_01, sim_backend):
        sim_backend.update_parameters(qpu_01)

        assert len(sim_backend.readouts) == 8
        assert sim_backend.readouts["R0"].chi == (-1e6, 1e6)

    def test_acquire_Q0X90_seq(self, qpu_01, sim_backend, exe_Q0X90):
        sim_backend.update_parameters(qpu_01)
        sim_backend.upload(exe_Q0X90)

        results1 = sim_backend.acquire(exe_Q0X90)
        assert list(results1.keys()) == [f"Q{i}" for i in range(8)]
        for fields in results1.values():
            assert fields.index.levshape == (512, 1, 1, 1000)

        results2 = sim_backend.acquire(exe_Q0X90, elements=[-2, -1])
        for fields in results2.values():
            assert fields.index.levshape == (512, 2, 1, 1000)

        results_3 = sim_backend.acquire(exe_Q0X90, elements=[5, -2, -1])
        for fields in results_3.values():
            assert fields.index.levshape == (512, 3, 1, 1000)

    def test_acquire_Q0X90_blobs(self, qpu_01, sim_backend, exe_Q0X90):
        sim_backend.update_parameters(qpu_01)
        sim_backend.upload(exe_Q0X90)

        raw_IQ = sim_backend.acquire(exe_Q0X90)
        processed_IQ_df = dict()

        freqs = np.array([0])
        ts = np.linspace(0, 1e-5, 1000)
        weights = {
            f"Q{i}": np.exp(-1j * 2 * np.pi * ts * freqs[i]) for i in range(len(freqs))
        }

        processor = HeterodyneDemodulation(weights=weights)

        for ch, data in raw_IQ.items():
            processed_IQ_df[ch] = processor.run(data)

        IQ_Q0 = processed_IQ_df["Q0"][0].data.to_numpy()[0]

        I_Q0 = np.real(IQ_Q0)
        num_excited = np.sum(np.array(I_Q0) <= 0, axis=0)
        num_ground = len(IQ_Q0) - num_excited

        assert abs(num_excited - num_ground) / len(IQ_Q0) < 0.2

    def test_acquire_Q0X180_seq(self, qpu_01, sim_backend, exe_Q0_X180):
        """Tests uneven distribution.

        Plots field amplitudes with strong drive to observe "Q0" bias towards excited
        state blob on the left
        """
        sim_backend.update_parameters(qpu_01)
        sim_backend.upload(exe_Q0_X180)

        raw_IQ = sim_backend.acquire(exe_Q0_X180, drive=10e6)
        processed_IQ_df = dict()

        freqs = np.array([0])
        ts = np.linspace(0, 1e-5, 1000)
        weights = {
            f"Q{i}": np.exp(-1j * 2 * np.pi * ts * freqs[i]) for i in range(len(freqs))
        }

        processor = HeterodyneDemodulation(weights=weights)

        for ch, data in raw_IQ.items():
            processed_IQ_df[ch] = processor.run(data)

        IQ_Q0 = processed_IQ_df["Q0"][0].data.to_numpy()[0]

        I_Q0 = np.real(IQ_Q0)
        num_excited = np.sum(np.array(I_Q0) <= 0, axis=0)

        assert num_excited / len(IQ_Q0) > 0.95
