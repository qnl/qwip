import cirq
import numpy as np
import pytest
import sympy as sym

from qwip.backends import DummyBackend
from qwip.interfaces.cirq import (
    CirqSampler,
    CirqSerializer,
    CirqTranspiler,
    LocalPulseLoader,
    qid_to_name,
)
from qwip.qpu import QPU
from qwip.sequencer import (
    CosineRampWaveform,
    CWWaveform,
    ModulatedWaveform,
    SquareWaveform,
    Timeline,
    VirtualZWaveform,
)


@pytest.mark.parametrize(
    "qid,name",
    [
        (cirq.NamedQubit("Q0"), "Q0"),
        (cirq.NamedQid("Q0", dimension=3), "Q0"),
        (cirq.LineQubit(0), "q0"),
        (cirq.LineQid(0, dimension=3), "q0"),
        (cirq.GridQubit(0, 1), "q0_1"),
        (cirq.GridQid(0, 1, dimension=3), "q0_1"),
    ],
)
def test_qid_to_name(qid, name):
    assert qid_to_name(qid) == name


class TestCirqTranspiler:
    @pytest.fixture
    def transpiler(self) -> CirqTranspiler:
        qubit_xy = Timeline.from_layers(
            [
                VirtualZWaveform(channel="drive", frame="frame", phase="zphase"),
                ModulatedWaveform(
                    envelope=CosineRampWaveform(
                        width="width",
                        ramp="ramp",
                    ),
                    modulation=CWWaveform(
                        channel="drive",
                        amplitude="amplitude",
                        frequency="frame",
                        phase="phase",
                        hardware_modulation=True,
                    ),
                ),
                VirtualZWaveform(channel="drive", frame="frame", phase="zphase"),
            ],
        )

        qubit_z = Timeline.from_layers(
            [VirtualZWaveform(channel="drive", frame="frame", phase="phase")]
        )

        readout = Timeline.from_layers(
            [
                [
                    ModulatedWaveform(
                        envelope=SquareWaveform(width="width"),
                        modulation=CWWaveform(
                            channel="drive",
                            amplitude="amplitude",
                            frequency="frequency",
                            hardware_modulation=True,
                        ),
                    ),
                    ModulatedWaveform(
                        envelope=SquareWaveform(width="width"),
                        modulation=CWWaveform(
                            channel="demod",
                            frequency="frequency",
                            hardware_modulation=True,
                        ),
                    ),
                ]
            ]
        )

        pulse_map = {}
        for i in range(3):
            pulse_map[f"Q{i}_X90"] = (
                qubit_xy,
                dict(
                    width=25e-9,
                    ramp=10e-9,
                    amplitude=0.1,
                    frame=f"Q{i}.freq_01",
                    phase=0,
                    zphase=0,
                ),
                dict(drive=f"CH{i}.qdrv"),
            )

            pulse_map[f"Q{i}_Z"] = (
                qubit_z,
                dict(frame=f"Q{i}.freq_01"),
                dict(drive=f"CH{i}.qdrv"),
            )

            pulse_map[f"Q{i}_measure"] = (
                readout,
                dict(
                    width=1e-6,
                    amplitude=0.1,
                    frequency=f"R{i}.freq_01",
                ),
                dict(drive=f"CH{i}.rdrv", demod=f"CH{i}.rdlo"),
            )

        return CirqTranspiler(loader=LocalPulseLoader(pulse_map=pulse_map))

    def test_pi_pulse(self, transpiler):
        q0 = cirq.NamedQubit("Q0")

        circuit = cirq.Circuit(
            [(cirq.X**0.5).on(q0), cirq.XPowGate(exponent=0.5).on(q0), cirq.measure(q0)]
        )

        tmln = transpiler.circuit_to_timeline(circuit)

        X90 = transpiler.loader.load_pulse("Q0_X90")
        readout = transpiler.loader.load_pulse("Q0_measure")

        expect = Timeline.from_layers([X90, X90, readout])
        assert tmln == expect

    def test_loschmidt_echo(self, transpiler):
        q0 = cirq.NamedQubit("Q0")

        circuit = cirq.Circuit(
            [
                (cirq.X**0.5).on(q0),
                cirq.PhasedXPowGate(exponent=0.5, phase_exponent=1).on(q0),
                cirq.measure(q0),
            ]
        )

        tmln = transpiler.circuit_to_timeline(circuit)

        X90 = transpiler.loader.load_pulse("Q0_X90")
        X90_r = transpiler.loader.load_pulse("Q0_X90", dict(phase=180))
        readout = transpiler.loader.load_pulse("Q0_measure")

        expect = Timeline.from_layers([X90, X90_r, readout])
        assert tmln == expect

    def test_ramsey(self, transpiler):
        q0 = cirq.NamedQubit("Q0")

        circuit = cirq.Circuit(
            [
                (cirq.X**0.5).on(q0),
                cirq.wait(q0, nanos=500),
                (cirq.Z**0.25).on(q0),
                (cirq.Y**0.5).on(q0),
            ]
        )

        tmln = transpiler.circuit_to_timeline(circuit)

        X90 = transpiler.loader.load_pulse("Q0_X90")
        Y90 = transpiler.loader.load_pulse("Q0_X90", dict(phase=90))
        Z = transpiler.loader.load_pulse("Q0_Z", dict(phase=45))

        expect = Timeline.from_layers([X90, 500 * 1e-9, Z, Y90])
        assert tmln == expect

    def test_T1(self, transpiler):
        q0 = cirq.NamedQubit("Q0")

        circuit = cirq.Circuit(
            [
                (cirq.X**0.5).on(q0),
                (cirq.X**0.5).on(q0),
                cirq.wait(q0, micros=sym.Symbol("time")),
                cirq.measure(q0),
            ]
        )

        tmln = transpiler.circuit_to_timeline(circuit)

        X90 = transpiler.loader.load_pulse("Q0_X90")
        readout = transpiler.loader.load_pulse("Q0_measure")

        expect = Timeline.from_layers([X90, X90, "time * 1000 * 1e-9", readout])
        assert tmln == expect

    def test_phased_XZ(self, transpiler):
        q0 = cirq.NamedQubit("Q0")

        circuit = cirq.Circuit(
            [
                cirq.PhasedXZGate(
                    x_exponent=1, z_exponent=-1, axis_phase_exponent=0
                ).on(q0)
            ]
        )

        tmln = transpiler.circuit_to_timeline(circuit)


class TestCirqSampler:
    @pytest.fixture
    def transpiler(self) -> CirqTranspiler:
        qubit_xy = Timeline.from_layers(
            [
                VirtualZWaveform(channel="drive", frame="frame", phase="zphase"),
                ModulatedWaveform(
                    envelope=CosineRampWaveform(
                        width="width",
                        ramp="ramp",
                    ),
                    modulation=CWWaveform(
                        channel="drive",
                        amplitude="amplitude",
                        frequency="frame",
                        phase="phase",
                        hardware_modulation=True,
                    ),
                ),
                VirtualZWaveform(channel="drive", frame="frame", phase="zphase"),
            ],
        )

        qubit_z = Timeline.from_layers(
            [VirtualZWaveform(channel="drive", frame="frame", phase="phase")]
        )

        readout = Timeline.from_layers(
            [
                [
                    ModulatedWaveform(
                        envelope=SquareWaveform(width="width"),
                        modulation=CWWaveform(
                            channel="drive",
                            amplitude="amplitude",
                            frequency="frequency",
                            hardware_modulation=True,
                        ),
                    ),
                    ModulatedWaveform(
                        envelope=SquareWaveform(width="width"),
                        modulation=CWWaveform(
                            channel="demod",
                            frequency="frequency",
                            hardware_modulation=True,
                        ),
                    ),
                ]
            ]
        )

        pulse_map = {}
        for i in range(3):
            pulse_map[f"Q{i}_X90"] = (
                qubit_xy,
                dict(
                    width=25e-9,
                    ramp=10e-9,
                    amplitude=0.1,
                    frame=f"Q{i}.freq_01",
                    phase=0,
                    zphase=0,
                ),
                dict(drive=f"CH{i}.qdrv"),
            )

            pulse_map[f"Q{i}_Z"] = (
                qubit_z,
                dict(frame=f"Q{i}.freq_01"),
                dict(drive=f"CH{i}.qdrv"),
            )

            pulse_map[f"Q{i}_measure"] = (
                readout,
                dict(
                    width=1e-6,
                    amplitude=0.1,
                    frequency=f"R{i}.freq_01",
                ),
                dict(drive=f"CH{i}.rdrv", demod=f"CH{i}.rdlo"),
            )

        return CirqTranspiler(loader=LocalPulseLoader(pulse_map=pulse_map))

    @pytest.fixture
    def sampler(self, transpiler):
        qpu = QPU()

        return CirqSampler(qpu=qpu, transpiler=transpiler)

    # def test_sweep(self, sampler):
    #     print(sampler)

    # def test_circuit(self, transpiler):
    #     q0, q1, q2 = (cirq.NamedQubit(f"Q{i}") for i in range(3))

    #     circuit = cirq.Circuit(
    #         [
    #             cirq.wait(q0, q1, micros=sym.Symbol("wait")),
    #             (cirq.X**0.5).on(q0),
    #             cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=0.5).on(q1),
    #             cirq.Rx(rads=np.pi / 2).on(q2),
    #             cirq.wait(q0, q1, micros=sym.Symbol("wait")),
    #             cirq.measure(q0, q1),
    #         ]
    #     )

    #     x = cirq.X.on(q0)
    #     x = cirq.Rx(rads=np.pi).on(q0)

    #     print(isinstance(cirq.Rx(rads=np.pi), cirq.XPowGate))
    #     print(x.qubits, x.gate, type(x.gate))
    #     print(type(cirq.X.on(q0)))
    #     transpiler.circuit_to_pulse_layers(circuit)


class TestCirqSerializer:
    @pytest.fixture
    def serializer(self):
        return CirqSerializer()

    def test_circuit(self, serializer):
        *data_qs, meas_q = (
            cirq.GridQubit(*rc) for rc in [(0, 1), (1, 0), (1, 2), (2, 1), (1, 1)]
        )

        circuit = (
            cirq.Circuit([cirq.H.on(q) for q in data_qs], cirq.H.on(meas_q))
            + cirq.Circuit([cirq.CZ.on(q, meas_q) for q in data_qs])
            + cirq.Circuit([cirq.H.on(q) for q in data_qs], cirq.H.on(meas_q))
            + cirq.Circuit(cirq.measure(meas_q))
        )

        stream = serializer.to_stream(circuit, fmt="json")
        reloaded = serializer.from_stream(stream, fmt="json")

        assert circuit == reloaded
        assert stream.closed

    @pytest.mark.parametrize(
        "qid", [cirq.GridQubit(0, 0), cirq.LineQubit(10), cirq.NamedQid("QA0", 3)]
    )
    def test_qid(self, serializer, qid):
        stream = serializer.to_stream(qid, fmt="json")
        reloaded = serializer.from_stream(stream, fmt="json")

        assert qid == reloaded
        assert stream.closed
