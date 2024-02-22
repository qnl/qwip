import itertools as it
from functools import singledispatchmethod, lru_cache

import cirq

from loguru import logger

from attrs import field
from qwip.attrs import qdefine, qfrozen
from qwip.config.interface import ConfigDB
from qwip.sequencer import Sequence, Timeline, Waveform


@qdefine
class CirqTranspiler:
    gate_map: dict = field(factory=dict)
    
    def transpile(self, circuit: cirq.Circuit, clear_cache: bool = True) -> Timeline:
        if clear_cache:
            for mapper in self.gate_map.values():
                mapper.

        layers = []

        for i, moment in enumerate(circuit):
            new_layers = []
            for op in moment:
                t = self.convert_operation(op, i)
                if not isinstance(t, list):
                    t = [t]

                new_layers.append(t)

            for layer in it.zip_longest(*new_layers):
                layers.append([o for o in layer if o is not None])
        
        return Timeline.from_layers(layers)

    @singledispatchmethod
    def convert_operation(
        self, op: cirq.Operation, m_index: int
    ) -> Timeline | Waveform | None:
        logger.debug(f"{m_index} ({type(op)}): {op}")
        return None

    @convert_operation.register(cirq.GateOperation) 
    def _(self, op: cirq.Operation, m_index: int) -> Timeline | Waveform | None:
        return self.gate_to_pulse(op.gate, tuple(q.col for q in op.qubits))

    @singledispatchmethod
    def gate_to_pulse(
        self, gate: cirq.Gate, qubits: tuple[cirq.Qid, ...]
    ) -> Timeline | Waveform | None:
        logger.debug(f"{qubits}: {type(gate)}")
        return None

    @gate_to_pulse.register(cirq.PhasedXZGate)
    def phasedXZ_to_pulse(
        self, gate: cirq.Gate, qubits: tuple[cirq.Qid, ...]
    ) -> Timeline | Waveform | None:
        axis = gate.axis_phase_exponent * 180
        rotation = gate.x_exponent * 180
        z = gate.z_exponent * 180

        X = self.gate_map["X"](qubits, axis=-axis, rotation=rotation)
        Z = self.gate_map["Z"](qubits, phase=z)
        return [X, Z]

    @gate_to_pulse.register(cirq.ISwapPowGate)
    def iSWAPPow_to_pulse(
        self, gate: cirq.Gate, qubits: tuple[cirq.Qid, ...]
    ) -> Timeline | Waveform | None:
        rotation = gate.exponent * 180
        return self.gate_map["iSWAP"](qubits, rotation=rotation)

    @gate_to_pulse.register(cirq.MeasurementGate)
    def measure_to_pulse(
        self, gate: cirq.Gate, qubits: tuple[cirq.Qid, ...]
    ) -> Timeline | Waveform | None:
        return readout

@qfrozen
class GateMapper:
    db: ConfigDB = field(eq=id)
    name: str = ""

    @lru_cache
    def pulse_from_db(self, name):
        variables = list(self.db.config["pulses"][name].variables.keys())
        varmap = {}
        for f in attrs.fields(type(self)):
            fname = getattr(self, f.name)
            if fname in variables:
                varmap[fname] = f.name
        
        return self.db.load_pulse(name, variables=varmap)

    def transform_variables(self, variables):
        return variables

    def __call__(self, qubits, **kwargs):
        kwargs = self.transform_variables(kwargs)

        name = self.name.format(*qubits, **{getattr(self, k, k): v for k, v in kwargs.items()})
        tmln = self.pulse_from_db(name).copy()
        tmln.rename_variables(lambda v: kwargs.get(v, v))
        return tmln


@qfrozen
class PhasedXMapper(GateMapper):
    name: str = "Q{0}_X{rotation}_cos"
    axis: str = "phase"
    rotation: str = "rotation"

    def transform_variables(self, variables):
        if variables.get("rotation", 90) == 90:
            variables["rotation"] = 90
        else:
            raise ValueError("Only 90 degree X rotations are allowed.")

        return variables
        
@qfrozen
class iSWAPMapper(GateMapper):
    name: str = "Q{0}_Q{1}_{rotation}"

    def transform_variables(self, variables):
        if variables.get("rotation", 90) == 90:
            variables["rotation"] = "sqiSWAP"
        else:
            raise ValueError("Only 90 degree iSWAP rotations are allowed.")

        return variables

@qfrozen
class VirtualZMapper(GateMapper):
    name: str = "mod_Q{0}_{subspace}"
    phase: str = "phase"
    subspace: str = "subspace"

    def __call__(self, qubits, subspace="GE", **kwargs):
        kwargs = kwargs | dict(subspace=subspace)

        frame = self.name.format(*qubits, **{getattr(self, k, k): v for k, v in kwargs.items()})
        return VirtualZWaveform(
            name="Q{0}".format(*qubits), 
            frame=frame,
            phase=kwargs.get(self.phase, 0)
        )