import codecs
import io
from functools import singledispatchmethod
from typing import Protocol, runtime_checkable

import cirq
import numpy as np
import pandas as pd
from attrs import field
from cirq import AbstractCircuit, Sampler, devices, study

from qwip.attrs import qdefine, qfrozen
from qwip.data.serializers import (
    SERIALIZERS,
    Serializer,
    detect_serializer,
    register_serializer,
)
from qwip.processing.processors import MeasurementResult, ReadoutBitstring
from qwip.qpu.qpu import QPU
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.timeline import Timeline
from qwip.sequencer.waveform import Operation

PulseLike = Timeline | Operation | str


def qid_to_name(qid: cirq.Qid):
    match qid:
        case cirq.NamedQubit(name=name):
            return name
        case cirq.NamedQid(name=name):
            return name
        case cirq.LineQubit(x=x):
            return f"q{x}"
        case cirq.LineQid(x=x):
            return f"q{x}"
        case cirq.GridQubit(row=r, col=c):
            return f"q{r}_{c}"
        case cirq.GridQid(row=r, col=c):
            return f"q{r}_{c}"
        case cirq.Qid():
            return str(qid)
        case _:
            raise ValueError("Not a valid qid!")


@runtime_checkable
class PulseLoader(Protocol):
    def load_pulse(self, name: str, variables: dict = {}, **kwargs): ...


@qdefine
class LocalPulseLoader:
    pulse_map: dict[str, tuple[PulseLike, dict, dict]] = field(factory=dict)

    def load_pulse(self, name, variables: dict = {}, **kwargs):
        template, default_vars, channels = self.pulse_map[name]

        replacements = default_vars | variables
        tmln = template.copy()
        tmln.rename_variables(lambda s: replacements.get(s, s))
        tmln.assign_channels(**channels)

        return tmln


@qdefine
class CirqTranspiler:
    loader: PulseLoader = field(factory=LocalPulseLoader)
    pulse_cache: dict = field(factory=dict)

    def to_pulse(self, op: cirq.Operation) -> PulseLike:
        layers = []

        pulse_info = self.to_name_variables(op.gate, op)

        if not isinstance(pulse_info, list):
            pulse_info = [pulse_info]

        for layer_info in pulse_info:
            match layer_info:
                case dict():
                    layer = []
                    for name, variables in layer_info.items():
                        cache_key = (name, *variables.items())
                        if cache_key not in self.pulse_cache:
                            self.pulse_cache[cache_key] = self.loader.load_pulse(
                                name, variables
                            )
                        layer.append(self.pulse_cache[cache_key])
                    layers.append(layer[0] if len(layer) == 1 else layer)
                case duration:
                    layers.append(duration)

        if len(layers) == 0:
            return None
        elif len(layers) == 1:
            return layers[0]
        else:
            return Timeline.from_layers(layers)

    @singledispatchmethod
    def to_name_variables(self, gate: cirq.Gate, op: cirq.Operation, **kwargs) -> list:
        print("Default", gate, op.qubits)

    @to_name_variables.register
    def _(self, gate: cirq.XPowGate, op: cirq.Operation):
        if gate.exponent != 0.5:
            raise ValueError(f"Only X^0.5 gates are currently supported, got {gate}!")

        return {f"{qid_to_name(op.qubits[0])}_X90": {}}

    @to_name_variables.register
    def _(self, gate: cirq.YPowGate, op: cirq.Operation):
        if gate.exponent != 0.5:
            raise ValueError(f"Only Y^0.5 gates are currently supported, got {gate}!")

        return {f"{qid_to_name(op.qubits[0])}_X90": dict(phase=90)}

    @to_name_variables.register
    def _(self, gate: cirq.PhasedXPowGate, op: cirq.Operation):
        if gate.exponent != 0.5:
            raise ValueError(f"Only X^0.5 gates are currently supported, got {gate}!")

        phase = -gate.phase_exponent * 180

        return {f"{qid_to_name(op.qubits[0])}_X90": dict(phase=phase)}

    @to_name_variables.register
    def _(self, gate: cirq.PhasedXZGate, op: cirq.Operation):
        if gate.x_exponent % 0.5 != 0:
            raise ValueError(
                f"Only multiples of X^0.5 gates are currently supported got {gate}!"
            )

        phase = -gate.axis_phase_exponent * 180
        if gate.x_exponent < 0:
            phase = (phase + 180) % 360
        zphase = gate.z_exponent * 180

        n = np.abs(gate.x_exponent / 0.5).astype(int)
        rotation = [{f"{qid_to_name(op.qubits[0])}_X90": dict(phase=phase)}] * n

        return rotation + [
            {f"{qid_to_name(op.qubits[0])}_Z": dict(phase=zphase)},
        ]

    @to_name_variables.register
    def _(self, gate: cirq.ZPowGate, op: cirq.Operation):
        return {f"{qid_to_name(op.qubits[0])}_Z": dict(phase=gate.exponent * 180)}

    @to_name_variables.register
    def _(self, gate: cirq.WaitGate, op: cirq.Operation):
        match gate.duration:
            case cirq.Duration():
                wait = gate.duration.total_nanos() * 1e-9
            case _:
                raise NotImplementedError(
                    f"Duration of type {type(gate.duration)} is not supported."
                )

        return wait

    @to_name_variables.register
    def _(self, gate: cirq.MeasurementGate, op: cirq.Operation):
        return {f"{qid_to_name(qid)}_measure": {} for qid in op.qubits}

    def update_registers(self, op: cirq.Operation, registers: dict):
        mkey = ",".join(qid_to_name(qid) for qid in sorted(op.qubits))
        registers[mkey] = ReadoutBitstring

    def circuit_to_timeline(
        self, circuit, registers: dict | None = None, clear_cache: bool = True
    ):
        if registers is None:
            registers = {}

        if clear_cache:
            self.pulse_cache.clear()

        layers = []
        for moment in circuit:
            layer = []
            for op in moment:
                pulse = self.to_pulse(op)
                if isinstance(pulse, list):
                    layer.extend(pulse)
                else:
                    layer.append(pulse)

                if isinstance(op.gate, cirq.MeasurementGate):
                    self.update_registers(op, registers)

            layers.append(layer)

        return Timeline.from_layers(layers)


@qdefine
class CirqSampler(Sampler):
    qpu: QPU
    transpiler: CirqTranspiler
    device: devices.Device | None = None
    batch: dict = field(factory=dict)
    compilation: dict = field(factory=dict)
    data: dict = field(factory=dict)

    def run_sweep(
        self, program: AbstractCircuit, params: study.Sweepable, repetitions: int = 1
    ) -> list[study.Result]:
        if self.device:
            self.device.validate_circuit(program)

        registers = {}
        tmln = self.transpiler.circuit_to_timeline(program, registers)
        assets = dict(circuit=program)

        if params is None:
            seq = Sequence([tmln])
        else:
            params_df = pd.DataFrame(p.param_dict for p in study.to_resolvers(params))
            idx = pd.MultiIndex.from_frame(params_df)
            seq = Sequence.sweep(
                tmln, **{name: idx.get_level_values(name) for name in idx.names}
            )

            assets["parameters"] = params_df

        qpu_result = self.qpu.run(
            seq,
            registers,
            repetitions=repetitions,
            batch=self.batch,
            compilation=self.compilation,
            data=assets | self.data,
        )

        return self.format_cirq_result(program, params, qpu_result)

    def format_cirq_result(
        self,
        program: AbstractCircuit,
        params: study.Sweepable,
        qpu_result: dict[str, MeasurementResult],
    ) -> list[study.Result]:
        shapes = self._get_measurement_shapes(program)

        result_dicts = []
        for i, p in enumerate(study.to_resolvers(params)):
            records = {}

            for k, qr in qpu_result.items():
                df = qr.d.xs(i, level="timeline").map(
                    lambda bitstring: [int(b) for b in bitstring]
                )

                repetitions, readouts = df.index.levshape
                _, qid_shape = shapes[k]

                records[k] = (
                    df.explode("state")
                    .to_numpy()
                    .reshape((repetitions, readouts, len(qid_shape)))
                )

            result_dicts.append(study.ResultDict(params=p, records=records))

        return result_dicts


@qfrozen
class CirqSerializer(Serializer):
    """A serializer for cirq types"""

    @property
    def formats(self) -> tuple[str, ...]:
        return ("json",)

    def to_stream_json(
        self,
        cirq_obj: cirq.SupportsJSON,
        **kwargs,
    ) -> io.BufferedReader:
        stream = codecs.getwriter("utf-8")(io.BytesIO())

        kwargs["format"] = "json"
        cirq.to_json(cirq_obj, stream, indent=4)
        stream.seek(0)

        return stream

    def from_stream_json(self, stream: io.BufferedReader):
        return cirq.read_json(stream)


@detect_serializer.register(cirq.Circuit)
@detect_serializer.register(cirq.Qid)
@detect_serializer.register(cirq.Gate)
@detect_serializer.register(cirq.Operation)
def detect_cirq(obj: cirq.SupportsJSON):
    return SERIALIZERS["cirq"]


register_serializer(CirqSerializer())
