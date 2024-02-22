import itertools as it

import cirq
import numpy as np
import pytest
from cirq.experiments import random_quantum_circuit_generation as rqcg

from qwip.compilation.cirq import CirqTranspiler

exponents = np.arange(8) / 4
SINGLE_QUBIT_GATES = [
    cirq.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)
    for a, z in it.product(exponents, repeat=2)
]


def test_circuit():
    q0, q1 = cirq.GridQubit.rect(1, 2)

    circuit = rqcg.random_rotations_between_two_qubit_circuit(
        q0,
        q1,
        depth=4,
        two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b),
        single_qubit_gates=SINGLE_QUBIT_GATES,
    )

    transpiler = CirqTranspiler()

    transpiler.transpile(circuit)
