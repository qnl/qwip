import attrs
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import qwip
from qwip.backends.qubic import (
    BarrierInstruction,
    DelayInstruction,
    PulseInstruction,
    QubicInstruction,
)
from qwip.sequencer import (
    CWWaveform,
    GaussianWaveform,
    ModulatedWaveform,
    SequenceElement,
)


class TestQubicInstruction:
    @pytest.mark.parametrize(
        "ins,expect",
        [
            (
                PulseInstruction(
                    env=np.linspace(0, 200, 100),
                    dest="Q0.drv",
                    freq=5e9,
                    twidth=12.5e-9,
                ),
                dict(
                    name="pulse",
                    env=np.linspace(0, 200, 100),
                    dest="Q0.drv",
                    freq=5e9,
                    phase=0,
                    amp=1,
                    twidth=12.5e-9,
                ),
            ),
            (DelayInstruction(t=0.0005), dict(name="delay", t=0.0005)),
        ],
    )
    def test_todict(self, ins, expect):
        d = ins.todict()

        assert set(d.keys()) == set(expect.keys())

        for k in d.keys():
            match k:
                case "env":
                    assert_array_equal(d[k], expect[k])
                case _:
                    assert d[k] == expect[k]


class TestPulseInstruction:
    @pytest.mark.parametrize(
        "p1,p2,expect",
        [
            (
                PulseInstruction(env=np.zeros(10), dest="Q0.drv", freq=0),
                PulseInstruction(env=np.zeros(10), dest="Q0.drv", freq=0),
                False,
            ),
            (
                p := PulseInstruction(env=np.ones(21), dest="Q0.drv", freq=1e9),
                PulseInstruction(env=p.env, dest="Q0.drv", freq=1e9),
                True,
            ),
        ],
    )
    def test_equality_and_hash(self, p1, p2, expect):
        assert (p1 == p2) is expect
        assert (hash(p1) == hash(p2)) is expect


class TestQubicCompiler:
    def test_compile_instruction(self):
        ...

    def test_compile_sequence_element(self):
        ...
