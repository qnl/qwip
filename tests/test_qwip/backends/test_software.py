import numpy as np
import pytest
from numpy.testing import assert_array_equal

from qwip.backends.software import HeterodyneCompiler
from qwip.sequencer.compilation import (
    ChannelInfo,
    DeviceInfo,
    IntermediateProgram,
    PlayInstruction,
    TriggerInfo,
    WaitTriggerInstruction,
    WaveformMemory,
)


@pytest.fixture
def device():
    demod = DeviceInfo.from_channels(
        channels=(
            ChannelInfo("R0", 0),
            ChannelInfo("R1", 1),
            ChannelInfo("R2", 2),
            ChannelInfo("R3", 3),
        ),
        sample_rate=1.0e9,
        trigger=TriggerInfo(device="tektronix", index=3, subchannel=0),
        name="demod",
        dtype=np.complex64,
    )

    return demod


class TestHeterodyneCompiler:
    @pytest.fixture
    def single_demod(self, device):
        program = IntermediateProgram(device="demod")

        for _ in range(21):
            program.instructions.append(
                WaitTriggerInstruction(device="tektronix", index=3, subchannel=1)
            )
            program.instructions.append(PlayInstruction(waveform_index=0))

        wmem = WaveformMemory.from_channels(4096, 1.0e9, device.channels, device.dtype)
        for i in range(4):
            wmem[i][:] = i + i * 1j
        program.waveforms.append(wmem)

        return program

    @pytest.fixture
    def alternating_demod(self, device):
        program = IntermediateProgram(device="demod")

        for i in range(10):
            program.instructions.append(
                WaitTriggerInstruction(device="tektronix", index=3, subchannel=1)
            )
            program.instructions.append(PlayInstruction(waveform_index=i % 2))

        for w in range(2):
            wmem = WaveformMemory.from_channels(
                4096, 1.0e9, device.channels[w::2], device.dtype
            )
            for i in range(2):
                c = w + 2 * i
                wmem[c][:] = c - c * 1j
            program.waveforms.append(wmem)

        return program

    @pytest.fixture
    def missing_channels(self, device):
        program = IntermediateProgram(device="demod")

        for i in range(10):
            program.instructions.append(
                WaitTriggerInstruction(device="tektronix", index=3, subchannel=1)
            )
            program.instructions.append(PlayInstruction(waveform_index=i))

        for w in range(10):
            wmem = WaveformMemory.from_channels(
                1024 * w, 1.0e9, device.channels[2:3], device.dtype
            )
            wmem[2][:] = w + w * 1j
            program.waveforms.append(wmem)

        return program

    @pytest.mark.parametrize(
        "program,expect",
        [
            ("single_demod", np.zeros(21, dtype=int)),
            ("alternating_demod", np.array([0, 1] * 5)),
            ("missing_channels", np.arange(10, dtype=int)),
        ],
    )
    def test_demods(self, device, program, expect, request):
        program = request.getfixturevalue(program)
        compiler = HeterodyneCompiler()
        demod_program = compiler.compile(program, device)

        assert_array_equal(demod_program.demods, expect)
        assert demod_program.demods.flags.writeable is False

    @pytest.mark.parametrize(
        "program,expect",
        [
            ("single_demod", [np.r_[:4][:, np.newaxis] * np.full(4096, 1 + 1j)]),
            (
                "alternating_demod",
                [
                    np.array([0 - 0j, np.nan, 2 - 2j, np.nan])[:, np.newaxis]
                    * np.ones(4096),
                    np.array([np.nan, 1 - 1j, np.nan, 3 - 3j])[:, np.newaxis]
                    * np.ones(4096),
                ],
            ),
            (
                "missing_channels",
                [
                    np.full(1024 * i, i + 1j * i, dtype=np.complex64).reshape(1, -1)
                    for i in range(10)
                ],
            ),
        ],
    )
    def test_weights(self, device, program, expect, request):
        program = request.getfixturevalue(program)
        compiler = HeterodyneCompiler()
        demod_program = compiler.compile(program, device)

        for w1, w2 in zip(demod_program.weights, expect):
            assert_array_equal(w1, w2)
            assert w1.flags.writeable is False

    @pytest.mark.parametrize(
        "program,expect",
        [
            ("single_demod", ("R0", "R1", "R2", "R3")),
            ("alternating_demod", ("R0", "R1", "R2", "R3")),
            ("missing_channels", ("R2",)),
        ],
    )
    def test_keys(self, device, program, expect, request):
        program = request.getfixturevalue(program)
        compiler = HeterodyneCompiler()
        demod_program = compiler.compile(program, device)

        assert demod_program.keys == expect

    @pytest.mark.parametrize(
        "program", ["single_demod", "alternating_demod", "missing_channels"]
    )
    def test_hash(self, device, program, request):
        program = request.getfixturevalue(program)
        p1 = HeterodyneCompiler().compile(program, device)
        p2 = HeterodyneCompiler().compile(program, device)

        assert p1 == p2
        assert hash(p1) == hash(p2)
        assert p1 is not p2
