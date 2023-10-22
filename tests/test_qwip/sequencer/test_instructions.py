import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from qwip.sequencer.compilation import (
    ChannelInfo,
    DeviceInfo,
    QuantumExecutable,
    QWiPCompiler,
    QWiPExecutable,
    TriggerInfo,
)
from qwip.sequencer.phase_tracker import Frame
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.timeline import Timeline
from qwip.sequencer.waveform import (
    CosineRampWaveform,
    CWWaveform,
    GaussianWaveform,
    ModulatedWaveform,
    ReadoutMarker,
    SquareWaveform,
    TriggeredWaveform,
    VirtualZWaveform,
)


class TestQWiPCompiler:
    @pytest.fixture
    def compiler(self):
        dac = DeviceInfo.from_channels(
            channels=(
                ChannelInfo("Q0_I", 0),
                ChannelInfo("Q0_Q", 1),
                ChannelInfo("Q1_I", 0),
                ChannelInfo("Q1_Q", 1),
                ChannelInfo("RO_I", 2),
                ChannelInfo("RO_Q", 3),
                ChannelInfo("RO_marker", 3, subchannel=1),
            ),
            sample_rate=1.0e9,
            name="tektronix",
        )

        adc = DeviceInfo.from_channels(
            channels=(ChannelInfo("RO", 0, read=True),),
            sample_rate=1.0e9,
            trigger=TriggerInfo(device="tektronix", index=3, subchannel=0),
            name="adc",
        )

        demod = DeviceInfo.from_channels(
            channels=(
                ChannelInfo("R0", 0),
                ChannelInfo("R1", 1),
            ),
            sample_rate=1.0e9,
            trigger=TriggerInfo(device="tektronix", index=3, subchannel=0),
            name="demod",
        )

        frames = dict(
            mod_Q0=Frame(200e6),
            mod_Q1=Frame(150e6),
            mod_R0=Frame(-300e6),
            mod_R1=Frame(-400e6),
        )

        return QWiPCompiler.from_devices([dac, adc, demod], frames=frames)

    @pytest.fixture
    def pulses(self):
        Q0_X90 = ModulatedWaveform(
            name="Q0_X90",
            envelope=GaussianWaveform(width=25e-9, amplitude=0.1),
            modulation=CWWaveform(frequency="mod_Q0", channels=("Q0_I", "Q0_Q")),
        )

        Q1_X90 = ModulatedWaveform(
            name="Q1_X90",
            envelope=GaussianWaveform(width=25e-9, amplitude=0.1),
            modulation=CWWaveform(frequency="mod_Q1", channels=("Q1_I", "Q1_Q")),
        )

        Q0_Z90 = VirtualZWaveform(name="Q0_Z", frame="mod_Q0", phase=90)
        Q1_Z90 = VirtualZWaveform(name="Q0_Z", frame="mod_Q1", phase=90)

        R0 = ModulatedWaveform(
            name="R0",
            envelope=SquareWaveform(width=1e-6, amplitude=0.2),
            modulation=CWWaveform(channels=("RO_I", "RO_Q"), frequency="mod_R0"),
        )

        R1 = ModulatedWaveform(
            name="R1",
            envelope=SquareWaveform(width=1e-6, amplitude=0.25),
            modulation=CWWaveform(channels=("RO_I", "RO_Q"), frequency="mod_R1"),
        )

        D0 = ModulatedWaveform(
            name="D0",
            envelope=SquareWaveform(width=1e-6, amplitude=1),
            modulation=CWWaveform(channels=("R0",), frequency="mod_R0"),
        )

        D1 = ModulatedWaveform(
            name="D0",
            envelope=SquareWaveform(width=1e-6, amplitude=1),
            modulation=CWWaveform(channels=("R1",), frequency="mod_R1"),
        )

        return dict(
            Q0_X90=Q0_X90,
            Q1_X90=Q1_X90,
            Q0_Z90=Q0_Z90,
            Q1_Z90=Q1_Z90,
            R0=R0,
            R1=R1,
            read=SquareWaveform(width=1e-6, channels=("RO",)),
            D0=D0,
            D1=D1,
        )

    def test_compile_timeline(self, compiler, pulses):
        tmln = Timeline()
        tmln.add(pulses["Q0_X90"])
        tmln.add(pulses["Q0_X90"], pulses["Q0_X90"].width)
        tmln.add(pulses["R0"], 2 * pulses["Q0_X90"].width)

        ro = Timeline()
        ro.add(pulses["read"])
        ro.add(pulses["D0"])
        ro.add(pulses["D1"])

        readout = TriggeredWaveform(target=ro, width=2e-9, channels=("RO_marker",))

        tmln.add(readout, 2 * pulses["Q0_X90"].width + 100e-9)

        exe = QWiPExecutable.from_devices(
            sequence=None, devices=compiler.channels.values()
        )
        exe.num_reads.append(0)

        instruction_cache = dict()
        compiler.compile_timeline(exe, tmln, instruction_cache=instruction_cache)

        for prog in exe.programs.values():
            print(prog)
