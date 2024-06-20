from qwip.sequencer import (
    CosineRampWaveform,
    CWWaveform,
    ModulatedWaveform,
    SquareWaveform,
    Timeline,
    VirtualZWaveform,
)

# --8<-- [start:intro]
tmln = Timeline()

print(tmln)
# Timeline(lw_pairs=[], width=None, constraints=set(), channels=set())
# --8<-- [end:intro]

# --8<-- [start:X90]
X = ModulatedWaveform(
    envelope=CosineRampWaveform(amplitude=1, width=30e-9, ramp=5e-9),
    modulation=CWWaveform(frequency="Q0.freq_01"),
)

Z = VirtualZWaveform(frame="Q0.freq_01", phase=1.2)

tmln1 = Timeline.fromtuples([(0, Z), (0, X), (X.width, Z)], width=X.width)
tmln2 = Timeline.from_layers([Z, X, Z], width=X.width)

assert tmln1 == tmln2
# Timeline(lw_pairs=[], width=None, constraints=set(), channels=set())
# --8<-- [end:X90]

# --8<-- [start:layers-setup]
X = ModulatedWaveform(
    envelope=CosineRampWaveform(amplitude="amplitude", width=30e-9, ramp=5e-9),
    modulation=CWWaveform(frequency="frame", channel="drive_channel"),
)

Z = VirtualZWaveform(frame="frame", phase="zphase")

Q0_X90 = Timeline.from_layers(
    [
        Z.resolve(frame="Q0.freq_01"),
        X.assign_channel("CH0").resolve(amplitude=0.5, frame="Q0.freq_01"),
        Z.resolve(frame="Q0.freq_01"),
    ],
    width=X.width,
)

Q1_X90 = Timeline.from_layers(
    [
        Z.resolve(frame="Q1.freq_01"),
        X.assign_channel("CH1").resolve(amplitude=0.5, frame="Q1.freq_01"),
        Z.resolve(frame="Q1.freq_01"),
    ],
    width=X.width,
)

Q0_RO = SquareWaveform(amplitude=0.1, width=1e-6, channel="RO")
Q1_RO = SquareWaveform(amplitude=0.12, width=1e-6, channel="RO")
# --8<-- [end:layers-setup]

# --8<-- [start:layers]
t1_tmln = Timeline.from_layers(
    [[Q0_X90, Q1_X90], [Q0_X90, Q1_X90], "wait", [Q0_RO, Q1_RO]]  # (1)
)
# --8<-- [end:layers]

# --8<-- [start:add-waveform]
Q0_flux = SquareWaveform(amplitude=0.1, width="time", channel="CH0_DC")
t1_tmln = Timeline.from_layers([Q0_X90, "time", Q0_RO])

t1_tmln.add(Q0_flux, Q0_X90.width)
# --8<-- [end:add-waveform]

# --8<-- [start:timeline-properties]
print(t1_tmln.locations)
# [0, 0, 3.00000000000000e-8, time + 3.0e-8, 3.00000000000000e-8]

print(t1_tmln.channels)
# {'CH0', 'RO', 'CH0_DC'}
# --8<-- [end:timeline-properties]

import qwip
from qwip.flatdict import FlatDict

locations, waveforms = zip(
    *((l, FlatDict(qwip.converter.unstructure(w)).toflatdict()) for l, w in t1_tmln)
)
print(locations)
print(waveforms)
