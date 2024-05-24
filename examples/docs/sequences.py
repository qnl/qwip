import numpy as np
import pandas as pd

from qwip.sequencer import (
    CWWaveform,
    GaussianWaveform,
    ModulatedWaveform,
    Sequence,
    SquareWaveform,
    Timeline,
    VirtualZWaveform,
)

# --8<-- [start:indexing]
seq = Sequence.empty((20, 4, 5, 2))  # (1)
seq1 = seq[np.newaxis, ::2, 1:3, ...]
assert seq1.shape == (1, 10, 2, 5, 2)

seq2 = seq[0, 0, 0]
assert seq2.shape == (2,)
# --8<-- [end:indexing]

# --8<-- [start:ufuncs]
X90 = GaussianWaveform(width=20e-9, amplitude=0.25, channels=("CH0",))
Z = VirtualZWaveform(frame="Q0", phase="zphase")

Y90 = Timeline.from_layers([Z.evolve(phase=-90), X90, Z.evolve(phase=90)])

seq = Sequence.empty(3) + Y90
assert np.all(seq == Y90)
# --8<-- [end:ufuncs]

Y90.width = X90.width

# --8<-- [start:labels]
readout = SquareWaveform(width=1e-6, amplitude=0.1, channels=("RO",))

labels = ["I", "X90", "Y90", "X180"]
basis = ([], [X90], [Y90], [X90, X90])

tmlns = [Timeline.from_layers(bs + [readout]) for bs in basis]

seq = Sequence(tmlns, basis=labels)

assert seq.names == ("basis",)
assert isinstance(seq["basis"], pd.Index)

print(seq["basis"])  # (1)
# Index(['I', 'X90', 'Y90', 'X180'], dtype='object', name='basis')
# --8<-- [end:labels]

# --8<-- [start:label-indexing]
first_half = seq[:2]

assert seq.names == ("basis",)
assert len(first_half["basis"]) == len(first_half)

print(first_half["basis"])
# Index(['I', 'X90'], dtype='object', name='basis')
# --8<-- [end:label-indexing]

# --8<-- [start:basic-sweep]
drive = SquareWaveform(width="time", amplitude=0.5)
rabi_tmln = Timeline.from_layers([drive, readout])

ts = np.linspace(10e-9, 100e-9, 10)
rabi_seq = Sequence.sweep(rabi_tmln, time=ts)

assert rabi_seq.names == ("time",)
assert rabi_seq.shape == (10,)
assert rabi_seq["time"].equals(pd.Index(ts))

print(rabi_seq["time"])
# Index([1e-08, 2e-08, ..., 9e-08, 1e-07], dtype='float64', name='time')
# --8<-- [end:basic-sweep]

# --8<-- [start:zipped-sweep]
ramsey_tmln = Timeline.from_layers([X90, "time", Z, X90, readout])

df = 5e6
ts = np.linspace(10e-9, 100e-9, 10)
phis = df * ts * 360  # Degrees

ramsey_seq = Sequence.sweep(rabi_tmln, time=ts, zphase=phis)

assert ramsey_seq.names == ("time_zphase",)
assert ramsey_seq.shape == (10,)
assert ramsey_seq["time_zphase"].equals(pd.MultiIndex.from_tuples(list(zip(ts, phis))))

print(ramsey_seq["time_zphase"])
"""
MultiIndex([
    (1e-08, 18.0),
    (2e-08, 36.0),
    ...,
    (1e-07, 180.0)],
    names=["time", "zphase"]
)
"""
# --8<-- [end:zipped-sweep]

# --8<-- [start:product-sweep]
drive = ModulatedWaveform(
    envelope=SquareWaveform(width="time", amplitude=0.5),
    modulation=CWWaveform(frequency="frequency", channels=("Q0",)),
)
rabi_tmln = Timeline.from_layers([drive, readout])

fs = np.linspace(100e6, 110e6, 5)
ts = np.linspace(10e-9, 100e-9, 10)

chevron = Sequence.product(rabi_tmln, frequency=fs, time=ts)

assert chevron.names == ("frequency", "time")
assert chevron.shape == (5, 10)

print(chevron["frequency"])
# Index([100e6, 102.5e6, 105e6, 107.5e6, 110e6], dtype='float64', name='frequency')

print(chevron["time"])
# Index([1e-08, 2e-08, ..., 9e-08, 1e-08], dtype='float64', name='time')
# --8<-- [end:product-sweep]
