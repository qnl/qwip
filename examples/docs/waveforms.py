import sympy as sym
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

plt.style.use("qwip.visualization.style")

savedir = __file__.parent.parent.parent / "docs/pages/assets/media"

# --8<-- [start:intro]
from qwip.sequencer import SquareWaveform

wave = SquareWaveform(name="w1", channel="chA", width=20e-9)

print(wave)
# SquareWaveform(name='w1', t0=0, width=2e-08, channel='chA', amplitude=1, phase=0)
# --8<-- [end:intro]

assert wave.name == "w1"
assert wave.channel == "chA"
assert wave.width == 2e-08

# --8<-- [start:evolve]
wave1 = wave.evolve(amplitude=0.25, phase=90)

print(wave1)
# SquareWaveform(name='w1', t0=0, width=2e-08, channel='chA', amplitude=0.25, phase=90)
# --8<-- [end:evolve]

assert wave1.amplitude == 0.25
assert wave1.phase == 90

# --8<-- [start:variables]
wave = SquareWaveform(channel="ch1", width="t", phase="360*t*freq")

print(wave)
# SquareWaveform(name='SquareWaveform', t0=0, width=t, channel='ch1', amplitude=1, phase=360*freq*t)
print(wave.variables())
# frozenset({'t', 'freq'})

wave1 = wave.resolve(t=100e-9, freq="frequency")
print(wave1)
# SquareWaveform(name='SquareWaveform', t0=0, width=1e-07, channel='ch1', amplitude=1, phase=3.6e-5*frequency)
print(wave1.variables())
# frozenset({'frequency'})
# --8<-- [end:variables]

assert wave.variables() == {"t", "freq"}
assert wave1.variables() == {"frequency"}

# --8<-- [start:evaluation]
import numpy as np
from qwip.sequencer import CosineRampWaveform

ts = np.linspace(0, 100e-9, 501)
wave = CosineRampWaveform(width="time", amplitude=0.5, ramp=10e-9)
wave1 = wave.evolve(t0="time", phase=30) # (1)

w_t = wave(ts, time=20e-9)
print(w_t)
# [0.00000000e+00+0.j 4.93317842e-04+0.j 1.97131932e-03+0.j ... 0.00000000e+00+0.j]

fig = wave.plot(variables=dict(time=30e-9))
# We can pass in an axes to plot the waveform on, as well as the specific timepoints
# to evaluate the waveform on
fig = wave1.plot(ts=ts, ax=fig.axes[0], variables=dict(time=50e-9))
fig.axes[0].set_title("Flat Top Cosine Ramp")
# --8<-- [end:evaluation]

fig.savefig(savedir / "waveforms-1.png")

# --8<-- [start:fft]
ts = np.linspace(0, 200e-9, 501)
fs, yfs = wave.resolve(time=20e-9).fft(ts)

fig, ax = plt.subplots()
ax.plot(fs, np.abs(yfs))
ax.set_yscale("log")
ax.xaxis.set_major_formatter(EngFormatter(unit="Hz"))
ax.set_title("Flat Top Cosine Ramp, FFT")
# --8<-- [end:fft]

fig.savefig(savedir / "waveforms-2.png")

# --8<-- [start:modulation]
from qwip.sequencer import CWWaveform, ModulatedWaveform

wave = ModulatedWaveform(
    envelope=CosineRampWaveform(width="time", amplitude=0.5),
    modulation=CWWaveform(
        frequency="frequency", amplitude=0.5, channel="Q0")
)

print(wave.width, wave.amplitude, wave.channel)
# time 0.25 Q0 (1)

wave1 = wave.evolve(modulation_hardware_modulation=True, envelope_t0=50e-9) # (2)

ts = np.linspace(0, 100e-9, 501)
fig = wave.plot(ts=ts, variables=dict(time=50e-9, frequency=100e6))
fig = wave1.plot(ts=ts, ax=fig.axes[0], variables=dict(time=50e-9, frequency=100e6))
# --8<-- [end:modulation]

fig.axes[0].set_title("Modulated Waveforms")
fig.savefig(savedir / "waveforms-3.png")

assert wave.width == sym.Symbol("time")
assert wave.channel == "Q0"
assert wave.amplitude == 0.25


# --8<-- [start:drag]
from qwip.sequencer import DRAG

detuning = -70e6
lmbda = 1 / (2*np.pi*detuning)

wave = CosineRampWaveform(width=30e-9, ramp=10e-9)
drag_wave = DRAG(envelope=wave, lmbda=lmbda)

ts = np.linspace(0, 200e-9, 501)
fs, yfs = wave.fft(ts=ts)
fs_drag, yfs_drag = drag_wave.fft(ts=ts)

fig, axes = plt.subplot_mosaic([["t", "f"], ["tdrag", "f"]])
wave.plot(ts=ts[:100], ax=axes["t"])
drag_wave.plot(ts=ts[:100], ax=axes["tdrag"])
axes["f"].plot(fs, np.abs(yfs), label="Cosine Ramp")
axes["f"].plot(fs_drag, np.abs(yfs_drag), label="Cosine Ramp DRAG")
# --8<-- [end:drag]

axes["t"].set_ylabel("Original")
axes["tdrag"].set_ylabel("DRAG")
ax = axes["f"]
ax.xaxis.set_major_formatter(EngFormatter(unit="Hz"))
ax.axvline(detuning, color="k", linestyle="--", label=f"$\\Delta = {detuning / 1e6:.0f}$ MHz")
ax.set_xlim(-250e6, 250e6)
ax.legend(loc="upper left")
ax.set_ylim(*(np.abs(yfs_drag).max() * np.array([-0.1, 1.5])))
fig.suptitle("DRAG Suppression")
fig.savefig(savedir / "waveforms-4.png")


