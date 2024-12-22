# Waveforms

Waveforms are the fundamental building blocks for pulse level control of superconducting quantum devices. They refer to a single pulse that maps a time $t$ to an amplitude $V \in \mathbb{C}$ on a single logical channel.

## Waveform Basics

All waveforms have a name and channel attribute, along with additional parameters that depend on the type of waveform.

```python title="waveforms.py - Introduction" linenums="1"
--8<-- "waveforms.py:intro"
```

Waveforms are immutable, which means that you cannot modify any waveform attributes inplace. Instead, you can evolve them, which creates a new waveform instance with some optionally modified parameters.

```python title="waveforms.py - Modifying Waveforms" linenums="1"
--8<-- "waveforms.py:evolve"
```

## Waveform Variables

Waveform attributes can also be variables or expressions, which act as placeholders for an attribute value until compile time. This done using [SymPy](https://www.sympy.org/en/index.html) under the hood. These variables or expressions can be passed in as strings, in which case they are automatically converted to a sympy expression. To create a new waveform with variables substituted, we can use the waveform `resolve` method.

```python title="waveforms.py - Waveform Variables" linenums="1"
--8<-- "waveforms.py:variables"
```

!!! note

    Remember, `evolve` is for attribute names, `resolve` is for variables!

## Evaluating Waveforms

### Time Domain
Waveforms can be evaluated in the time domain by calling the waveform instance. They can also be directly plotted using the `plot` method. If a waveform contains any variables, these should either be resolved first, or a value can be passed in via keyword argument, in which case they will be resolved internally.

```python title="waveforms.py - Evaluating Waveforms" linenums="1"
--8<-- "waveforms.py:evaluation"
```

1. The `t0` parameter allows us to shift the waveform with respect to time.

![Waveform Plot](/assets/media/waveforms-1.png)

When evaluating the waveform, the output will be a complex array, which allows the waveform to carry phase information natively.

### Frequency Domain
We can also take the Fourier transform of a waveform to look at a waveform in the frequency domain.

```python title="waveforms.py - FFT" linenums="1"
--8<-- "waveforms.py:fft"
```

![Waveform FFT](/assets/media/waveforms-2.png)

## Waveform Examples

We will now show some examples for some commonly used waveforms. You can find the exhaustive list of provided waveforms and their documentation in the [API docs][qwip.sequencer.waveform].

### Modulated Waveforms

We often want to drive a pulse that can be parameterized as a CW (constant-wave) tone with some envelope shape. These waveforms can be specified with a [`ModulatedWaveform`][qwip.sequencer.waveform.ModulatedWaveform].

```python title="waveforms.py - Modulated Waveforms" linenums="1"
--8<-- "waveforms.py:modulation"
```

1. Modulated waveforms take their channel from the inner CW waveform, and their width from the envelope.
2. When modifying waveforms that contain additional waveforms as attributes, we can chain the attributes with an underscore to modify an attribute on the inner waveform.

![Modulated Waveform](/assets/media/waveforms-3.png)

The modulation itself contains a flag `hardware_modulation` that specifies whether the frequency modulation should be evaluated in software or on the control hardware itself. When using hardware modulation, the frequency, amplitude, and phase attached to the CW waveform will be passed to the hardware instead, and the waveform evaluation will be equivalent to evaluating just the envelope.

!!! Note

    When using hardware modulation, you should typically put amplitudes and phases on the `CWWaveform` rather than the envelope. This is typically more memory-efficient, since the envelope can now be shared across two waveforms with a different amplitude/phase multiplier.

### Virtual Z Waveforms

[Virtual Z Waveforms][qwip.sequencer.waveform.VirtualZWaveform][^1] are used to implement software Z gates via phase updates and frame tracking. See [frame tracking](./frame-tracking.md) for more details on frame tracking.

```python title="waveforms.py - Virtual-Z Gates" linenums="1"
--8<-- "waveforms.py:virtual-z"
```

[^1]: D. C. McKay, *et. al.*, Efficient Z gates for quantum computing. (2017) [DOI: 10.1103/PhysRevA.96.022330](https://doi.org/10.1103/PhysRevA.96.022330)

### DRAG

DRAG[^2] (derivative removal by adiabatic gate) is a commonly used waveform-shaping technique for minimizing unwanted transitions due to non-adiabatic effects. To modify an arbitrary envelope waveform using DRAG, we can nest it inside a DRAG waveform.

```python title="waveforms.py - DRAG Correction" linenums="1"
--8<-- "waveforms.py:drag"
```

We can see the effect of first order DRAG by looking at the FFT, where we see that the fourier component at the detuning frequency is significantly suppressed compared to the original waveform.

![DRAG Suppression](/assets/media/waveforms-4.png)


[^2]: L. S. Theis, *et. al.*, Counteracting systems of diabaticities using DRAG controls: The status after 10 years. (2018) [DOI: 10.1209/0295-5075/123/60001](https://doi.org/10.1209/0295-5075/123/60001)

### Convolved Waveforms

Waveforms can also be multiplied together, which results in a [convolution](https://en.wikipedia.org/wiki/Convolution) between two waveforms. This can be used to apply a smoothing filter on a pulse to reduce its bandwidth.

```python title="waveforms.py - Waveform Convolutions" linenums="1"
--8<-- "waveforms.py:convolution"
```

1. We compute the normalization constant for the filter by integrating
2. We can ignore the imaginary component because the waveforms all have zero phase.

![Waveform Convolution](/assets/media/waveforms-5.png)

