# Sequences

Sequences are the final construct necessary for building up measurements and experiments. At its core, sequences are just a collection of timelines that have been grouped together, often to be executed as a single batch of measurements on the physical hardware. While these timelines could be completely unrelated, they are more often not simply variations of a single timeline structure.

## Array Indexing

Sequences behave very much like [Numpy arrays](https://numpy.org/doc/stable/reference/arrays.ndarray.html), because they are in fact subclasses of `numpy.ndarray`. This allows us to slice and index sequences using a familiar notation, 

```python title="sequences.py - Indexing" linenums="1"
--8<-- "sequences.py:indexing"
```

1. Creates a sequence of empty timelines with shape `(20, 4, 5, 2)`

We can also perform "vectorized" operations on an entire collection of timelines, and broadcast sequences with compatible shapes according to standard numpy broadcasting rules.

```python title="sequences.py - Universal Functions" linenums="1"
--8<-- "sequences.py:ufuncs"
```

## Sequence Labels

Sequences also carry labels for each axis, allowing us to attach some semantic meaning to the timelines that they hold. For example, suppose you want to perform state tomography and repeat the same preparation circuit with one of 4 basis rotations prior to the measurement. You can then create a sequence labeled by the basis rotations, allowing you to keep track of which timeline corresponds to which basis rotation.

```python title="sequences.py - Labels" linenums="1"
--8<-- "sequences.py:labels"
```

1. You can access the labels by key via the `__getitem__` notation, or by axis like `seq.labels[0]`

These labels are also automatically transferred to measurement results, making it straightforward to match a result value to a particular timeline.

Sequence labels are preserved during indexing:

```python title="sequences.py - Label Indexing" linenums="1"
--8<-- "sequences.py:label-indexing"
```


## Parameter Sweeps

While sequences can contain an arbitrary collection of timelines, they are most often constructed by sweeping a parameter(s) in a prototype timeline. For example, we can construct a rabi sequence by sweeping a width parameter:

```python title="sequences.py - Basic Sweep" linenums="1"
--8<-- "sequences.py:basic-sweep"
```

Sometimes, we may want to sweep multiple parameters simultaneously, such as in a ramsey sequence, where the Z phase and delay time are related by a detuning frequency.

```python title="sequences.py - Zipped Sweep" linenums="1"
--8<-- "sequences.py:zipped-sweep"
```

In this case, the resulting label will be a pandas [multiindex](https://pandas.pydata.org/docs/user_guide/advanced.html) with the values for each parameter that is swept.

Finally, we may want to construct a 2D (or N-D) sweep consisting of all possible combinations of two 1D sweeps. A common example is a chevron, where we sweep the frequency and time of a drive pulse.

```python title="sequences.py - Prooduct Sweep" linenums="1"
--8<-- "sequences.py:product-sweep"
```