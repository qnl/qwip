# Timelines

Mathematically speaking, timelines describe a set of channel amplitudes $\overrightarrow{\boldsymbol{V}}(t)$ as a function of time $t$ given some common reference point $t=0$. Practially speaking, timelines are a mapping from locations (start times) to waveforms.

Timelines are used to schedule a set of waveforms into a more complex circuit or experimental protocol. In addition, timelines can be used to group a set of elementary waveforms into a logical gate operation.

Every timeline holds a list of location/waveform pairs, an optional width, a set of constraints, and a set of channels. In general, you only need to set an explicit width on a timeline when using a timeline as a composite gate operation.

## Creating a Timeline

By default, instantiating a timeline will create an empty timeline, but we can also create and populate a timeline using one of several class constructors.

```python title="timelines.py - Introduction" linenums="1"
--8<-- "timelines.py:intro"
```

Below we show two ways of creating a composite X90 pulse (90 degree rotation about the X axis), which is often implemented by sandwiching a microwave drive with a virtual-Z on either side.

```python title="timelines.py - X90" linenums="1"
--8<-- "timelines.py:X90"
```

The first method allows us to place the start times of each waveform at an arbitrary location, giving us maximum flexibility. However, in many cases we don't need this much flexibility and simply want to put each waveform after the end of the preceding waveform.

The `Timeline.from_layers` constructor, also allows us to add multiple waveforms to the same layer or add time delays between layers. These time delays can be explicit numeric delays or variable delays.

```python title="timelines.py - Layers" linenums="34"
--8<-- "timelines.py:layers"
```

1. See the expandable "Gate Setup" block for an example on creating the gates that we use.

??? example "Gate Setup"
    ```python title="timelines.py - Layers" linenums="1"
    --8<-- "timelines.py:layers-setup"
    ```

## Modifying a Timeline

Given an existing timeline, we can also add waveforms or timelines to it. This allows us to build up more complex timelines from simple ones.

```python title="timelines.py - Adding Waveforms" linenums="1"
--8<-- "timelines.py:add-waveform"
```

## Inspecting a Timeline

Finally, we show some examples on how to inspect timelines, which is often useful for debugging.

```python title="timelines.py - Timeline Properties" linenums="1"
--8<-- "timelines.py:timeline-properties"
```

To all the available methods for timelines, see the [API docs][qwip.sequencer.timeline.Timeline].