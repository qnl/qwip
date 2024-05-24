# Sequencer

At the lowest level, control and measurement of superconducting qubits consists of sending a series of waveforms to the on-chip circuit elements and measuring the reflected or transmitted signal that comes back. As you can imagine, doing this from scratch for every experiment or measurement can be confusing and difficult, which is where the sequencer comes in. The sequencer's job is to simplify the construction of these control/measurement waveforms, and compile them into a form that can be understood by the specific measurement hardware being used to physically generate and record microwave inputs/output signals.

## Pulse Level Control

There are three fundamental building blocks that you will primarily interface with as a user when performing pulse-level measurements on a superconducting quantum device: waveforms, timelines, and sequences.

[Waveforms](./waveforms.md) are the fundamental building blocks of any measurement, and refer to a single microwave pulse to be executed on a single logical channel.

[Timelines](./timelines.md) are collections of waveforms at different points in time. A given timeline sets a common reference point in time, allowing multiple waveforms to be scheduled with respect to each other and a common start time.

[Sequences](./sequences.md) are collections of timelines, that are often executed as part of a single measurement or set of measurements on the physical hardware.

Below, we'll dive into some math to formalize the concepts of waveforms and timelines for those who are interested. Otherwise, if you'd just like to learn how to construct pulses to run on your quantum computer, you can skip to the linked sections above.

## Mathematical Preliminaries

Formally speaking, we can think of waveforms and timelines as elements in a vector space of functions. In this picture, timelines are functions mapping a time $t$ to a d-dimensional vector of channel amplitudes.

$$ \overrightarrow{\boldsymbol{s}}(t): t \rightarrow \overrightarrow{\boldsymbol{V}}, \overrightarrow{\boldsymbol{V}} \in \mathbb{C}^n$$

Here, the number of channels $n$, specifies the dimension of the vector space representing all possible combinations of output voltages for the set of channels. Since waveforms map to only a single "logical" channel, we can think of waveforms on different channels as orthogonal basis vectors in this vector space.

!!! note

    While waveforms can only map to a single "logical" channel, this may correspond to 2 or more physical channels. A common situation where this occurs is when you IQ mix and combine two physical output channels via an IQ mixer to form a single channel capable of outputting signals at higher frequencies.

## A Ring of Functions

Now that we have a set of elements in this vector space, we might ask what additional structure we can define over this set. The short answer is that we can define an addition and multiplication operator over the set of all timelines in $\mathbb{C}^n$ such that they form an algebraic ring. We will show that this can be done below.

### Addition

Given two timelines $g(t)$ and $h(t)$ we can define addition in the intuitive way such that

$$s(t) = (g + h)(t) = g(t) + h(t)$$

This has all the right properties of addition that we expect and care about:

- **Commutativity**: $g + h = h + g$
- **Associativity**: $g + (h + s) = (g + h) + s$
- The existence of an **identity** operation: $I + g = g$, where $I(t) = 0$ is a null timeline that evaluates to zero at every point in time.

All of these properties are easy to check and follow directly from the corresponding properties of $\mathbb{C}^n$.

### Multiplication

Multiplication is slightly more interesting. Now we could define a multiplication operator the same way, as pointwise multiplication at each time point, but a more interesting (and useful) multiplication is given by the convolution of two timelines.

Given two timelines $g(t)$ and $h(t)$, we can define multiplication as a convolution such that

$$
s(t) = (g*h)(t) = \int_{-\infty}^\infty g(\tau)h(t - \tau) d\tau
$$

This also satisfies the same properties as above, so we have

- **Commutativity**

    $$
    (g*h)(t) = \int_{-\infty}^\infty g(\tau)h(t-\tau)d\tau = \int_{-\infty}^\infty h(\tau') g(t - \tau')d\tau' = (h*g)(t)
    $$

    where we make a change of variables $\tau' = t - \tau$

- **Associativity**

    $$f*(g*h)(t) = (f*g)*h(t)$$

    the proof of which follows from Fubini’s theorem.

- **Identity**

    $$(f*\delta)(t) = (\delta*f)(t)$$

    where the Dirac delta function $\delta(t)$ is the identity element.

- **Distributive** over addition

    $$
    \begin{align*}
    (f*(g+h))(t) &= \int_{-\infty}^\infty f(\tau)\left[g(t-\tau) + h(t-\tau)\right] d\tau \\
    &= \int_{-\infty}^\infty f(\tau)g(t-\tau) d\tau + \int_{-\infty}^\infty f(\tau)h(t-\tau) d\tau \\
    &= (f*g)(t) + (f*h)(t)
    \end{align*}
    $$

Another nice property of the convolution is that a convolution of two functions in the time domain is equivalent to their multiplication in the frequency domain. This makes multiplication a useful operation when defining or constructing filters, which are often more natural to think about in the frequency domain.

## Time Domain vs Frequency Domain

Up until now, we've discussed waveforms and timelines as functions in the time domain, however, we can also these as functions in the frequency domain by taking a fourier transform.