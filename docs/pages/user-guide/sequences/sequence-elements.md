# Sequence Elements

Sequence elements describe the waveform outputs for a single timeline of pulses. This can have both an abstract representation as well as a concrete compiled representation. The first representation is convenient for constructing elements piecewise or building up a more complex element by combining several smaller sequence elements.

## Locations

At its core, sequence elements are a mapping of locations to waveforms. Each sequence element has a `locations` attribute that holds this mapping. Locations can be either concrete points in time, variable locations that represent an (temporarily) unknown point in time, or a linear combination of variable locations.

In order to resolve variable locations into concrete times, each sequence element contains another mapping of constraints that specify how variable locations should be resolved. The set of constraints forms a linear system of equations that is solved during the compilation process.

## A Ring of Functions

In this picture, sequence elements are functions mapping a time t to a d-dimensional vector of channel amplitudes.  

We can then define a ring over the set of sequence elements (or functions) by defining an addition and multiplication operator over the set.

### Addition

Given two sequence elements $g(t)$ and $h(t)$ we define addition the intuitive way such that

$$s(t) = (g + h)(t) = g(t) + h(t)$$

This has all the right properties of addition that we expect and care about:

- Commutativity: $g + h = h + g$
- Associativity: $g + (h + s) = (g + h) + s$
- The existence of an identity operation: $I + g = g$, where $I(t) = 0$ is an empty sequence element with no pulses.

All of these properties are easy to check and follow directly from the corresponding properties of $\mathbb{C}^n$.

### Multiplication

Now we could define a multiplication operator the same way, as pointwise multiplication at each time point, but a more interesting (and useful) multiplication is given by the convolution of two sequence elements.

Given two sequence elements $g(t)$ and $h(t)$, we can define multiplication as a convolution such that

$$
s(t) = (g*h)(t) = \int_{-\infty}^\infty g(\tau)h(t - \tau) d\tau
$$

This also satisfies the same properties as above, so we have

- Commutativity

    $$
    (g*h)(t) = \int_{-\infty}^\infty g(\tau)h(t-\tau)d\tau = \int_{-\infty}^\infty h(\tau') g(t - \tau')d\tau' = (h*g)(t)
    $$

    where we make a change of variables $\tau' = t - \tau$

- Associativity

    $$f*(g*h)(t) = (f*g)*h(t)$$

    the proof of which follows from Fubini’s theorem.

- Identity

    $$(f*\delta)(t) = (\delta*f)(t)$$

    where the Dirac delta function $\delta(t)$ is the identity element.

- Distributive over addition
