# Frame Tracking

Frame tracking refers to the tracking of the dynamical phase evolution of a quantum state via a classical hardware or software oscillator. 

## Single Qubit Frame Tracking

A general single qubit state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ has two complex phases associated with each of the two states, which we will call $\phi_0$ and $\phi_1$. However, since global phases are unphysical, the only measurable phase is the phase difference $\phi_{01} = \phi_1 - \phi_0$ between the two states. When the qubit is not subject to any external drives, this phase will evolve as $\phi_{01}(t) = \omega_{01}t$, where $\hbar\omega_{01}$ is the energy difference between the two states. This phase evolution is typically tracked with a classical oscillator, in which case we say that we are operating in the “rotating frame” of the qubit.

### Rotations

A rotation about the z axis of angle $\phi$ has a unitary of the form

$$
R_z(\phi) = e^{-i\frac{1}{2}\phi Z} = \begin{pmatrix} e^{-i\frac{1}{2}\phi} & 0 \\ 0 & e^{i\frac{1}{2}\phi}\end{pmatrix}
$$

A rotation of angle $\theta$ about an axis in the x-y plane defined by an angle $\beta$ has a unitary of the form

$$
R(\theta, \beta) = e^{-i\frac{1}{2}\theta(\cos(\beta) X + \sin(\beta) Y)} = \begin{pmatrix}
\cos\left(\frac{\theta}{2}\right) & 
-i\sin\left(\frac{\theta}{2}\right)e^{i\beta} \\
-i\sin\left(\frac{\theta}{2}\right)e^{-i\beta} &
\cos\left(\frac{\theta}{2}\right) 
\end{pmatrix}
$$

The entire family of $R(\theta,\beta)$ gates can typically be implemented on a superconducting qubit platform with a single microwave drive at the 0-1 frequency. Since this frequency is the same as the phase evolution of the qubit, it tracks the phase evolution perfectly and the overall phase of the drive then defines the axis of rotation, which we will call $\beta$. Note that the phase of the first pulse is arbitrary and simply sets the angle $\beta_0 = 0$ that we subsequently call the x-axis.

### Virtual and Physical Z Gates 

The set of $R_z(\phi)$ gates, on the other hand, can be implemented in several ways.

In one case, the frequency of the qubit can be shifted to $\tilde{\omega}$ for some duration during which the qubit evolves at a frequency detuned from our frame-tracking oscillator, and the qubit will then pick up a phase $\phi = \Delta t$ where $\Delta = \tilde{\omega} - \omega$. This often referred to as a ***physical Z gate***, since it is actually realized at the time the frequency shift occurs.

The other way that $R_z(\phi)$ gates are generally implemented is via phase shifts on the frame-tracking oscillator. These are referred to as ***virtual Z gate***, since this is equivalent to commuting a Z gate across all other rotation gates and deferring the Z gates to the end of a circuit. Since we typically measure our qubits in the Z basis, these final Z rotations have no effect on the measurement result and are simply ignored. To see that this can be done, we can consider how an $R_z(\alpha)$ gate applied before and a $R_z(\gamma)$ applied after a $R(\theta, \beta)$ rotation affect the overall unitary.

$$
\begin{align*}
R_z(\gamma)R(\theta, \beta)R_z(\alpha) &= 
\begin{pmatrix} e^{-i\frac{1}{2}\gamma} & 0 \\ 0 & e^{i\frac{1}{2}\gamma}\end{pmatrix}
\begin{pmatrix}
\cos\left(\frac{\theta}{2}\right) & 
-i\sin\left(\frac{\theta}{2}\right)e^{i\beta} \\
-i\sin\left(\frac{\theta}{2}\right)e^{-i\beta} &
\cos\left(\frac{\theta}{2}\right) 
\end{pmatrix}
\begin{pmatrix} e^{-i\frac{1}{2}\alpha} & 0 \\ 0 & e^{i\frac{1}{2}\alpha}\end{pmatrix} \\
&= \begin{pmatrix}
\cos\left(\frac{\theta}{2}\right) e^{-i(\alpha + \gamma)/2}& 
-i\sin\left(\frac{\theta}{2}\right)e^{i(\beta + (\alpha - \gamma) / 2)} \\
-i\sin\left(\frac{\theta}{2}\right)e^{-i(\beta + (\alpha - \gamma)/2)} &
\cos\left(\frac{\theta}{2}\right) e^{i(\alpha + \gamma)/2}
\end{pmatrix}
\end{align*}
$$

When $\gamma = -\alpha$, this reduces to 

$$
R_z(-\alpha)R(\theta, \beta)R_z(\alpha) = 
\begin{pmatrix}
\cos\left(\frac{\theta}{2}\right) & 
-i\sin\left(\frac{\theta}{2}\right)e^{i(\beta + \alpha)} \\
-i\sin\left(\frac{\theta}{2}\right)e^{-i(\beta + \alpha)} &
\cos\left(\frac{\theta}{2}\right) 
\end{pmatrix} = R(\theta, \beta + \alpha)
$$

This implies that 

$$
R(\theta, \beta)R_z(\alpha) = R_z(\alpha)R(\theta, \beta + \alpha)R_z(-\alpha)R_z(\alpha) = R_z(\alpha)R(\theta, \beta + \alpha)
$$

so we see that any Z rotation $R_z(\alpha)$ applied ***before*** $R(\theta, \beta)$ is equivalent to applying $R_z(\alpha)$ ***after*** $R(\theta, \beta + \alpha)$. In this way, we can keep deferring Z gates without changing the effective unitary by updating the phase of our rotation axis and pushing all Z rotations to the very end of our circuit.

## Qudit Frame Tracking

If we move beyond a two-level qubit system to a $d$-level *qudit* system, the quantum state is now described by
$$|\psi\rangle = \sum_{n=0}^{d-1} c_n|n\rangle $$
and the number of dynamical phases increases to $d$. Again, since we can measure only phase *differences*, we are left with $d-1$ independent and measurable phases.

However, when $d > 2$, there is no longer a unique way to define a set of $d - 1$ independent phases. For example, let us consider the simplest non-trivial case where $d = 3$. In this case, we have three phases $\phi_0, \phi_1, \phi_2$ associated with the states $|0\rangle, |1\rangle, |2\rangle$. The three physically measurable phases are given by

$$\begin{align*}
\phi_{01} &= \phi_1 - \phi_0 \\
\phi_{12} &= \phi_2 - \phi_0 \\
\phi_{02} &= \phi_2 - \phi_0
\end{align*}$$

from which we can see that they are related by $\phi_{02} = \phi_{01} + \phi_{12}$. What this means is that if we have a phase coherent drive at $\omega_{02} / 2$, the phase of the drive must be $\phi_{02}/2 = (\phi_{01} + \phi_{12}) / 2$ and take into account phase updates on both the 0-1 and 1-2 frames.

## Two Qubit Frame Tracking

In the case where we have more than one qubit, frame tracking now depends on the implementation details of the two qubit gates being used. The simplest situation arises when all two qubit gates commute with single qubit Z gates (i.e. CZ gates). In this case, the multi-qubit frame tracking is no different from the single qubit case, since we can still commute all Z gates across the two qubit gates. However, many two qubit gates do not commute with a single qubit Z gate, in which case we must determine how the dynamical phases are modified by the action of the gate. We will discuss several examples below:

### Cross Resonance Gate

### Parametric fSim Gate

The fermionic simulation gate implemented via a parametrically driven coupler has a unitary defined by

$$
\mathrm{fSim}(\theta,\phi,\beta) = \begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\left(\frac{\theta}{2}\right) &
i\sin\left(\frac{\theta}{2}\right)e^{i\beta} & 0 \\
0 & i\sin\left(\frac{\theta}{2}\right)e^{-i\beta} & 
\cos\left(\frac{\theta}{2}\right) & 0\\
0 & 0 & 0 & e^{i\phi}
\end{pmatrix}
$$

which looks like a rotation in the $|01\rangle-|10\rangle$ subspace with an additional $\mathrm{CPHASE}(\phi)$. Here, the axis of rotation, $\beta$, in the single excitation subspace is defined by the phase of the parametric flux drive at a frequency $\Delta = \omega_{01} - \omega_{10}$. To see how frame tracking can be implemented with this gate, we consider the action of two Z gates $R_{z,1}(\alpha_1)$ and $R_{z,2}(\alpha_2)$ before and two Z gates $R_{z,1}(\gamma_1)$ and $R_{z,2}(\gamma_2)$ after the fSim gate. The resulting unitary is given by

$$
\begin{align*}
U &= R_{z,1}(\gamma_1)R_{z,2}(\gamma_2)\mathrm{fSim}(\theta,\phi,\beta)R_{z,1}(\alpha_1)R_{z,2}(\alpha_2) \\
&= 
\begin{pmatrix}
e^{-i(\gamma_1 +\gamma_2)/2} & 0 & 0 & 0 \\
0 & e^{-i(\gamma_1 - \gamma_2) / 2} & 0 & 0\\
0 & 0 & e^{i(\gamma_1 -\gamma_2)/2} & 0 \\
0 & 0 & 0 & e^{i(\gamma_1 +\gamma_2)/2}
\end{pmatrix} \\
&\times \begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\left(\frac{\theta}{2}\right) &
i\sin\left(\frac{\theta}{2}\right)e^{i\beta} & 0 \\
0 & i\sin\left(\frac{\theta}{2}\right)e^{-i\beta} & 
\cos\left(\frac{\theta}{2}\right) & 0\\
0 & 0 & 0 & e^{i\phi}
\end{pmatrix}\\
&\times \begin{pmatrix}
e^{-i(\alpha_1 +\alpha_2)/2} & 0 & 0 & 0 \\
0 & e^{-i(\alpha_1 - \alpha_2) / 2} & 0 & 0\\
0 & 0 & e^{i(\alpha_1 -\alpha_2)/2} & 0 \\
0 & 0 & 0 & e^{i(\alpha_1 +\alpha_2)/2}
\end{pmatrix}\\
&= \begin{pmatrix}
e^{i\phi_{00}} & 0 & 0 & 0 \\
0 & \cos\left(\frac{\theta}{2}\right)e^{i\phi_{11}} &
i\sin\left(\frac{\theta}{2}\right)e^{i\phi_{12}} & 0 \\
0 & i\sin\left(\frac{\theta}{2}\right)e^{i\phi_{21}} & 
\cos\left(\frac{\theta}{2}\right) e^{i\phi_{22}} & 0\\
0 & 0 & 0 & e^{i\phi_{33}}
\end{pmatrix}
\end{align*}
$$

Where the phases $\phi_{ij}$ on each matrix element are given by

$$
\begin{align*}
\phi_{00} &= -\frac{1}{2}\left(\alpha_1 + \alpha_2 + \gamma_1 + \gamma_2\right) \\
\phi_{11} &=-\frac{1}{2}\left(\alpha_1 + \gamma_1 - \alpha_2 - \gamma_2\right)\\
\phi_{12} &= \beta + \frac{1}{2}\left((\alpha_1 - \gamma_1) -(\alpha_2-\gamma_2)\right)\\
\phi_{21} &= -\beta - \frac{1}{2}\left((\alpha_1 - \gamma_1) -(\alpha_2-\gamma_2)\right) \\
\phi_{22} &=\frac{1}{2}\left(\alpha_1 + \gamma_1 - \alpha_2 - \gamma_2\right)\\
\phi_{33} &= \frac{1}{2}\left(\alpha_1 + \alpha_2 + \gamma_1 + \gamma_2\right) + \phi \\
\end{align*}
$$

Just as in the single qubit case, we see that in the special case where $\gamma_1=-\alpha_1$ and $\gamma_2=-\alpha_2$, this reduces to

$$
\begin{align*}
U &= R_{z,1}(-\alpha_1)R_{z,2}(-\alpha_2)\mathrm{fSim}(\theta,\phi,\beta)R_{z,1}(\alpha_1)R_{z,2}(\alpha_2)\\
&= 
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\left(\frac{\theta}{2}\right) &
i\sin\left(\frac{\theta}{2}\right)e^{i(\beta + \alpha_1 - \alpha_2)} & 0 \\
0 & i\sin\left(\frac{\theta}{2}\right)e^{-i(\beta + \alpha_1 - \alpha_2)} & 
\cos\left(\frac{\theta}{2}\right) & 0\\
0 & 0 & 0 & e^{i\phi}
\end{pmatrix} \\
&= \mathrm{fSim}(\theta, \phi, \beta + \alpha_1 - \alpha_2)
\end{align*}
$$

which is equivalent to a rotation in the the single excitation subspace around and axis shifted by $\alpha_1-\alpha_2$. As a result, we can commute single qubit Z gates across the fSim gate in a similar way to the single qubit rotation gates, with phase updates given by $\alpha_1 - \alpha_2$. We can push all Z gates after the fSim gate using the commutation relation

$$
\begin{align*}
\mathrm{fSim}(\theta,\phi,\beta)R_{z,1}(\alpha_1)R_{z,2}(\alpha_2) &= R_{z,1}(\alpha_1)R_{z,2}(\alpha_2)\mathrm{fSim}(\theta,\phi,\beta + \alpha_1 - \alpha_2)R_{z,1}(-\alpha_1)R_{z,2}(-\alpha_2)R_{z,1}(\alpha_1)R_{z,2}(\alpha_2) \\
&= R_{z,1}(\alpha_1)R_{z,2}(\alpha_2)\mathrm{fSim}(\theta,\phi,\beta + \alpha_1 - \alpha_2)
\end{align*}
$$

## Implementation in QWiP

Now that we understand how frame tracking works theoretically, we can discuss its implementation in QWiP. When computing the phase on a frame we must take into account both its dynamical evolution and the sum of all prior phase updates on that frame.

Each frame is associated with a dynamical evolution frequency that specifies the rate of phase accumulation, as well as a list of [`PhaseJump`][qwip.sequencer.phase_tracker.PhaseJump] that add or subtract a phase increment at a given point in time.

[^1]: D. C. McKay, *et. al.*, Efficient Z gates for quantum computing. (2017) [DOI: 10.1103/PhysRevA.96.022330](https://doi.org/10.1103/PhysRevA.96.022330)

[^2]: J. Chen, *et. al.*, Compiling arbitrary single-qubit gates via the phase shifts of microwave pulses. (2023) [DOI: 10.1103/PhysRevResearch.5.L022031](https://doi.org/10.1103/PhysRevResearch.5.L022031)

[^3]: K. X. Wei, *et. al.*, Native Two-Qubit Gates in Fixed-Coupling, Fixed-Frequency Transmons Beyond Cross-Resonance Interaction. (2024) [DOI: 10.1103/PRXQuantum.5.020338](https://doi.org/10.1103/PRXQuantum.5.020338)