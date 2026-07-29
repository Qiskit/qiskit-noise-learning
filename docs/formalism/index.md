# Mathematical formalism

Given the task of learning Pauli noise for a set of Clifford gates, a common analysis technique is
to track how individual Pauli operators are transformed through a sequence of gate applications
(under the assumption that the gates are Clifford, and the noise is a Pauli channel, a single Pauli
will always be mapped to another Pauli up to a scalar that is a function of the noise). Obviously,
the evolution of *any* state under such a sequence can be captured by a linear combination of such
trajectories, but under the assumption that we always prepare the state in a Pauli eigenstate, and
always measure and post-process the results to compute the expectation value of a Pauli operator,
the relationship between the expectation value and the noise model parameters will *always* depend
on only a single such trajectory. This follows from the simple fact that the initial state is a
linear combination of Pauli operators, each gate and noise model maps Paulis to Paulis and preserves
their orthogonality, and the final expectation value "selects" only one of the Paulis in the final
decomposition before measurement.

This type of reasoning appears in many parallel research tracks in noise learning, most notably for
this package in gauge-aware Pauli-noise learning literature {cite}`chen_efficient_2026` and in the
ACES literature {cite}`flammia_averaged_2022`. This reasoning was further generalized in
{cite}`zhang_generalized_2025` to include Clifford-MCM gates (a Clifford gate followed by a
projective mid-circuit measurement). This package most closesly follows
{cite}`chen_efficient_2026,zhang_generalized_2025`. While not explicitly named, we adopt the
*Pattern Transfer Graph* (PTG) formalism for describing how Pauli operators evolve through learning
circuits; providing a direct data representation of *paths* through the graph.

The following is a review of some core mathematical concepts from the literature. It is primarily
meant to consolidate notation, and to serve as a conceptual documentation reference for the rest of
the package.

## 1. Background

### 1.1 Notation

For $K \in \mathbb{N}$, let $[K] = \{0, 1, ..., K - 1\}$. For a finite set $S \subset \mathbb{N}$ of
qubit indices, let $\P^S$ denote the set of *unphased* Pauli operators acting on those qubits. Note
that we think of elements of $\P^S$ as functions mapping $S \rightarrow \{I, X, Y, Z\}$, so that for
any $P \in \P^S$ and $T \subseteq S$, $P|_T$ denotes the restriction of $P$ to the qubit subset $T$.
Along these lines, for disjoint sets $S, T$, and $P \in \P^S$ and $Q \in \P^T$, $P \otimes Q$
denotes the element of $\P^{S \cup T}$ such that $P \otimes Q |_S = P$ and $P \otimes Q |_T = Q$.
This notation is helpful to avoid explicitly dealing with subsystem orderings, and to make it easy
to describe restrictions.

For finite subsets $M \subset \mathbb{N}$, we denote $\Z_2^M$ as the set of the bit
strings whose elements are indexed by $M$. Similarly to the above, we think of elements as functions
mapping $M \rightarrow \Z_2$, so that we may easily describe substrings in terms of restrictions of
the index set.

Lastly, for a matrix $X$, we use $\opket{X}$ to denote its vectorization. Based on the limited way
in which we use this notation, it is not actually necessary to choose a specific vectorization
convention. For a classical bit string $m$, we use the shorthand
$\opket{m} = \opket{\ket{m}\bra{m}}$.

### 1.2 Quantum instruments

A quantum operation producing classical bits $m \in \Z_2^M$ (e.g. the result of measurement) is
generally modelled as a linear map of the form:

$$ \rho \mapsto \sum_{m \in \Z_2^M} \E_m(\rho) \otimes \ket{m}\bra{m}, $$

where the set $\{\E_m: m \in \Z_2^M\}$ are completely positive, and $\sum_m \E_m$ is trace
preserving. A set of completely positive maps $\{\E_m\}$ satisfying these properties is called a
*quantum instrument*.

## 2. Noisy Clifford-MCM-reset gates

The formalism utilized in this package assumes every gate in the gate set to be characterized
consists of the following sequence of operations on $K$ qubits:

1. A Clifford operation on all qubits.
2. A mid-circuit projective measurement along $Z$ on some subset of qubits $M \subseteq [K]$.
3. A mid-circuit reset to the $Z$ ground state on some subset of qubits $R \subseteq [K]$.

This is a general class of operations that includes unitary Clifford gates, measurement,
state preparation, and any combination of the above. Note that we assume measurement and reset are
always along the $Z$-axis for each qubit. While this is not strictly required, it is a common
feature of many quantum computing modalities, and enables simplified representations and analysis.

As outlined in {cite}`beale_randomized_2023,zhang_generalized_2025` for the no-reset case, if a
specific twirling strategy is applied to a noisy instance of such a gate, then the action of the
resulting operation can be modelled mathematically as a *uniform Pauli instrument*. That is, within
the quantum instrument notation, $\E_m = \U_mG$, where $G$ is the Clifford unitary, and:

$$ \U_m = \sum_{a,b \in \mathbb{Z}^M_2} \Lambda_{a,b} \otimes \opket{m + a}\opbra{m + b}, $$

where each $\Lambda_{a,b}$ is a sub-normalized Pauli channel on the unmeasured qubits $N = [K]
\setminus M$. It is implied by this being a quantum instrument that $\sum_{a,b} \Lambda_{a,b}$ is
also trace preserving. Some notes:

- The initial untwirled noise is modelled to include both "quantum" and "classical" errors: i.e.
  erroneous operations on the quantum registers, as well as mistakes in the measurement value
  reporting.
- The noise map $\Lambda_{a,b}$ is *independent* of the measurement outcome $m$.

In words, a single term in the above sum represents observing a measurement outcome of $m$
when the measured state was $\ket{m + b}$ (a misclassification if $b \neq 0$), and when
the output state on the measurement register is $\ket{m + a}$ (the "wrong" state when $a \neq 0$).
The map $\Lambda_{a,b}$ simultaneously encodes the action on the unmeasured qubits (conditioned on
the measurement behaviour) and the *probability* of the specific measurement behaviour (through the
normalization).

Note that we are not concerned here with the specifics of the twirling strategy: that such a
strategy exists to put the channel into the above form is enough. Note that "finer" twirling
strategies exist which can further restrict the form of the Pauli channels
{cite}`berg_techniques_2024`, however we take the above form as the most general mathematical
representation under consideration.

In Lemma 1 of {cite}`zhang_generalized_2025`, it is shown that $\E_m = \U_mG$ can be rewritten
as:

$$ \E_m = \frac{1}{2^{2|M| + |N|}}\sum_{x, y \in \Z_2^M, Q \in \P^N} (-1)^{m \cdot (x + y)}
\lambda^Q_{x, y}\opket{Q \otimes Z^y}\opbra{G^\dagger(Q \otimes Z^x)}. $$ (clifford_mcm_form)

for some real numbers $\lambda_{x,y}^Q$, which are called the *fidelities* of the instrument.

Adding reset to this picture is relatively straightforward. A noisy reset operation on qubits
$R \subset [K]$ can be modelled according to the decomposition:

$$ \frac{1}{2^{|R|}}\sum_{r \in \Z_2^{R}} \lambda_r \opket{Z^r}\opbra{I_R}. $$ (reset_form)

The above form explicitly utilizes the assumption that the reset is along the $Z$-axis for each
qubit. Noise in the operation is encoded in the rest fidelities $\lambda_r$, which are simply
indexed by qubit subsets.

An analog to Equation {eq}`clifford_mcm_form` that includes a reset operation at the end can be
attained by simply composing it with Equation {eq}`reset_form`. This composition, after some
simplification, yields:

$$ \frac{1}{2^{2|M|+|N|}} \sum_{\substack{x \in \Z_2^M, y \in \Z_2^{M \cup R} \\ Q \in
\P^{N\setminus R}}} (-1)^{m \cdot (x + y|_M)} \lambda^Q_{x,y}  \opket{Q \otimes
Z^y}\opbra{G^\dagger(Q \otimes I_{N \cap R} \otimes Z^x)}. $$ (clifford_mcm_reset_form)

This decomposition indicates that the number of fidelities $\lambda^Q_{x, y}$ is

$$4^{|N \setminus R|} 2^{|M|}2^{|M \cup R|} = 4^{K}2^{|M|}2^{-|M \cup R|} = 4^K 2^{-|R \setminus M|}.$$

Consider limiting cases: (1) For no measurement or reset, $M = R = \emptyset$, and the expression
yields $4^K$, the number of phase-less Pauli operators on $K$ qubits. (2) For $M = \emptyset$ and
$R=[K]$, this yields $2^K$, which is the number of subsets of $[K]$. Lastly, (3) for $M = [K]$ and
$R = \emptyset$, this yields $4^K$. Note that this may initially seem surprising: for standard
learning, measurement and reset are typically considered to have the same number of fidelities:
$2^K$. However, here the all-measurement gate in this context is understood not as a *terminal*
measurement, but as an MCM, and as such fidelities corresponding to non-identity *outputs* of the
measurement are also included (i.e. $y \neq 0$ in Equation {eq}`clifford_mcm_reset_form`).

Finally, it is important to note that any twirling strategy yielding the form in Equation
{eq}`clifford_mcm_form` in the non-reset case still applies equally well to the non-trivial reset
case with a simple modification. The potential problem is that such a twirling strategy may require
operations on measured qubits *after* the measurement. Therefore, if a qubit is measured *and*
reset, implementing the strategy seemingly requires inserting operations *within* the gate between
the measurement and reset. However, given the nature of reset, any such operations have no effect,
and can simply be skipped while still yielding the same desired structure.

## 3. Path formalism

As described in the introduction, all learning algorithms for Pauli-type noise on Clifford gates are
based around tracking how individual Paulis evolve through a circuit, picking up noise fidelity
factors along the way. The expectation values of a circuit composed of such gates are therefore
related simply to products of the underlying fidelities of the individual gates, and in this way
model parameters can be inferred by *inverting* whatever sparse parameter-to-fidelity mapping is
being assumed.

In {cite}`chen_efficient_2026`, in the context of unitary gate sets, this is formalized into the
*Pattern Transfer Graph* (PTG): a direct graph describing all possible experiments consisting of
elements of the gate set and layers of single qubit Clifford gates (assumed to be perfect, or
"free", operations). This was generalized in {cite}`zhang_generalized_2025` to include gate sets
with mid-circuit measurements. In both cases, each experiment is described by tracking a single
Pauli operator through the circuit, under the assumption of a particular observables being computed
at each measurement site. Here we do not directly review the graph or path constructions, however we
present the required facts for justifying them within our own notation and with resets included.


While the tracking of a single Pauli operator through the circuit may be intuitive in the unitary
gate set case, it does not so obviously hold in the more general Clifford-MCM-reset case, due to the
non-deterministic nature of measurement. The follow proposition recovers this picture even in the
more general case {cite}`zhang_generalized_2025`:


```{prf:proposition}
:label: prop-mcm-evolution

For any $x \in \Z_2^M$, $y \in \Z_2^{M \cup R}$, and $Q\in P^{N \setminus R}$, it holds that:

$$
\opbra{Q \otimes Z^y}\otimes \opbra{Z^{x + y|_M}}_M \left(\sum_{m\in \Z_2^M} \E_m \otimes \opket{m}\right)
= \lambda_{x,y}^Q\opbra{G^\dagger(Q \otimes I_{N \cap R} \otimes Z^x)}
$$
```

In other words, if we compute the expectation value $Z^{x + y|_M}$ on the classical register output
by the MCM, then the "Heisenberg picture" evolution of an observable through the Clifford-MCM-reset
gate maps $Q \otimes Z^y \rightarrow Q \otimes I_{N \cap R} \otimes Z^x$. This fact alone enables
recovery of PTG-like analysis beyond the unitary gate set case: with a particular post-processing of
the measurement results, even gates with measurements and resets can be viewed as deterministically
mapping one Pauli operator to another.

Before proving this, we write Equation {eq}`clifford_mcm_reset_form` in a more suggestive way. Note
that for $x, y \in \Z_2^M$, it holds that $Z^{x+y} = \sum_{m \in \Z_2^M} (-1)^{m \cdot (x +
y)}\ket{m}\bra{m}$, and as such (in super operator notation) we have:

$$ \begin{aligned} \sum_{m \in \Z_2^M} \E_m \otimes \opket{m}_M = \sum_{\substack{x \in
\Z_2^M, y \in \Z_2^{M \cup R} \\ Q \in \P^{N\setminus R}}} \lambda^Q_{a,b}  \opket{Q \otimes Z^y}
\opbra{G^\dagger(Q \otimes I_{N \cap R} \otimes Z^x)} \otimes \opket{Z^{x + y|_M}}_M, \end{aligned} $$

where we have applied Equation {eq}`clifford_mcm_reset_form` and collected terms, and we use the
subscript $M$ to indicate the classical measurement result register, which we treat above in a
similar notation to the quantum registers.

```{prf:proof}
It holds that:

$$
\begin{aligned}
&\opbra{Q \otimes Z^y} \otimes \opbra{Z^{x + y|_M}} \left(\sum_{m\in \Z_2^M} \E_m \otimes \opket{m}\right)\\
&= \opbra{Q \otimes Z^y} \otimes \opbra{Z^{x + y}} \left( \frac{1}{2^{2|M|+|N|}} \sum_{\substack{a \in \Z_2^M, b \in \Z_2^{M \cup R} \\ P \in \P^{N\setminus R}}} \lambda^Q_{a,b}  \opket{P \otimes Z^b}\opbra{G^\dagger(P \otimes I_{N \cap R} \otimes Z^a)} \otimes \opket{Z^{a + b|_M}} \right) \\
&=\frac{1}{2^{2|M| + |N|}}\sum_{a, b \in \Z_2^M, P \in \P^N} \lambda^P_{a, b}\underbrace{\ip{Q \otimes Z^y}{P \otimes Z^b}}_{2^{|M| + |N|} \delta_{P, Q}\delta_{y,b}} \underbrace{\ip{Z^{x + y|_M}}{Z^{a + b|_M}}}_{2^{|M|}\delta_{x+y|_M,a+b|_M}}\opbra{G^\dagger(P \otimes I_{N \cap R} \otimes Z^a)}
\end{aligned}
$$

Lastly, observe that $\delta_{y,b}\delta_{x+y|_M,a+b|_M} = \delta_{y,b}\delta_{x,a}$, and hence the
final term above collapses down to the single term
$\lambda^Q_{x,y} \opbra{G^\dagger(Q \otimes I_{N \cap R} \otimes Z^x)}$, yielding the desired
result.
```

See {cite}`zhang_generalized_2025` for development beyond this point: the definition of the PTG,
paths through the PTG, and the proof that any properly-defined path corresponds to an experiment.

## References

```{bibliography}
```
