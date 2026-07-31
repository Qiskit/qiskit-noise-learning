---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Learning a layer with NoiseLearner

{class}`~.NoiseLearner` is the high-level entry point: give it a backend and a boxed layer of
gates, and it builds the learning experiment, runs it, and fits a Pauli-Lindblad model, all
behind a single {meth}`~.NoiseLearner.run` call.

This tutorial learns the noise of one layer of six `CZ` gates on a ring of twelve qubits.
Rather than send circuits to hardware, it executes them locally with {class}`~.AerExecutor`
and a noise model written out below, so the learned answer can be compared against a known
truth.

```{code-cell} python
:tags: [remove-cell]

# Emit plotly figures as self-contained HTML so Sphinx can render them.
import plotly.io as pio

pio.renderers.default = "notebook_connected"
```

## Define the layer

The layer is a {class}`~qiskit.circuit.BoxOp` holding six disjoint `CZ` gates. Two samplomatic
annotations tell the toolkit what to do with it:

- `Twirl()` marks the box as a layer to Pauli-twirl, which is what turns the layer's noise into
  a Pauli channel in the first place.
- `InjectNoise("layer")` names the box. That name is the key under which the learned noise map
  is reported, and — because this tutorial simulates the layer instead of running it — also the
  key under which noise is injected.

```{code-cell} python
from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
from samplomatic import InjectNoise, Twirl

backend = FakeMarrakesh()

layer_pairs = [(91, 92), (93, 94), (95, 99), (98, 111), (112, 113), (114, 115)]

circuit = QuantumCircuit(backend.num_qubits)
with circuit.box([Twirl(), InjectNoise("layer")]):
    for pair in layer_pairs:
        circuit.cz(*pair)
```

## Choose the noise to be learned

An {class}`~.AerExecutor` runs a program on a local Aer simulator, injecting Pauli-Lindblad
noise at the barriers samplomatic places around each twirled layer. Its `noise_dict` maps a
layer name to the noise applied to that layer; `"P"` and `"M"` are the implicit state
preparation and measurement layers.

The Pauli indices inside each map are local to the layer, in ascending physical-qubit order.
For this layer the twelve physical qubits map to local indices `0`–`11` as follows:

```{code-cell} python
layer_qubits = sorted({qubit for pair in layer_pairs for qubit in pair})
local = {qubit: index for index, qubit in enumerate(layer_qubits)}
local
```

The layer gets a correlated `ZZ` term and a weaker `XX` term on each `CZ` pair, plus a
single-qubit `Z` term everywhere; preparation and measurement each get a bit-flip term per
qubit. All rates are chosen to be roughly device-realistic.

```{code-cell} python
from qiskit.quantum_info import PauliLindbladMap

num_qubits = len(layer_qubits)

layer_noise = PauliLindbladMap.from_sparse_list(
    [("ZZ", [local[a], local[b]], 8e-4) for a, b in layer_pairs]
    + [("XX", [local[a], local[b]], 4e-4) for a, b in layer_pairs]
    + [("Z", [index], 3e-4) for index in range(num_qubits)],
    num_qubits=num_qubits,
)

spam_noise = PauliLindbladMap.from_sparse_list(
    [("X", [index], 5e-3) for index in range(num_qubits)], num_qubits=num_qubits
)

noise_dict = {"layer": layer_noise, "P": spam_noise, "M": spam_noise}
```

## Run the learner

{class}`~.LearningOptions` controls the shape of the experiment: how deep the twirled layer is
repeated, and how many randomizations and shots are spent at each depth.

Passing an `executor` diverts the generated program away from the backend. Leave it out and
{class}`~.NoiseLearner` submits to `backend` through IBM Quantum instead, which is what you
want against real hardware. The `root_seed` makes the simulated data reproducible.

:::{note}
Supplying `executor` alongside `backend` is redundant, and exists so that an experiment can be
run against a simulated executor as it is here. It is not a stable interface.
:::

```{code-cell} python
from qiskit_aer import AerSimulator

from qiskit_noise_learning.aer_executor import AerExecutor
from qiskit_noise_learning.noise_learner import LearningOptions, NoiseLearner

options = LearningOptions(
    fragment_depths=[2, 16, 64, 128],
    num_randomizations=50,
    shots_per_randomizations=20,
)

executor = AerExecutor(
    AerSimulator(method="stabilizer"), noise_dict=noise_dict, root_seed=1234
)
learner = NoiseLearner(backend, options=options, executor=executor)

job = learner.run([circuit[0]])
result = job.result()
```

Because the layer is Clifford and twirled with Paulis, the whole experiment simulates in the
stabilizer formalism — no state vector required, which is why twelve qubits and 128-deep
circuits are cheap here.

## Read the learned noise

{meth}`~.NoiseLearnerResult.to_dict` returns one
{class}`~qiskit.quantum_info.PauliLindbladMap` per learned layer, keyed by the name from the
`InjectNoise` annotation. Unlike the maps handed to the executor, these are expressed in the
backend's own qubit indexing.

```{code-cell} python
learned = result.to_dict()["layer"]
learned.num_terms
```

By default {class}`~.NoiseLearner` fits a 2-local model, so the map carries a term for every
Pauli supported on a connected pair of the layer's qubits — 144 of them, of which only 24 were
actually given a nonzero rate above. Lining those 24 up against what came out of the fit:

```{code-cell} python
learned_rates = {
    (pauli, tuple(sorted(indices))): rate for pauli, indices, rate in learned.to_sparse_list()
}

for pauli, indices, injected in layer_noise.to_sparse_list():
    qubits = tuple(sorted(layer_qubits[index] for index in indices))
    print(
        f"{pauli:2s} {str(list(qubits)):12s} "
        f"injected {injected:.1e}   learned {learned_rates[pauli, qubits]:.2e}"
    )
```

The six `ZZ` rates land within a few percent of their injected `8e-4`, and the twelve `Z` rates
within about thirty percent of `3e-4` — reasonable, given that this ran fifty randomizations of
twenty shots.

The `XX` column has one conspicuous entry: `XX` on qubits 91 and 92 came back at exactly zero,
even though it was injected at `4e-4` like all the others. Nothing went wrong; that rate is
simply not measurable. Looking at the largest learned terms that were *not* injected shows where
its weight went:

```{code-cell} python
injected_terms = {
    (pauli, tuple(sorted(layer_qubits[index] for index in indices)))
    for pauli, indices, _ in layer_noise.to_sparse_list()
}

spurious = [
    (pauli, indices, rate)
    for (pauli, indices), rate in learned_rates.items()
    if (pauli, indices) not in injected_terms and rate > 0
]
for pauli, indices, rate in sorted(spurious, key=lambda term: -term[2])[:5]:
    print(f"{pauli:2s} {str(list(indices)):12s} {rate:.2e}")
```

`YY` on qubits 91 and 92, at `3.8e-4` — almost exactly the rate that went missing from `XX`.
Every other uninjected term is smaller by a factor of seven or more.

This is a genuine degeneracy rather than a fluke. Conjugating $X \otimes X$ by a `CZ` gives
$Y \otimes Y$, so an experiment built on an even number of layer repetitions only ever sees the
two together: no amount of data taken this way separates them. Injecting `YY` at `4e-4` instead
of `XX` splits the weight exactly the same way — five pairs assigned to `XX`, this one to `YY`,
with matching totals. Which of the pair carries the weight is decided by the non-negative least
squares solve, not by the data.

What *is* determined is any quantity that depends on the degenerate pair only through its total,
which is why an aggregate check is the honest one to report:

```{code-cell} python
injected_total = float(layer_noise.rates.sum())
learned_total = float(learned.rates.sum())
print(f"injected total rate: {injected_total:.5f}")
print(f"learned total rate:  {learned_total:.5f}")
```

Per-generator agreement even this good is more than should be expected in general, and it is not
what a learning experiment guarantees. It holds here because the injected noise is sparse and
lies in exactly the basis the model fits, which suits the non-negative least squares solve. The
`XX`/`YY` pair above is the one place that breaks down, and it is the small visible corner of a
larger effect: the experiment constrains the model only through the quantities it measures, and
whole directions in rate space leave those untouched. The [next tutorial](workflow.md) injects a
dense set of random rates instead, where the effect is unmissable, and shows what can be checked
in its place.

## Inspect the underlying fit

Everything the analysis pipeline produced is reachable through
{attr}`~.NoiseLearnerResult.fit`, including the per-qubit-pair fidelity decays the exponential
fits were taken from. One subplot per `CZ` pair, one trace per Pauli:

```{code-cell} python
result.fit.plot_qubit_pair_decays(
    pairs=layer_pairs,
    observable_type="means",
    exponential_fit=True,
)
```

The [next tutorial](workflow.md) rebuilds this same experiment stage by stage.
