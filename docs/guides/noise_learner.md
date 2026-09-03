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

# Learn the noise model of a gate with NoiseLearner

This guide demonstrates using {class}`~.NoiseLearner` to learn a noise model for a
unitary gate.

1. Define the gate
2. Set up local simulation
3. Run the learner
4. Read the results

:::{admonition} Running on real hardware
:class: note

The circuits below are simulated locally, so this walkthrough needs no IBM Quantum&reg; credentials. Two
changes take it to a real device, each flagged again where it applies:

* **Step 1**: replace {class}`~qiskit_ibm_runtime.fake_provider.FakeMarrakesh` with a real backend.
* **Step 2**: skip it, and drop the `executor` argument in step 3.
:::

## 1. Define the gate

The gate whose noise we will learn is a {class}`~qiskit.circuit.BoxOp` holding a layer of six
disjoint `CZ` gates, with two Samplomatic annotations: `Twirl()` marks the box for Pauli twirling,
and `InjectNoise("cz_gate")` names it. That name is the key under which the learned noise map is
reported &mdash; and, because this guide simulates the gate, also the key under which noise is
injected.

```{code-cell} python
from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
from samplomatic import InjectNoise, Twirl

backend = FakeMarrakesh()

cz_pairs = [(91, 92), (93, 94), (95, 99), (98, 111), (112, 113), (114, 115)]

circuit = QuantumCircuit(backend.num_qubits)
with circuit.box([Twirl(), InjectNoise("cz_gate")]):
    for pair in cz_pairs:
        circuit.cz(*pair)
```

:::{admonition} Running on real hardware
:class: note

```python
from qiskit_ibm_runtime import QiskitRuntimeService

backend = QiskitRuntimeService().backend("ibm_marrakesh")
```
:::

## 2. Set up local simulation

An {class}`~.AerExecutor` runs a program on a local Aer simulator, injecting Pauli-Lindblad
noise at the barriers Samplomatic places around each twirled gate.

The Pauli indices inside each map are local to the gate, in ascending physical-qubit order:

```{code-cell} python
cz_qubits = sorted({qubit for pair in cz_pairs for qubit in pair})
local = {qubit: index for index, qubit in enumerate(cz_qubits)}
local
```

The gate gets a correlated `ZZ` term and a weaker `XX` term on each `CZ` pair, plus a single-qubit
`Z` term everywhere; preparation and measurement each get a bit-flip term per qubit.

```{code-cell} python
from qiskit.quantum_info import PauliLindbladMap

num_qubits = len(cz_qubits)

cz_noise = PauliLindbladMap.from_sparse_list(
    [("ZZ", [local[a], local[b]], 8e-4) for a, b in cz_pairs]
    + [("XX", [local[a], local[b]], 4e-4) for a, b in cz_pairs]
    + [("Z", [index], 3e-4) for index in range(num_qubits)],
    num_qubits=num_qubits,
)

spam_noise = PauliLindbladMap.from_sparse_list(
    [("X", [index], 5e-3) for index in range(num_qubits)], num_qubits=num_qubits
)

noise_dict = {"cz_gate": cz_noise, "P": spam_noise, "M": spam_noise}
```

Instantiate {class}`~.AerExecutor` with the stabilizer method. Set `root_seed` to make the simulated
data reproducible.

```{code-cell} python
from qiskit_aer import AerSimulator

from qiskit_noise_learning.aer_executor import AerExecutor

executor = AerExecutor(
    AerSimulator(method="stabilizer"), noise_dict=noise_dict, root_seed=1234
)
```

:::{admonition} Running on real hardware
:class: note

Skip this step entirely.
:::

## 3. Run the learner

{class}`~.LearningOptions` controls the shape of the experiment: how deep the twirled gate is
repeated, and how many randomizations and shots are spent at each depth. Passing `executor` diverts
the generated program to the simulator; leave it out and {class}`~.NoiseLearner` submits to
`backend` through IBM Quantum instead.

```{code-cell} python
from qiskit_noise_learning.noise_learner import LearningOptions, NoiseLearner

options = LearningOptions(
    fragment_depths=[2, 16, 64, 128],
    num_randomizations=50,
    shots_per_randomizations=20,
)

learner = NoiseLearner(backend, options=options, executor=executor)

job = learner.run([circuit[0]])
result = job.result()
```

:::{admonition} Running on real hardware
:class: note

```python
learner = NoiseLearner(backend, options=options)
```
:::

## 4. Read the results

Everything the analysis pipeline produced is reachable through {attr}`~.NoiseLearnerResult.fit`.
Use the fit to plot per-qubit-pair fidelity decays: both the data and the exponential fit.

```{code-cell} python
result.fit.plot_qubit_pair_decays(
    pairs=cz_pairs,
    observable_type="means",
    exponential_fit=True,
)
```

Extract the learned noise from {meth}`~.NoiseLearnerResult.to_dict`: one
{class}`~qiskit.quantum_info.PauliLindbladMap` per learned gate, keyed by the name from the
`InjectNoise` annotation, and expressed in the backend's own qubit indexing rather than that of the gate.

```{code-cell} python
learned = result.to_dict()["cz_gate"]
learned.num_terms
```

By default {class}`~.NoiseLearner` fits a 2-local model, so the map carries a term for every Pauli
supported on a connected pair of the gate's qubits &mdash; 144 of them, of which only 24 were given a
nonzero rate in step 2.
