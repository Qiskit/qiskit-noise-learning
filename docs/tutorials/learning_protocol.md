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

# Learn the noise model of a gate

This tutorial demonstrates learning a noise model for a unitary gate with a standard learning
protocol accessible through the high-level protocol functions {func}`~.prepare_learning_program` and
{func}`~.process_learning_result`.

1. Define the gate
2. Set up local simulation
3. Prepare the quantum program
4. Run the quantum program
5. Read the results

:::{admonition} Running on real hardware
:class: note

The circuits below are simulated locally, so this walkthrough needs no IBM Quantum&reg;
credentials. Two changes take it to a real device, each flagged again where it applies:

* **Step 1**: replace {class}`~qiskit_ibm_runtime.fake_provider.FakeMarrakesh` with a real backend.
* **Step 2**: skip it, and submit through {class}`~qiskit_ibm_runtime.Executor` in step 4.
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

## 3. Prepare the quantum program

Use {func}`~.prepare_learning_program` to build the quantum program containin gthe learning
experiments.

```{code-cell} python
from qiskit_noise_learning.protocols import prepare_learning_program

program = prepare_learning_program(
    backend,
    [circuit[0]],
    fragment_depths=[2, 16, 64, 128],
    num_randomizations=50,
    shots_per_randomization=20,
)

print(f"Number of template circuits: {len(program.items)}")
```

## 4. Run the quantum program

Submit the quantum program to your backend of choice. Here, we use the simulated executor; to
simulate against a real device use {class}`~qiskit_ibm_runtime.Executor`.

```{code-cell} python
result = executor.run(program).result()
```

:::{admonition} Running on real hardware
:class: note

```python
from qiskit_ibm_runtime import Executor

result = Executor(mode=backend).run(program).result()
```
:::

## 5. Read the results

Process the result with {func}`~.process_learning_result`, obstaining a {class}`~.Fit` object.

```{code-cell} python
from qiskit_noise_learning.protocols import process_learning_result

fit = process_learning_result(result)
```

Use the fit to plot per-qubit-pair fidelity decays: both the data and the exponential fit.

```{code-cell} python
fit.plot_qubit_pair_decays(
    pairs=cz_pairs,
    observable_type="means",
    exponential_fit=True,
)
```

Read out the learned noise, keyed by the name from the `InjectNoise` annotation.

```{code-cell} python
from qiskit_noise_learning.models import split_pauli_lindblad_model

pauli_lindblad_model = split_pauli_lindblad_model(fit.model).model
learned = pauli_lindblad_model.to_pauli_lindblad_maps(fit.model_data)["cz_gate"]
learned.num_terms
```
