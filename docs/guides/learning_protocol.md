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
protocol accessible through the high-level protocol functions {func}`~.prepare_vanilla_program` and
{func}`~.process_vanilla_result`.

1. Define the gate
2. Set up local simulation
3. Prepare the quantum program
4. Run the quantum program
5. Read the results

:::{admonition} Running on real hardware
:class: note

The circuits below are simulated locally, so this walkthrough needs no IBM Quantum&reg;
credentials. To alternatively run it on a real device:

* **Step 1**: replace {class}`~qiskit_ibm_runtime.fake_provider.FakeMarrakesh` with a real backend.
* **Step 2**: pass that backend as the executor's mode, and drop the simulator options.
:::

## 1. Define the gate

The gate is a {class}`~qiskit.circuit.BoxOp` holding a twirled layer of six disjoint `CZ` gates.

```{code-cell} python
from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
from samplomatic import InjectNoise, Tag, Twirl

backend = FakeMarrakesh()

cz_pairs = [(91, 92), (93, 94), (95, 99), (98, 111), (112, 113), (114, 115)]

circuit = QuantumCircuit(backend.num_qubits)
with circuit.box([Twirl(), Tag("cz_gate"), InjectNoise("cz_gate")]):
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

The Pauli indices inside each map are local to the gate, in ascending physical-qubit order:

```{code-cell} python
cz_qubits = sorted({qubit for pair in cz_pairs for qubit in pair})
local = {qubit: index for index, qubit in enumerate(cz_qubits)}
local
```

Define the noise with a correlated `ZZ` term and a weaker `XX` term on each `CZ` pair, plus a
single-qubit `Z` term everywhere; measurement gets a bit-flip term per qubit.

```{code-cell} python
from qiskit.quantum_info import PauliLindbladMap

num_qubits = len(cz_qubits)

cz_noise = PauliLindbladMap.from_sparse_list(
    [("ZZ", [local[a], local[b]], 8e-4) for a, b in cz_pairs]
    + [("XX", [local[a], local[b]], 4e-4) for a, b in cz_pairs]
    + [("Z", [index], 3e-4) for index in range(num_qubits)],
    num_qubits=num_qubits,
)

readout_noise = PauliLindbladMap.from_sparse_list(
    [("X", [index], 5e-3) for index in range(num_qubits)], num_qubits=num_qubits
)
```

Build a box for measurement to learn.

```{code-cell} python
measurement = QuantumCircuit(backend.num_qubits, num_qubits)
with measurement.box([Twirl(), Tag("readout"), InjectNoise("readout")]):
    for clbit, qubit in enumerate(cz_qubits):
        measurement.measure(qubit, clbit)
```

Each entry of `layer_noise_model` pairs one of those boxes with the map to apply to it. Use the
stabilizer simulation method, and set `seed_simulator` to make the simulated data reproducible.

```{code-cell} python
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Executor

executor = Executor(
    mode=AerSimulator(method="stabilizer"),
    options={
        "simulator": {
            "layer_noise_model": [
                (circuit[0], cz_noise),
                (measurement[0], readout_noise),
            ],
            "seed_simulator": 1234,
        }
    },
)
```

:::{admonition} Running on real hardware
:class: note

```python
executor = Executor(mode=backend)
```
:::

## 3. Prepare the quantum program

Use {func}`~.prepare_vanilla_program` to build the quantum program containing the learning
experiments.

```{code-cell} python
from qiskit_noise_learning.protocols import prepare_vanilla_program

program = prepare_vanilla_program(
    backend,
    [circuit[0], measurement[0]],
    fragment_depths=[2, 16, 64, 128],
    num_randomizations=50,
    shots_per_randomization=20,
)

print(f"Number of template circuits: {len(program.items)}")
```

## 4. Run the quantum program

Submit the quantum program. This step is the same whether the executor is pointed at a simulator or
at a real device.

```{code-cell} python
result = executor.run(program).result()
```

## 5. Read the results

Process the result with {func}`~.process_vanilla_result`, which reports the noise of each
instruction the program was built from, in the order they were given.

```{code-cell} python
from qiskit_noise_learning.protocols import process_vanilla_result

learned = process_vanilla_result(result)
learned.gate_names
```

Those are the names the boxes were keyed by: the reference from each `InjectNoise` annotation.
`noise_maps[0]` is therefore the noise of the `CZ` layer and `noise_maps[1]` the readout noise.

```{code-cell} python
cz_learned = learned.noise_maps[0]
cz_learned.num_terms
```

Each map acts on the qubits of its own instruction, its qubit `j` being the `j`-th of those in
ascending physical order --- the same indexing `cz_noise` was defined with above. So the learned
map can be compared with the injected one term by term:

```{code-cell} python
def rates(noise_map):
    return {
        (pauli, tuple(sorted(idxs))): rate
        for pauli, idxs, rate in noise_map.to_sparse_list()
    }


a, b = cz_pairs[0]
term = ("ZZ", (local[a], local[b]))
rates(cz_noise)[term], rates(cz_learned)[term]
```

The fit the noise was read out of comes back alongside it, holding the data at every level of the
analysis. Use it to plot per-qubit-pair fidelity decays: both the data and the exponential fit.

```{code-cell} python
learned.fit.plot_qubit_pair_decays(
    pairs=cz_pairs,
    observable_type="means",
    exponential_fit=True,
)
```
