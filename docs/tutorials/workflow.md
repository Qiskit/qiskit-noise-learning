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

# Building a learning experiment

In this tutorial we will build a standard learning experiment from scratch, and use it to learn a
model from simulated noisy data.

1. Define a gate set on a ring of qubits
2. Choose a 2-local Pauli-Lindblad model
3. Build the learning experiment
4. Generate circuits
5. Set up local simulation
6. Run the program
7. Analyze the data

:::{admonition} Running on real hardware
:class: note

The circuits below are simulated locally, so this tutorial needs no IBM Quantum credentials. Three
changes take it to a real device, each flagged again where it applies:

* **Step 1**: replace {class}`~qiskit_ibm_runtime.fake_provider.FakeMarrakesh` with a real backend.
* **Step 5**: skip it.
* **Step 6**: submit through {class}`~qiskit_ibm_runtime.Executor`.
:::

```{code-cell} python
:tags: [remove-cell]

# Emit plotly figures as HTML for a static page that supplies its own MathJax.
import plotly.io as pio

from qiskit_noise_learning.visualizations import RENDERER_NAME, html_page_renderer

pio.renderers[RENDERER_NAME] = html_page_renderer()
pio.renderers.default = RENDERER_NAME
```

## 1. Define a gate set on a ring of qubits

Build a {class}`~.QiskitGateSet` from a backend {class}`~qiskit.transpiler.Target` and a subset of
its qubits — here a twelve-qubit ring of
{class}`~qiskit_ibm_runtime.fake_provider.FakeMarrakesh` — then add one gate to it: a layer of six
`CZ` gates covering that ring. By default the gate set is initialized with a preparation gate `P`
and a measurement gate `M`.

```{code-cell} python
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from qiskit_noise_learning.gate_sets import QiskitGateSet

backend = FakeMarrakesh()

qubit_subset = [*range(25, 30), *range(37, 39), *range(45, 50)]
cz_pairs = [(25, 26), (27, 28), (29, 38), (37, 45), (46, 47), (48, 49)]

gate_set = QiskitGateSet(backend.num_qubits, target=backend.target, qubit_subset=qubit_subset)

with gate_set.build_new_gate("cz_gate", latex_str=r"\mathrm{CZ}") as builder:
    for pair in cz_pairs:
        builder.circuit.cz(*pair)

list(gate_set)
```

:::{admonition} Running on real hardware
:class: note

```python
from qiskit_ibm_runtime import QiskitRuntimeService

backend = QiskitRuntimeService().backend("ibm_marrakesh")
```
:::

Call {meth}`~.GateSet.draw` to see the gate on the device topology: the ring of qubits in the gate
set, and which pairs of them the gate entangles.

```{code-cell} python
gate_set.draw()
```

## 2. Choose a 2-local Pauli-Lindblad model

Decide which Pauli-Lindblad generators the noise is allowed to have with a
{class}`~.PauliLindbladModel`. Build that generator set from the gate set's connectivity with
{meth}`~.PauliLindbladModel.k_local` — here every 2-local Pauli on connected qubit pairs of the
unitary gate, and single-qubit Paulis for preparation and measurement.

```{code-cell} python
from qiskit_noise_learning.models import PauliLindbladModel

model = PauliLindbladModel.k_local(gate_set, gate_k={"cz_gate": 2, "M": 1, "P": 1})
```

Display the unknown parameter count for the noise model of each gate.

```{code-cell} python
{name: len(generators) for name, generators in model.generators.items()}
```

## 3. Build the learning experiment

Assemble an {class}`~.Experiment` by composing builder stages. Each stage reads what earlier stages
wrote, so order matters:

```{code-cell} python
from qiskit_noise_learning.experiment_builder import (
    BindFragmentDepths,
    CompleteSequences,
    EvenDepthVanillaPaths,
    Experiment,
    GenerateInstructionSequences,
    IdentifyRelations,
    MergeInstructionSequences,
    SPAMPaths,
    VanillaInstructionSequences,
)

experiment_builder = (
    # Add standard vanilla learning paths and instruction sequences, and identify relations
    EvenDepthVanillaPaths()
    + VanillaInstructionSequences()
    + IdentifyRelations()
    # Add paths for learning SPAM, then generate and merge instruction sequences for measuring them
    + SPAMPaths()
    + GenerateInstructionSequences()
    + MergeInstructionSequences()
    # Finalize by completing the instruction sequences and setting experiment depths
    + CompleteSequences()
    + BindFragmentDepths([2, 16, 32, 64])
)

experiment = experiment_builder.run(
    Experiment(fidelity_model=model, shots=20, randomizations=50)
)

print(f"Number of paths: {len(experiment.paths)}")
print(f"Number of instruction sequences: {len(experiment.instruction_sequences)}")
```

Observe design matrix rank:

```{code-cell} python
print(f"Design matrix rank: {experiment.design_matrix.rank}")
```

## 4. Generate circuits

Compile the experiment with an {class}`~.ExecutorCircuitGenerator`, generating a
{class}`~qiskit_ibm_runtime.quantum_program.QuantumProgram` — one parameterized template circuit
per fragment depth — together with a data mapper that records how to interpret results.

```{code-cell} python
from qiskit_noise_learning.circuit_generator import ExecutorCircuitGenerator

circuit_generator = ExecutorCircuitGenerator(gate_set)
quantum_program, data_mapper = circuit_generator.generate(experiment)

print(f"Number of template circuits: {len(quantum_program.items)}")
```

Draw a template circuit to inspect.

```{code-cell} python
quantum_program.items[0].circuit.draw("mpl", idle_wires=False, fold=False)
```

## 5. Set up local simulation

An {class}`~.AerExecutor` runs a program on a local Aer simulator, injecting Pauli-Lindblad
noise at the barriers samplomatic places around each twirled gate.

Unlike the hand-picked noise of the {class}`~.NoiseLearner` tutorial, give *every one* of the
model's 168 generators an independent random rate. The model's generators are already exactly the
Paulis to put in a {class}`~qiskit.quantum_info.PauliLindbladMap`, so pair each one with a rate and
build the map per gate directly:

```{code-cell} python
import numpy as np
from qiskit.quantum_info import PauliLindbladMap

rng = np.random.default_rng(1234)

true_maps = {
    name: PauliLindbladMap.from_terms(
        [
            PauliLindbladMap.GeneratorTerm(rate, generator)
            for generator, rate in zip(generators, rng.uniform(2e-4, 1.5e-3, len(generators)))
        ]
    )
    for name, generators in model.generators.items()
}
{name: noise_map.num_qubits for name, noise_map in true_maps.items()}
```

Those maps are as wide as the device, since the model is expressed in `backend`'s qubit indexing,
but {class}`~.AerExecutor` wants each map to be as wide as the gate it applies to, with Pauli
indices running over the gate's qubits in ascending physical order. Narrow them with
{meth}`~qiskit.quantum_info.PauliLindbladMap.keep_qubits`, which traces out everything off the ring:

```{code-cell} python
noise_dict = {
    name: noise_map.keep_qubits(qubit_subset) for name, noise_map in true_maps.items()
}
{name: noise_map.num_qubits for name, noise_map in noise_dict.items()}
```

Instantiate {class}`~.AerExecutor` with the stabilizer method. Set `root_seed` to make the simulated
data reproducible.

```{code-cell} python
from qiskit_aer import AerSimulator

from qiskit_noise_learning.aer_executor import AerExecutor

executor = AerExecutor(
    AerSimulator(method="stabilizer"), noise_dict=noise_dict, root_seed=42
)
```

:::{admonition} Running on real hardware
:class: note

Skip this step entirely.
:::

## 6. Run the program

Run the program, then pair the returned bitstrings with the data mapper using
{meth}`~.ExecutorCircuitGenerator.collect`, which produces a {class}`~.Fit` — the container the
analysis stages read from and write to.

```{code-cell} python
job = executor.run(quantum_program)
fit = circuit_generator.collect(job.result(), data_mapper)
```

:::{admonition} Running on real hardware
:class: note

```python
from qiskit_ibm_runtime import Executor

executor = Executor(mode=backend)
```
:::

## 7. Analyze the data

Build an analysis pipeline around non-negative least squares fitting of the model.

```{code-cell} python
from qiskit_noise_learning.analysis import (
    ComputeObservables,  # computes observables from raw data
    CurveFitObservables,  # performs exponential fitting
    NNLSSolve,  # solves model with non-negative least squares
)

analyzer = ComputeObservables() + CurveFitObservables() + NNLSSolve()

fit = analyzer.run(fit)
```

Plot the measured observable means against their fitted exponentials, one subplot per `CZ` pair, to
see whether the decays being fitted are exponential at all.

```{code-cell} python
fit.plot_qubit_pair_decays(
    pairs=cz_pairs,
    observable_type="means",
    exponential_fit=True,
)
```

Swap `exponential_fit` for `model_prediction` to compare the same data against the decays implied by
the rates that came out of the non-negative least squares solve.

```{code-cell} python
fit.plot_qubit_pair_decays(
    pairs=cz_pairs,
    observable_type="means",
    model_prediction=True,
)
```
