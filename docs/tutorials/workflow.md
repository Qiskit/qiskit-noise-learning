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

# Building a learning experiment by hand

{class}`~.NoiseLearner` is a fixed pipeline behind a single call. In this tutorial we will build the
same kind of experiment out of its parts, so that each one can be inspected, replaced, or reused.

1. A gate set with a single layer on a ring of qubits
2. A 2-local Pauli-Lindblad model
3. Build the learning experiment
4. Generate circuits
5. Set up local simulation
6. Run the program
7. Analyze the data
8. Grade the fit

:::{admonition} Running on real hardware
:class: note

As in the [previous tutorial](noise_learner.md), the circuits run locally under
{class}`~.AerExecutor` with a noise model written out below, so the fitted answer can be checked
against a known truth. Three changes take this to a real device, each flagged again where it
applies:

* **Step 1**: use a real backend in place of `FakeMarrakesh`.
* **Step 5**: skip it.
* **Step 6**: submit through {class}`~qiskit_ibm_runtime.Executor`.
:::

```{code-cell} python
:tags: [remove-cell]

# Emit plotly figures as self-contained HTML so Sphinx can render them.
import plotly.io as pio

pio.renderers.default = "notebook_connected"
```

## 1. A gate set with a single layer on a ring of qubits

A {class}`~.QiskitGateSet` is built from a backend {class}`~qiskit.transpiler.Target` and a subset
of its qubits — here a twelve-qubit ring of `FakeMarrakesh`, with one gate added to it: a layer of
six `CZ` gates covering that ring. Adding the layer also brings in an implicit preparation gate `P`
and measurement gate `M`, whose noise is learned alongside the layer's.

```{code-cell} python
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from qiskit_noise_learning.gate_sets import QiskitGateSet

backend = FakeMarrakesh()

qubit_subset = [*range(25, 30), *range(37, 39), *range(45, 50)]
layer_1_pairs = [(25, 26), (27, 28), (29, 38), (37, 45), (46, 47), (48, 49)]

gate_set = QiskitGateSet(backend.num_qubits, target=backend.target, qubit_subset=qubit_subset)

with gate_set.build_new_gate("layer_1", latex_str=r"\mathrm{CZ}") as builder:
    for pair in layer_1_pairs:
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

{meth}`~.GateSet.draw` shows the layer on the device topology: the ring of qubits in the gate set,
and which pairs of them the layer entangles.

```{code-cell} python
gate_set.draw()
```

## 2. A 2-local Pauli-Lindblad model

A {class}`~.PauliLindbladModel` fixes which Pauli-Lindblad generators the noise is allowed to have.
{meth}`~.PauliLindbladModel.k_local` builds that generator set from the gate set's connectivity —
here every 2-local Pauli on connected qubit pairs of the layer, and single-qubit Paulis for
preparation and measurement.

```{code-cell} python
from qiskit_noise_learning.models import PauliLindbladModel

model = PauliLindbladModel.k_local(gate_set, gate_k={"layer_1": 2, "M": 1, "P": 1})

{name: len(generators) for name, generators in model.generators.items()}
```

That is 168 unknown rates. The model doubles as the linear map from those rates to the
log-fidelities the experiment can measure, which is the map the analysis pipeline inverts.

## 3. Build the learning experiment

An {class}`~.Experiment` is assembled by composing builder stages with `+`. Each stage reads what
earlier stages wrote, so order matters:

- {class}`~.EvenDepthVanillaPaths` and {class}`~.VanillaInstructionSequences` lay down the
  Pauli decay paths for the layer and the instruction sequences that realize them, and
  {class}`~.IdentifyRelations` records which paths each sequence traverses.
- {class}`~.SPAMPaths` adds the paths needed to separate preparation and measurement error from
  the layer's, with {class}`~.GenerateInstructionSequences` producing sequences for them.
- {class}`~.MergeInstructionSequences` and {class}`~.CompleteSequences` fold duplicate
  sequences together and fill in the remaining gate slots.
- {class}`~.BindFragmentDepths` sets the depths at which the repeatable fragment of each path is
  measured.

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
    EvenDepthVanillaPaths()
    + VanillaInstructionSequences()
    + IdentifyRelations()
    + SPAMPaths()
    + GenerateInstructionSequences()
    + MergeInstructionSequences()
    + CompleteSequences()
    + BindFragmentDepths([2, 16, 32, 64, 128])
)

experiment = experiment_builder.run(
    Experiment(fidelity_model=model, shots=20, randomizations=50)
)

print(f"Number of paths: {len(experiment.paths)}")
print(f"Number of instruction sequences: {len(experiment.instruction_sequences)}")
```

## 4. Generate circuits

{class}`~.ExecutorCircuitGenerator` compiles the experiment into a
{class}`~qiskit_ibm_runtime.quantum_program.QuantumProgram` — one parameterized template circuit per
fragment depth — together with a data mapper that records how to read results back out of it.

```{code-cell} python
from qiskit_noise_learning.circuit_generator import ExecutorCircuitGenerator

circuit_generator = ExecutorCircuitGenerator(gate_set)
quantum_program, data_mapper = circuit_generator.generate(experiment)

print(f"Number of template circuits: {len(quantum_program.items)}")
```

The shallowest template shows the anatomy of every circuit in the experiment: a prepared basis, the
twirled layer repeated some number of times, and a measurement. The single-qubit gates around each
layer are parameterized, since twirls are drawn per randomization at run time.

```{code-cell} python
quantum_program.items[0].circuit.draw("mpl", idle_wires=False, fold=False)
```

## 5. Set up local simulation

This step exists only because the tutorial simulates the program rather than running it. An
{class}`~.AerExecutor` runs it on a local Aer simulator, injecting Pauli-Lindblad noise at the
barriers samplomatic places around each layer.

Where the [previous tutorial](noise_learner.md) hand-wrote a sparse noise map, this one gives *every
one* of the model's 168 generators an independent random rate. A {class}`~.ModelData` holds a value
for each generator, and {meth}`~.PauliLindbladModel.to_pauli_lindblad_maps` turns those values into
one {class}`~qiskit.quantum_info.PauliLindbladMap` per gate.

```{code-cell} python
import numpy as np

from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.models import GeneratorIndex

generator_indices = [
    GeneratorIndex(gate_name=name, generator=generator)
    for name, generators in model.generators.items()
    for generator in generators
]

rng = np.random.default_rng(1234)
true_rates = rng.uniform(2e-4, 1.5e-3, size=len(generator_indices))

num_parameters = len(generator_indices)
true_model_data = ModelData.from_arrays(
    parameter_indices=generator_indices,
    parameter_values=true_rates,
    covariance=np.zeros((num_parameters, num_parameters)),
    time_lbs=np.zeros(num_parameters, dtype="datetime64[ns]"),
    time_ubs=np.zeros(num_parameters, dtype="datetime64[ns]"),
)

true_maps = model.to_pauli_lindblad_maps(true_model_data, include_spam=True)
{name: noise_map.num_qubits for name, noise_map in true_maps.items()}
```

Those maps are as wide as the device, since the model is expressed in `backend`'s qubit indexing.
{class}`~.AerExecutor` instead wants each map to be as wide as the layer it applies to, with Pauli
indices running over the layer's qubits in ascending physical order.
{meth}`~qiskit.quantum_info.PauliLindbladMap.keep_qubits` performs exactly that conversion, tracing
out everything off the ring:

```{code-cell} python
noise_dict = {
    name: noise_map.keep_qubits(qubit_subset) for name, noise_map in true_maps.items()
}
{name: noise_map.num_qubits for name, noise_map in noise_dict.items()}
```

The twirls make every circuit Clifford, so the stabilizer method applies. The `root_seed` makes the
simulated data reproducible.

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

{meth}`~.ExecutorCircuitGenerator.collect` pairs the returned bitstrings with the data mapper to
produce a {class}`~.Fit`, the container the analysis stages read from and write to. Fifty
randomizations at five depths takes about a minute.

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

The analysis pipeline composes with `+` in the same way the experiment builder does, and each stage
advances the fit one level: {class}`~.ComputeObservables` turns raw bits into Pauli observables,
{class}`~.CurveFitObservables` fits an exponential decay to each path, and {class}`~.NNLSSolve`
inverts the model to recover non-negative generator rates.

```{code-cell} python
from qiskit_noise_learning.analysis import (
    ComputeObservables,
    CurveFitObservables,
    NNLSSolve,
)

analyzer = ComputeObservables() + CurveFitObservables() + NNLSSolve()

fit = analyzer.run(fit)
```

Plotting the measured observable means against their fitted exponentials, one subplot per `CZ` pair,
shows whether the decays being fitted are exponential at all.

```{code-cell} python
fit.plot_qubit_pair_decays(
    pairs=layer_1_pairs,
    observable_type="means",
    exponential_fit=True,
)
```

The same plot against the *model's* prediction compares the data to the decays implied by the rates
that came out of the non-negative least squares solve.

```{code-cell} python
fit.plot_qubit_pair_decays(
    pairs=layer_1_pairs,
    observable_type="means",
    model_prediction=True,
)
```

## 8. Grade the fit

The previous plot, made quantitative. For every decay path and every fragment depth the experiment
produced an averaged observable value, and the fitted model predicts one: the path's intercept times
its base raised to the depth. `predicted_path_decays`, from
`qiskit_noise_learning.analysis.utils`, returns those `(bases, intercepts)`.

```{code-cell} python
from qiskit_noise_learning.analysis.utils import predicted_path_decays

dataset = fit.observable_data.dataset
path_column = dataset["unbound_path"].data
depth_column = dataset["fragment_depth"].data
observables = dataset["observables"].data

decay_paths = [
    path for path in dict.fromkeys(path_column) if path.is_unbound and path.repeatable_fragment
]
bases, intercepts = predicted_path_decays(model, fit.model_data, decay_paths)

measured, predicted, errors_of_mean = [], [], []
for row, (path, depth) in enumerate(zip(path_column, depth_column)):
    if path not in bases:
        continue
    values = observables[row][~np.isnan(observables[row])]
    measured.append(values.mean())
    errors_of_mean.append(values.std(ddof=1) / np.sqrt(values.size))
    predicted.append(intercepts[path] * bases[path] ** depth)

measured = np.array(measured)
predicted = np.array(predicted)
errors_of_mean = np.array(errors_of_mean)
residuals = np.abs(measured - predicted)

print(f"points compared:        {len(measured)}")
print(f"correlation:            {np.corrcoef(measured, predicted)[0, 1]:.4f}")
print(f"median |residual|:      {np.median(residuals):.4f}")
print(f"median standard error:  {np.median(errors_of_mean):.4f}")
print(f"within two std. errors: {(residuals < 2 * errors_of_mean).mean():.1%}")
```

The residuals are *smaller than the statistical error of the data they are residuals of* — a median
of about half a standard error, with almost every point inside two. The fitted model
reproduces the experiment to within its shot noise, at every depth rather than only the shallow ones
where the signal is largest.

:::{note}
Do not grade a fit by comparing its recovered per-generator *rates* against the injected ones. An
experiment of this shape is rank deficient, and beyond that a gauge freedom leaves many different
rate assignments predicting identical observables, so the solve is free to return any of them.
`true_rates` is not even this model's own description of the channel that was applied:
{class}`~.AerExecutor` injects noise *after* the layer, whereas {class}`~.PauliLindbladModel` models
a unitary gate's noise as occurring *before* it, and conjugating the injected map back through the
layer takes some of its 2-local generators outside the model's generator set entirely. What the
experiment measures, graded above, is what a learned model supports.
:::
