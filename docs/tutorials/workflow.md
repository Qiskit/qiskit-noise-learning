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

{class}`~.NoiseLearner` is a fixed pipeline behind a single call. This tutorial builds the same
kind of experiment out of its parts, so that each one can be inspected, replaced, or reused:

1. a **gate set** describing the layer to be learned and the device it runs on,
2. a **fidelity model** relating Pauli-Lindblad rates to measurable fidelities,
3. an **experiment** built by composing builder stages,
4. **circuit generation** into a runnable quantum program,
5. an **analysis pipeline** that turns bitstrings back into model parameters.

As in the [previous tutorial](noise_learner.md), the circuits run locally under
{class}`~.AerExecutor` with a noise model written out below, so the fitted answer can be
checked against a known truth.

```{code-cell} python
:tags: [remove-cell]

# Emit plotly figures as self-contained HTML so Sphinx can render them.
import plotly.io as pio

pio.renderers.default = "notebook_connected"
```

## 1. A gate set with a single layer on a ring of qubits

A {class}`~.QiskitGateSet` is built from a backend {class}`~qiskit.transpiler.Target` and a
subset of its qubits. The subset here is a twelve-qubit ring of `FakeMarrakesh`, and the one
gate added to it is a layer of six `CZ` gates covering that ring.

Adding the layer also gives the gate set two more gates for free: an implicit preparation gate
`P` and measurement gate `M`, whose noise is learned alongside the layer's.

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

{meth}`~.GateSet.draw` shows the layer on the device topology — the ring of qubits in the gate
set, and which pairs of them the layer entangles.

```{code-cell} python
gate_set.draw()
```

## 2. A 2-local Pauli-Lindblad model

A {class}`~.PauliLindbladModel` fixes which Pauli-Lindblad generators the noise is allowed to
have. {meth}`~.PauliLindbladModel.k_local` builds the generator set from the gate set's
connectivity: `gate_k` asks for every 2-local Pauli on connected qubit pairs of the layer, and
single-qubit Paulis for preparation and measurement.

```{code-cell} python
from qiskit_noise_learning.models import PauliLindbladModel

model = PauliLindbladModel.k_local(gate_set, gate_k={"layer_1": 2, "M": 1, "P": 1})

{name: len(generators) for name, generators in model.generators.items()}
```

That is 168 unknown rates in total. The model is also the linear map from those rates to the
log-fidelities the experiment can actually measure, which is what the analysis pipeline
inverts.

## 3. Build the learning experiment

An {class}`~.Experiment` is assembled by composing builder stages with `+`. Each stage reads
what earlier stages wrote, so order matters:

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
{class}`~qiskit_ibm_runtime.quantum_program.QuantumProgram` — one parameterized template
circuit per fragment depth — together with a data mapper that records how to read results back
out of it.

```{code-cell} python
from qiskit_noise_learning.circuit_generator import ExecutorCircuitGenerator

circuit_generator = ExecutorCircuitGenerator(gate_set)
quantum_program, data_mapper = circuit_generator.generate(experiment)

print(f"Number of template circuits: {len(quantum_program.items)}")
```

The shallowest template shows the anatomy of every circuit in the experiment: a prepared basis,
the twirled layer repeated some number of times, and a measurement. The single-qubit gates
around each layer are parameterized — the twirls are drawn per randomization at run time rather
than baked into the circuit.

```{code-cell} python
quantum_program.items[0].circuit.draw("mpl", idle_wires=False, fold=False)
```

## 5. Choose the noise to be learned

On hardware, this program would go to an {class}`~qiskit_ibm_runtime.Executor`. Here it goes to
an {class}`~.AerExecutor`, which simulates it locally and injects Pauli-Lindblad noise at the
barriers samplomatic places around each layer.

Where the [previous tutorial](noise_learner.md) hand-wrote a sparse noise map, this one gives
*every one* of the model's 168 generators an independent random rate. The model itself provides
the machinery: a {class}`~.ModelData` holds a value for each generator, and
{meth}`~.PauliLindbladModel.to_pauli_lindblad_maps` turns those values into one
{class}`~qiskit.quantum_info.PauliLindbladMap` per gate.

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

Those maps are as wide as the device, since the model is expressed in `backend`'s qubit
indexing. {class}`~.AerExecutor` instead wants each map to be as wide as the layer it applies
to, with Pauli indices running over the layer's qubits in ascending physical order.
{meth}`~qiskit.quantum_info.PauliLindbladMap.keep_qubits` performs exactly that conversion,
tracing out everything off the ring:

```{code-cell} python
noise_dict = {
    name: noise_map.keep_qubits(qubit_subset) for name, noise_map in true_maps.items()
}
{name: noise_map.num_qubits for name, noise_map in noise_dict.items()}
```

## 6. Run the program

{meth}`~.ExecutorCircuitGenerator.collect` pairs the returned bitstrings with the data mapper to
produce a {class}`~.Fit`, the container the analysis stages read from and write to.

Simulating fifty randomizations at five depths takes about a minute, since the twirls make every
circuit Clifford and so the stabilizer method applies.

```{code-cell} python
from qiskit_aer import AerSimulator

from qiskit_noise_learning.aer_executor import AerExecutor

executor = AerExecutor(
    AerSimulator(method="stabilizer"), noise_dict=noise_dict, root_seed=42
)

job = executor.run(quantum_program)
fit = circuit_generator.collect(job.result(), data_mapper)
```

## 7. Analyze the data

The analysis pipeline composes with `+` in the same way the experiment builder does, and each
stage advances the fit one level: {class}`~.ComputeObservables` turns raw bits into Pauli
observables, {class}`~.CurveFitObservables` fits an exponential decay to each path, and
{class}`~.NNLSSolve` inverts the model to recover non-negative generator rates.

```{code-cell} python
from qiskit_noise_learning.analysis import (
    ComputeObservables,
    CurveFitObservables,
    NNLSSolve,
)

analyzer = ComputeObservables() + CurveFitObservables() + NNLSSolve()

fit = analyzer.run(fit)
```

Plotting the measured observable means against their fitted exponentials, one subplot per `CZ`
pair, is the first thing to look at: it shows directly whether the decays the model is being
fitted to are exponential at all.

```{code-cell} python
fit.plot_qubit_pair_decays(
    pairs=layer_1_pairs,
    observable_type="means",
    exponential_fit=True,
)
```

The same plot against the *model's* prediction closes the loop, comparing the data to the decays
implied by the rates that came out of the non-negative least squares solve.

```{code-cell} python
fit.plot_qubit_pair_decays(
    pairs=layer_1_pairs,
    observable_type="means",
    model_prediction=True,
)
```

## 8. Grade the fit

The previous plot is the check to make quantitative. For every decay path and every fragment
depth, the experiment produced an averaged observable value, and the fitted model predicts one:
the path's intercept times its base raised to the depth. Agreement between those two sets of
numbers is what a learning experiment can actually be held to.

`predicted_path_decays`, from `qiskit_noise_learning.analysis.utils`, returns the
`(bases, intercepts)` the model implies for a set of unbound decay paths — the same quantity the
model-prediction curves above were drawn from.

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

The residuals are *smaller than the statistical error of the data they are residuals of* — a
median of about six tenths of a standard error, with almost every point inside two. The fitted
model reproduces the experiment to within its shot noise, and does so at every depth, not just
the shallow ones where the signal is largest. That is the substantive result.

## 9. Why not compare the rates directly?

Because the noise here was chosen rather than measured, it is tempting to grade the fit by
lining the recovered rates up against the injected ones. That comparison is not very meaningful,
and it is worth seeing why.

The natural first step is to move from rates to **Pauli fidelities** — for each Pauli, the factor
by which one application of the layer damps it — since fidelities are much closer to what the
experiment measures. The model maps generator rates to log-fidelities, so applying it to the true
rates and to the fitted rates gives two predictions for the same set of fidelities.

```{code-cell} python
from qiskit_noise_learning.sequences import FidelityIndex

layer_gate = gate_set.model_gate_set["layer_1"]
fidelity_indices = [
    FidelityIndex.from_gate(layer_gate, pauli) for pauli in model.generators["layer_1"]
]


def fidelities(rates):
    log_fidelities = model.projected_output(fidelity_indices, rates)
    return np.array([np.exp(-log_fidelities[index]) for index in fidelity_indices])


fitted_rates = dict(
    zip(
        fit.model_data.dataset["parameter"].data,
        fit.model_data.dataset["parameter_values"].data,
    )
)

true_fidelities = fidelities(dict(zip(generator_indices, true_rates)))
fitted_fidelities = fidelities(fitted_rates)

errors = np.abs(true_fidelities - fitted_fidelities)
print(f"number of fidelities:  {len(true_fidelities)}")
print(f"true fidelity range:   {true_fidelities.min():.4f} to {true_fidelities.max():.4f}")
print(f"correlation:           {np.corrcoef(true_fidelities, fitted_fidelities)[0, 1]:.3f}")
print(f"median absolute error: {np.median(errors):.5f}")
print(f"max absolute error:    {errors.max():.5f}")
```

Across all 144 Pauli fidelities of the layer this tracks reasonably well — a correlation around
0.92 and a median error of a few parts in a thousand on fidelities spanning roughly 0.94 to 0.98
— but it is not the near-exact agreement of the previous section, and it should not be.

The individual **rates** are further off still:

```{code-cell} python
fitted_rate_array = np.array(
    [fitted_rates.get(index, 0.0) for index in generator_indices]
)

print(f"rate correlation:      {np.corrcoef(true_rates, fitted_rate_array)[0, 1]:.3f}")
print(f"rates fitted to zero:  {int((fitted_rate_array == 0).sum())} of {num_parameters}")
print(f"mean true rate:        {true_rates.mean():.2e}")
print(f"mean fitted rate:      {fitted_rate_array.mean():.2e}")
```

The aggregate scale of the noise comes back correctly, but the per-generator correlation is
mediocre and more than a third of the rates are driven to exactly zero. None of this indicates a
bad fit — the previous section already established that this model reproduces the data to within
shot noise. Two separate things are going on.

The first is **rank deficiency**. This experiment does not measure enough independent decays to
pin down 168 rates, so many rate assignments explain the data equally well and the non-negative
least squares solve returns the sparsest of them. Which is why so many come back at exactly zero.

The second survives even a full-rank design: a **gauge space** of directions in rate space that
no experiment of this kind can see. Rates related by a gauge transformation produce identical
observable predictions, so the fit is free to return any representative. Comparing a recovered
rate to an injected one therefore compares two arbitrary choices of gauge, unless the noise map
was built with symmetries that fix one. The $X \otimes X$ versus $Y \otimes Y$ degeneracy
isolated in the [previous tutorial](noise_learner.md) is the smallest visible instance.

Pauli fidelities are better behaved than rates but not immune, which is why the fidelity
comparison above lands at 0.92 rather than at the 0.997 of the observable-level check: the
individual fidelities of a degenerate pair are no more separable than the rates are, only their
product over an even number of layers.

This sets the boundary of what an experiment of this shape supports. Predictions that depend on
the model only through the quantities the experiment measured — which includes probabilistic
error cancellation and most error-mitigation uses of a learned model — are on solid ground.
Reading physical meaning into an individual generator's rate is not. That takes an experiment
designed for it, with more layers or a richer set of paths, and a convention that fixes the
gauge.
