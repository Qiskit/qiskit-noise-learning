# Tutorials

Two end-to-end walkthroughs of learning the Pauli-Lindblad noise of a layer of two-qubit
gates.

[Learning a layer with NoiseLearner](noise_learner.md) uses the high-level API: annotate a
boxed layer, hand it to {class}`~.NoiseLearner`, and read the noise maps off the result. Start
here.

[Building a learning experiment by hand](workflow.md) unpacks the same calculation into the
stages {class}`~.NoiseLearner` runs internally — gate set, fidelity model, experiment builder,
circuit generation, and the analysis pipeline — so that each one can be inspected and swapped
out.

Neither tutorial needs hardware or IBM Quantum credentials. Both target the ``FakeMarrakesh``
backend and execute their circuits locally with {class}`~.AerExecutor`, injecting a
Pauli-Lindblad noise model that the tutorial writes itself. Every number below is therefore
synthetic — which is exactly what makes it possible to check the learned answer against a
known truth.

```{toctree}
:maxdepth: 1
:hidden:

noise_learner
workflow
```
