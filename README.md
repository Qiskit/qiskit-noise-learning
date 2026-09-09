# qiskit-noise-learning

A Python toolkit for randomization-based quantum noise characterization.

> [!NOTE]
> **Alpha software.** This library is in the `0.x` stage of development and under active
> development. No part of the public interface is yet stable: while the major version is `0`,
> expect breaking changes between releases and pin your dependency accordingly (for example,
> `qiskit-noise-learning==0.1.*`). We do not currently issue deprecation warnings, but all changes
> are recorded in the changelog. See the [deprecation policy](DEPRECATION.md) for details. All
> feedback is appreciated.

## Installation

You can install `qiskit-noise-learning` via pip from the GitHub repository (`main` branch):

```bash
pip install "git+https://github.com/Qiskit/qiskit-noise-learning@main"

```

For visualization support, include the visualization dependencies:

```bash
pip install "qiskit-noise-learning[vis] @ git+https://github.com/Qiskit/qiskit-noise-learning@main"
```

See the [contribution guidelines](CONTRIBUTING.md) for developer dependencies and editable
installations.

## Interfaces and Documentation

This library has two levels of interface. The first is the low-level interface where the user
directly interacts with objects representing core concepts in noise learning, enabling custom design
of every aspect of a noise learning protocol. The second is a higher-level interface that wraps a
stock workflow into an easy-to-use `NoiseLearner` object. You can see both demonstrated in the
following tutorials, which run locally against a fake backend and need no IBM Quantum credentials:

- [`docs/guides/noise_learner.md`](docs/tutorials/noise_learner.md) — end-to-end use of `NoiseLearner`
- [`docs/guides/workflow.md`](docs/tutorials/workflow.md) — step-by-step walkthrough of the internal pipeline


## Development

```bash
python -m pytest          # run tests (includes doctests)
ruff check --fix . && ruff format .   # lint and format
pre-commit run --all-files            # all checks
```

See the [contribution guidelines](CONTRIBUTING.md) for details on developer setup, testing,
building the documentation, and the changelog workflow.

## License

[Apache 2.0](LICENSE.txt)

## Citing this package

If you use this package in your research, use the [CITATION.bib](CITATION.bib) file in this project’s repository to cite the appropriate reference(s).
