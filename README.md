# qiskit-noise-learning

A Python toolkit for randomization-based quantum noise characterization.

> **Alpha software.** This library is unversioned and under active development. Interfaces are unstable and may change without notice between commits. All users not directly involved in active development and testing should be cautious, though all feedback is appreciated.

## Installation

```bash
pip install .
```

Include development dependencies with `[dev]` and visualization dependencies with `[vis]`:

```bash
pip install -e ".[dev,vis]"
pre-commit install
```


## Interfaces and Documentation

This library has two levels of interface.
The first is the low-level interface where the user directly interacts with objects representing core concepts in noise learning, enabling custom design of every aspect of a noise learning protocol.
The second is a higher-level interface that wraps a stock workflow into an easy-to-use `NoiseLearner` object.
You can see both demonstrated in the following tutorials, which run locally against a fake
backend and need no IBM Quantum credentials:

- [`docs/tutorials/noise_learner.md`](docs/tutorials/noise_learner.md) — end-to-end use of `NoiseLearner`
- [`docs/tutorials/workflow.md`](docs/tutorials/workflow.md) — step-by-step walkthrough of the internal pipeline


## Development

```bash
python -m pytest          # run tests (includes doctests)
ruff check --fix . && ruff format .   # lint and format
pre-commit run --all-files            # all checks
```

## License

[Apache 2.0](LICENSE.txt)

## Citing this package

If you use this package in your research, use the [CITATION.bib](CITATION.bib) file in this project’s repository to cite the appropriate reference(s).
