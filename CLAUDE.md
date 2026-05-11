# CLAUDE.md

## Project

`qiskit-noise-learning` is a Python toolkit for randomization-based quantum noise characterization, built on top of Qiskit.

- **Python**: >=3.11
- **License**: Apache 2.0

## Development Setup

```bash
pip install -e ".[dev,vis]"
pre-commit install
```

## Commands

| Task | Command |
|------|---------|
| Run tests | `python -m pytest` |
| Lint + format | `ruff check --fix . && ruff format .` |
| All pre-commit checks | `pre-commit run --all-files` |

Tests include doctests (`--doctest-modules` is set in `pyproject.toml`).

## Code Style

- **Formatter/linter**: `ruff` (line length 100, target Python 3.11)
- Rules: pycodestyle (E), pyflakes (F), isort (I), private member access (SLF), pyupgrade (UP)
- All files (except `docs/`) must have the IBM copyright header — see `tools/verify_headers.py`
- Constructor (`__init__`) parameters are documented in the **class docstring**, not in a separate `__init__` docstring.

## CI

- **`test_package.yaml`**: runs on push/PR to `main`; tests Python 3.11, 3.12, 3.13; pre-commit runs on 3.12
- **`nightly.yml`**: nightly tests against dev Qiskit on 3.11 and 3.13

## Project Structure

```
qiskit_noise_learning/   # Main package
  analysis/
  circuit_generator/
  data/
  experiment_builder/
  gate_sets/
  math/
  models/
  noise_learner/
  sequences/
  utils/
  visualizations/
qiskit_pandora/          # Secondary package
test/unit/               # Unit tests
docs/                    # Notebooks and demos
tools/                   # Dev scripts (verify_headers.py)
```
