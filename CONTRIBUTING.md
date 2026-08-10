# Contributing

Thank you for your interest in contributing to `qiskit-noise-learning`!

## Contributions during the alpha phase

This library is in an alpha stage of development, where interfaces are unstable and breaking
changes are permitted between releases (see the [deprecation policy](DEPRECATION.md)).

During this phase, the project would benefit greatly from bug report issues, or PRs that fix bugs.
The core team is also very interested in feature request issues that help us better understand the
needs of the community. However, we have limited capacity to work on new features, and likewise
limited capacity to review large PRs that implement them. If you are considering a substantial
change, please open an issue to discuss it before investing significant effort.

## Installation

Developers should install in editable mode with the development and visualization requirements:

```bash
pip install -e ".[dev,vis]"
```

## Testing

Testing is done with `pytest`, and tests live in the `test` directory:

```bash
python -m pytest
```

The test suite also runs the doctests embedded in docstrings (`--doctest-modules` is configured in
`pyproject.toml`). We use [scipy-doctest](https://github.com/scipy/scipy_doctest) to relax some of
doctest's every-line-must-assert constraints. Please make sure any code examples you add to
docstrings actually run.

## Linting and formatting

[`ruff`](https://docs.astral.sh/ruff/) is used for both linting and formatting (line length 100,
target Python 3.11). Run the checks manually with:

```bash
ruff check --fix .
ruff format .
```

More conveniently, configure `ruff` to run on save in your editor.

## Pre-commit hooks

We recommend installing and using [`pre-commit`](https://pre-commit.com/), which automatically runs
the linters, formatters, and other checks on staged files during `git commit`.

1. Install the hooks defined in `.pre-commit-config.yaml`:

    ```bash
    pre-commit install
    ```

2. Optionally, run the hooks against all files:

    ```bash
    pre-commit run --all-files
    ```

3. To update the hooks to their latest versions:

    ```bash
    pre-commit autoupdate
    ```

## Copyright headers

Every source file (except those under `docs/`) must carry the IBM copyright header. This is checked
by `tools/verify_headers.py`, which also runs as a pre-commit hook. Copy the header from any
existing source file into new files you add.

## Documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/). To build it locally, install
the docs requirements and run:

```bash
pip install -e ".[docs]"
sphinx-build -b html -W docs docs/_build/html
```

The `-W` flag turns warnings into errors, matching CI. The rendered HTML is written to
`docs/_build/html`.

The tutorials in `docs/tutorials/` are [MyST Markdown](https://myst-parser.readthedocs.io/)
notebooks that [myst-nb](https://myst-nb.readthedocs.io/) executes for real at build time, so a
full build takes several minutes. myst-nb caches those executions in `docs/_build/.jupyter_cache`,
keyed on the notebook source only — a change to the library itself does not invalidate the cache.
If you change anything that affects tutorial *output*, remove `docs/_build` before rebuilding:

```bash
rm -rf docs/_build
```

## Changelog

We use [Towncrier](https://towncrier.readthedocs.io/) for changelog management. All PRs that make a
changelog-worthy change should add a changelog entry, which is just a file in the `changelog.d/`
directory named `<PR-number>.<type>.md`. The `<type>` is one of `added`, `changed`, `fixed`,
`deprecated`, `removed`, or `security`, and the file's contents are the changelog entry itself. For
example, for PR `#42` you might add `changelog.d/42.added.md` containing:

```markdown
Added a cool feature.
```

You can create this file however you like; `towncrier create -c "Added a cool feature." 42.added.md`
is a convenient shortcut for doing so.

## Releasing a version

Releases are cut by the core team. To assemble the changelog for a new version `X.Y.Z`, a maintainer
runs Towncrier to consume the pending fragments in `changelog.d/` and prepend a new section to
`CHANGELOG.md`:

```bash
towncrier build --version X.Y.Z
```

The change is committed on a release branch and merged into `main`, after which a GitHub release is
created for the tagged version.
