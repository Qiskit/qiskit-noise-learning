# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import pytest

from qiskit_noise_learning.analysis import Fit, LSQLinearSolve, NNLSSolve
from qiskit_noise_learning.data import AveragedData
from qiskit_noise_learning.math import IndexedVector
from qiskit_noise_learning.sequences import Path

_SOLVERS = [LSQLinearSolve(), NNLSSolve()]

_SOLVERS = [LSQLinearSolve(), NNLSSolve()]


class MockFidelityModel:
    def __init__(self, rows: dict, path_rows: dict | None = None):
        self._rows = rows
        self._path_rows = path_rows or {}

    def multiplicative_row_from_path_pattern(self, path_pattern):
        return self._rows[path_pattern]

    def row_from_path(self, path):
        return self._path_rows[path]


def _make_decay_data(f_values, f_std_values=None):
    """Build an AveragedData with depth=-1 entries from dictionaries of decay fidelity values."""
    if f_std_values is None:
        f_std_values = {pp: 0.001 for pp in f_values}

    keys = list(f_values.keys())
    n = len(keys)

    return AveragedData.from_arrays(
        path_patterns=keys,
        depths=[-1] * n,
        observables=np.array([f_values[pp] for pp in keys]),
        std=np.array([f_std_values[pp] for pp in keys]),
        time_lbs=np.empty(n, dtype="datetime64[us]"),
        time_ubs=np.empty(n, dtype="datetime64[us]"),
    )


@pytest.mark.parametrize("solver", _SOLVERS)
def test_single_pattern(solver):
    """Test solving a single pattern with a 1x1 design matrix."""
    f_true = 0.8
    pp = "pp0"

    decay_data = _make_decay_data({pp: f_true})
    model = MockFidelityModel({pp: IndexedVector({"theta": 1.0})})

    fit = Fit(model=model)
    fit[AveragedData] = decay_data
    result = solver.run(fit)

    model_data = result.model_data
    assert np.isclose(
        model_data.dataset["parameter_values"].sel(parameter="theta").item(),
        -np.log(f_true),
        atol=1e-6,
    )


@pytest.mark.parametrize("solver", _SOLVERS)
def test_multiple_patterns(solver):
    """Test solving two path patterns mapped to different parameters."""
    pp0, pp1 = "pp0", "pp1"
    f0, f1 = 0.9, 0.7

    decay_data = _make_decay_data({pp0: f0, pp1: f1})
    model = MockFidelityModel(
        {
            pp0: IndexedVector({"r0": 2.0}),
            pp1: IndexedVector({"r1": 1.0}),
        }
    )

    fit = Fit(model=model)
    fit[AveragedData] = decay_data
    result = solver.run(fit)

    model_data = result.model_data
    assert np.isclose(
        model_data.dataset["parameter_values"].sel(parameter="r0").item(),
        -np.log(f0) / 2,
        atol=1e-6,
    )
    assert np.isclose(
        model_data.dataset["parameter_values"].sel(parameter="r1").item(),
        -np.log(f1),
        atol=1e-6,
    )


@pytest.mark.parametrize("solver", _SOLVERS)
def test_underdetermined(solver):
    """Test solving one decay to two parameters — solver picks a non-negative split."""
    f_true = 0.8
    pp = "pp0"

    decay_data = _make_decay_data({pp: f_true})
    model = MockFidelityModel({pp: IndexedVector({"r0": 1.0, "r1": 1.0})})

    fit = Fit(model=model)
    fit[AveragedData] = decay_data
    result = solver.run(fit)

    model_data = result.model_data
    r0 = model_data.dataset["parameter_values"].sel(parameter="r0").item()
    r1 = model_data.dataset["parameter_values"].sel(parameter="r1").item()
    assert r0 >= 0
    assert r1 >= 0
    assert np.isclose(r0 + r1, -np.log(f_true), atol=1e-6)
    assert model_data.dataset["covariance"].values.shape == (2, 2)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_overdetermined(solver):
    """Test solving three decays to two parameters."""
    pp0, pp1, pp2 = "pp0", "pp1", "pp2"
    f0, f1 = 0.9, 0.8
    f2 = f0 * f1

    decay_data = _make_decay_data({pp0: f0, pp1: f1, pp2: f2})
    model = MockFidelityModel(
        {
            pp0: IndexedVector({"r0": 1.0}),
            pp1: IndexedVector({"r1": 1.0}),
            pp2: IndexedVector({"r0": 1.0, "r1": 1.0}),
        }
    )

    fit = Fit(model=model)
    fit[AveragedData] = decay_data
    result = solver.run(fit)

    model_data = result.model_data
    assert np.isclose(
        model_data.dataset["parameter_values"].sel(parameter="r0").item(),
        -np.log(f0),
        atol=1e-6,
    )
    assert np.isclose(
        model_data.dataset["parameter_values"].sel(parameter="r1").item(),
        -np.log(f1),
        atol=1e-6,
    )
    assert model_data.dataset["covariance"].values.shape == (2, 2)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_covariance_identity_design(solver):
    """Test covariance computation with a 1x1 identity design matrix."""
    f_true = 0.8
    f_std = 0.02
    pp = "pp0"

    decay_data = _make_decay_data({pp: f_true}, f_std_values={pp: f_std})
    model = MockFidelityModel({pp: IndexedVector({"theta": 1.0})})

    fit = Fit(model=model)
    fit[AveragedData] = decay_data
    result = solver.run(fit)

    expected_var = (f_std / f_true) ** 2
    cov_data = result.model_data.dataset["covariance"].values
    assert cov_data.shape == (1, 1)
    assert np.isclose(cov_data[0, 0], expected_var, rtol=1e-6)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_covariance_constrained_params(solver):
    """Test that covariance is zero for parameters constrained to zero by the solver."""
    pp0, pp1 = "pp0", "pp1"

    decay_data = _make_decay_data({pp0: 0.8, pp1: 0.999})
    model = MockFidelityModel(
        {
            pp0: IndexedVector({"r0": 1.0}),
            pp1: IndexedVector({"r0": 0.5, "r1": 1.0}),
        }
    )

    fit = Fit(model=model)
    fit[AveragedData] = decay_data
    result = solver.run(fit)

    model_data = result.model_data
    r1_val = model_data.dataset["parameter_values"].sel(parameter="r1").item()
    assert r1_val >= 0

    cov = model_data.dataset["covariance"]
    assert np.allclose(cov.sel(parameter_row="r1").values, 0.0)
    assert np.allclose(cov.sel(parameter_col="r1").values, 0.0)


@pytest.mark.parametrize("solver", [NNLSSolve()])
def test_metadata_contains_residual(solver):
    """Test that metadata contains the residual."""
    pp = "pp0"

    decay_data = _make_decay_data({pp: 0.8})
    model = MockFidelityModel({pp: IndexedVector({"theta": 1.0})})

    fit = Fit(model=model)
    fit[AveragedData] = decay_data
    result = solver.run(fit)

    assert "residual" in result.model_data.metadata


def _make_fixed_depth_data(entries, std_values=None):
    """Build AveragedData with fixed-depth entries.

    Args:
        entries: List of (path_pattern, depth, fidelity) tuples.
        std_values: Optional list of std values. Defaults to 0.001 for all.
    """
    n = len(entries)
    if std_values is None:
        std_values = [0.001] * n

    return AveragedData.from_arrays(
        path_patterns=[e[0] for e in entries],
        depths=[e[1] for e in entries],
        observables=np.array([e[2] for e in entries]),
        std=np.array(std_values),
        time_lbs=np.empty(n, dtype="datetime64[us]"),
        time_ubs=np.empty(n, dtype="datetime64[us]"),
    )


@pytest.mark.parametrize("solver", _SOLVERS)
def test_fixed_depth_single(solver):
    """Test solving a single fixed-depth entry using the additive row."""
    pp = "pp0"
    depth = 3
    fidelity = 0.7
    path = Path(pattern=pp, depth=depth)

    data = _make_fixed_depth_data([(pp, depth, fidelity)])
    model = MockFidelityModel(
        rows={},
        path_rows={path: IndexedVector({"theta": 2.0})},
    )

    fit = Fit(model=model)
    fit[AveragedData] = data
    result = solver.run(fit)

    model_data = result.model_data
    assert np.isclose(
        model_data.dataset["parameter_values"].sel(parameter="theta").item(),
        -np.log(fidelity) / 2,
        atol=1e-6,
    )


@pytest.mark.parametrize("solver", _SOLVERS)
def test_mixed_decay_and_fixed_depth(solver):
    """Test solving with both decay (depth=-1) and fixed-depth entries."""
    pp0, pp1 = "pp0", "pp1"
    decay_fidelity = 0.9
    depth = 2
    fixed_depth_fidelity = 0.75
    path = Path(pattern=pp1, depth=depth)

    decay_data = _make_decay_data({pp0: decay_fidelity})
    fixed_data = _make_fixed_depth_data([(pp1, depth, fixed_depth_fidelity)])
    data = decay_data.merge(fixed_data)

    model = MockFidelityModel(
        rows={pp0: IndexedVector({"r0": 1.0})},
        path_rows={path: IndexedVector({"r1": 1.0})},
    )

    fit = Fit(model=model)
    fit[AveragedData] = data
    result = solver.run(fit)

    model_data = result.model_data
    assert np.isclose(
        model_data.dataset["parameter_values"].sel(parameter="r0").item(),
        -np.log(decay_fidelity),
        atol=1e-6,
    )
    assert np.isclose(
        model_data.dataset["parameter_values"].sel(parameter="r1").item(),
        -np.log(fixed_depth_fidelity),
        atol=1e-6,
    )
