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

_SOLVERS = [LSQLinearSolve(), NNLSSolve()]


class MockPath:
    """A minimal mock path that supports bind_at, without_depth, and is_unbound."""

    def __init__(self, name, depth=None):
        self._name = name
        self._depth = depth

    @property
    def depth(self):
        return self._depth

    @property
    def is_unbound(self):
        return self._depth is None

    def bind_at(self, depth):
        return MockPath(self._name, depth=depth)

    def without_depth(self):
        return MockPath(self._name, depth=None)

    def __eq__(self, other):
        return (
            isinstance(other, MockPath)
            and self._name == other._name
            and self._depth == other._depth
        )

    def __hash__(self):
        return hash((self._name, self._depth))

    def __repr__(self):
        return f"MockPath({self._name!r}, depth={self._depth})"


class MockFidelityModel:
    def __init__(self, rows: dict):
        self._rows = rows

    def row_from_path(self, path):
        return self._rows[path]


def _make_decay_data(f_values, f_std_values=None):
    """Build an AveragedData with depth=-1 entries from dictionaries of decay fidelity values."""
    if f_std_values is None:
        f_std_values = {pp: 0.001 for pp in f_values}

    keys = list(f_values.keys())
    n = len(keys)

    return AveragedData.from_arrays(
        unbound_paths=keys,
        depths=[-1] * n,
        observables=np.array([f_values[pp] for pp in keys]),
        std=np.array([f_std_values[pp] for pp in keys]),
        time_lbs=np.empty(n, dtype="datetime64[us]"),
        time_ubs=np.empty(n, dtype="datetime64[us]"),
    )


def _make_averaged_data(entries, std_default=0.001):
    """Build AveragedData from a list of (unbound_path, depth, fidelity) tuples."""
    unbound_paths = [e[0] for e in entries]
    depths = [e[1] for e in entries]
    observables = np.array([e[2] for e in entries])
    std = np.array([e[3] if len(e) > 3 else std_default for e in entries])
    n = len(entries)
    return AveragedData.from_arrays(
        unbound_paths=unbound_paths,
        depths=depths,
        observables=observables,
        std=std,
        time_lbs=np.empty(n, dtype="datetime64[us]"),
        time_ubs=np.empty(n, dtype="datetime64[us]"),
    )


@pytest.mark.parametrize("solver", _SOLVERS)
def test_single_unbound_path(solver):
    """Test solving a single unbound_path with a 1x1 design matrix."""
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
def test_multiple_unbound_paths(solver):
    """Test solving two unbound paths mapped to different parameters."""
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


def test_metadata_contains_residual():
    """Test that metadata contains the NNLS residual."""
    pp = "pp0"

    decay_data = _make_decay_data({pp: 0.8})
    model = MockFidelityModel({pp: IndexedVector({"theta": 1.0})})

    fit = Fit(model=model)
    fit[AveragedData] = decay_data
    result = NNLSSolve().run(fit)

    assert "residual" in result.model_data.metadata


@pytest.mark.parametrize("solver", _SOLVERS)
def test_bound_paths_in_fit_paths(solver):
    """Test solving with bound paths specified in fit.paths."""
    pp = MockPath("pp0")
    pp_bound = pp.bind_at(3)
    f_true = 0.7

    averaged_data = _make_averaged_data([(pp, 3, f_true)])
    model = MockFidelityModel({pp_bound: IndexedVector({"r0": 3.0, "r_se": 1.0})})

    fit = Fit(model=model, paths=[pp_bound])
    fit[AveragedData] = averaged_data
    result = solver.run(fit)

    model_data = result.model_data
    r0 = model_data.dataset["parameter_values"].sel(parameter="r0").item()
    r_se = model_data.dataset["parameter_values"].sel(parameter="r_se").item()
    assert r0 >= 0
    assert r_se >= 0
    assert np.isclose(3 * r0 + r_se, -np.log(f_true), atol=1e-6)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_mixed_bound_and_unbound_paths(solver):
    """Test solving with a mix of bound and unbound paths in fit.paths."""
    pp0 = MockPath("pp0")
    pp1 = MockPath("pp1")
    pp1_bound = pp1.bind_at(2)

    f0 = 0.9
    f1 = 0.8

    averaged_data = _make_averaged_data([(pp0, -1, f0), (pp1, 2, f1)])
    model = MockFidelityModel(
        {
            pp0: IndexedVector({"r0": 1.0}),
            pp1_bound: IndexedVector({"r1": 2.0, "r_se": 1.0}),
        }
    )

    fit = Fit(model=model, paths=[pp0, pp1_bound])
    fit[AveragedData] = averaged_data
    result = solver.run(fit)

    model_data = result.model_data
    r0 = model_data.dataset["parameter_values"].sel(parameter="r0").item()
    assert np.isclose(r0, -np.log(f0), atol=1e-6)

    r1 = model_data.dataset["parameter_values"].sel(parameter="r1").item()
    r_se = model_data.dataset["parameter_values"].sel(parameter="r_se").item()
    assert r1 >= 0
    assert r_se >= 0
    assert np.isclose(2 * r1 + r_se, -np.log(f1), atol=1e-6)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_no_paths_uses_all_data(solver):
    """Test that when fit.paths is not specified, all data is used."""
    pp0 = MockPath("pp0")
    pp0_bound = pp0.bind_at(2)

    f_decay = 0.85
    f_bound = 0.7

    averaged_data = _make_averaged_data([(pp0, -1, f_decay), (pp0, 2, f_bound)])
    model = MockFidelityModel(
        {
            pp0: IndexedVector({"r0": 1.0}),
            pp0_bound: IndexedVector({"r0": 2.0, "r_se": 1.0}),
        }
    )

    fit = Fit(model=model)
    fit[AveragedData] = averaged_data
    result = solver.run(fit)

    model_data = result.model_data
    assert "r0" in model_data.dataset["parameter"].values
    assert "r_se" in model_data.dataset["parameter"].values
