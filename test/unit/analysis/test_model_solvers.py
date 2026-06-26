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
from qiskit.quantum_info import QubitSparsePauli, QubitSparsePauliList

from qiskit_noise_learning.analysis import Fit, LSQLinearSolve, NNLSSolve
from qiskit_noise_learning.data import AveragedData
from qiskit_noise_learning.models._legacy import GeneratorIndex, PauliLindbladModel

_SOLVERS = [LSQLinearSolve(), NNLSSolve()]

# Each CZ path's row is built from real Pauli-Lindblad commutation, so every coefficient is 2.0 per
# anticommuting generator. With the generator sets chosen below the CZ fidelity "XI" anticommutes
# only "ZI", "IX" anticommutes only "IZ", and "XX" anticommutes both. Prep and measurement
# generators are needed for model construction but only appear in rows of paths with SPAM fragments.
_PM_GENS = {"P": QubitSparsePauliList(["XI"]), "M": QubitSparsePauliList(["XI"])}


def _get_rate_from_fit(fit, gate_name, label):
    """Read a fitted rate by its generator label."""
    gen = GeneratorIndex(gate_name, QubitSparsePauli(label))
    return fit.model_data.dataset["parameter_values"].sel(parameter=gen).item()


@pytest.mark.parametrize("solver", _SOLVERS)
def test_single_unbound_path(solver, gate_set_cz, make_cz_path, make_averaged_data):
    """Solving a single unbound path with a 1x1 design matrix."""
    f_true = 0.8
    model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI"]), **_PM_GENS})
    path = make_cz_path("XI")  # row = {GeneratorIndex("CZ", "ZI"): 4.0}

    fit = Fit(model=model)
    fit[AveragedData] = make_averaged_data([(path, -1, f_true)])
    result = solver.run(fit)

    # A = [[4]], b = -log(f), so the rate is -log(f) / 4.
    assert np.isclose(_get_rate_from_fit(result, "CZ", "ZI"), -np.log(f_true) / 4, atol=1e-6)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_multiple_unbound_paths(solver, gate_set_cz, make_cz_path, make_averaged_data):
    """Solving two unbound paths mapped to different generators (diagonal design matrix)."""
    f0, f1 = 0.9, 0.7
    model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI", "IZ"]), **_PM_GENS})
    path0 = make_cz_path("XI")  # row = {CZ:ZI: 4.0}
    path1 = make_cz_path("IX")  # row = {CZ:IZ: 4.0}

    fit = Fit(model=model)
    fit[AveragedData] = make_averaged_data([(path0, -1, f0), (path1, -1, f1)])
    result = solver.run(fit)

    assert np.isclose(_get_rate_from_fit(result, "CZ", "ZI"), -np.log(f0) / 4, atol=1e-6)
    assert np.isclose(_get_rate_from_fit(result, "CZ", "IZ"), -np.log(f1) / 4, atol=1e-6)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_underdetermined(solver, gate_set_cz, make_cz_path, make_averaged_data):
    """Solving one decay onto two generators — solver picks a non-negative split."""
    f_true = 0.8
    model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI", "IZ"]), **_PM_GENS})
    path = make_cz_path("XX")  # row = {CZ:ZI: 4.0, CZ:IZ: 4.0}

    fit = Fit(model=model)
    fit[AveragedData] = make_averaged_data([(path, -1, f_true)])
    result = solver.run(fit)

    r0 = _get_rate_from_fit(result, "CZ", "ZI")
    r1 = _get_rate_from_fit(result, "CZ", "IZ")
    assert r0 >= 0
    assert r1 >= 0
    # 4 * (r0 + r1) = -log(f)
    assert np.isclose(r0 + r1, -np.log(f_true) / 4, atol=1e-6)
    assert result.model_data.dataset["covariance"].values.shape == (2, 2)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_overdetermined(solver, gate_set_cz, make_cz_path, make_averaged_data):
    """Solving three decays onto two generators with consistent data."""
    f0, f1 = 0.9, 0.8
    f2 = f0 * f1
    model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI", "IZ"]), **_PM_GENS})
    path0 = make_cz_path("XI")  # row = {CZ:ZI: 4.0}
    path1 = make_cz_path("IX")  # row = {CZ:IZ: 4.0}
    path2 = make_cz_path("XX")  # row = {CZ:ZI: 4.0, CZ:IZ: 4.0}

    fit = Fit(model=model)
    fit[AveragedData] = make_averaged_data([(path0, -1, f0), (path1, -1, f1), (path2, -1, f2)])
    result = solver.run(fit)

    assert np.isclose(_get_rate_from_fit(result, "CZ", "ZI"), -np.log(f0) / 4, atol=1e-6)
    assert np.isclose(_get_rate_from_fit(result, "CZ", "IZ"), -np.log(f1) / 4, atol=1e-6)
    assert result.model_data.dataset["covariance"].values.shape == (2, 2)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_covariance_identity_design(solver, gate_set_cz, make_cz_path, make_averaged_data):
    """Covariance computation with a 1x1 design matrix."""
    f_true = 0.8
    f_std = 0.02
    model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI"]), **_PM_GENS})
    path = make_cz_path("XI")

    fit = Fit(model=model)
    fit[AveragedData] = make_averaged_data([(path, -1, f_true, f_std)])
    result = solver.run(fit)

    # sigma_b = f_std / f, A = [[4]] so var = (sigma_b / 4) ** 2.
    expected_var = (f_std / f_true / 4) ** 2
    cov = result.model_data.dataset["covariance"].values
    assert cov.shape == (1, 1)
    assert np.isclose(cov[0, 0], expected_var, rtol=1e-6)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_covariance_constrained_params(solver, gate_set_cz, make_cz_path, make_averaged_data):
    """Covariance is zero for a parameter constrained to zero by the solver."""
    model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI", "IZ"]), **_PM_GENS})
    path0 = make_cz_path("XI")  # row = {CZ:ZI: 4.0}
    path1 = make_cz_path("XX")  # row = {CZ:ZI: 4.0, CZ:IZ: 4.0}

    # path0 (f=0.8) has more decay than path1 (f=0.9), so the unconstrained CZ:IZ rate is negative
    # and the solver pins it at the zero boundary.
    fit = Fit(model=model)
    fit[AveragedData] = make_averaged_data([(path0, -1, 0.8), (path1, -1, 0.9)])
    result = solver.run(fit)

    assert _get_rate_from_fit(result, "CZ", "IZ") >= 0

    cov = result.model_data.dataset["covariance"]
    iz = GeneratorIndex("CZ", QubitSparsePauli("IZ"))
    assert np.allclose(cov.sel(parameter_row=iz).values, 0.0)
    assert np.allclose(cov.sel(parameter_col=iz).values, 0.0)


def test_metadata_contains_residual(gate_set_cz, make_cz_path, make_averaged_data):
    """Metadata contains the residual norm."""
    model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI"]), **_PM_GENS})
    path = make_cz_path("XI")

    fit = Fit(model=model)
    fit[AveragedData] = make_averaged_data([(path, -1, 0.8)])
    result = NNLSSolve().run(fit)

    assert "residual" in result.model_data.metadata


@pytest.mark.parametrize("solver", _SOLVERS)
def test_bound_paths_in_fit_paths(solver, gate_set_cz, make_cz_path, make_averaged_data):
    """Solving with a bound path specified in fit.paths (repeatable scaled by depth + SPAM)."""
    f_true = 0.7
    model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI"]), **_PM_GENS})
    unbound = make_cz_path("XI")
    bound = unbound.bind_at(3)
    # row = {CZ:ZI: 12.0, P:XI: 2.0, M:XI: 2.0}

    fit = Fit(model=model, paths=[bound])
    fit[AveragedData] = make_averaged_data([(unbound, 3, f_true)])
    result = solver.run(fit)

    r_cz = _get_rate_from_fit(result, "CZ", "ZI")
    r_p = _get_rate_from_fit(result, "P", "XI")
    r_m = _get_rate_from_fit(result, "M", "XI")
    assert r_cz >= 0
    assert r_p >= 0
    assert r_m >= 0
    assert np.isclose(12 * r_cz + 2 * r_p + 2 * r_m, -np.log(f_true), atol=1e-6)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_mixed_bound_and_unbound_paths(solver, gate_set_cz, make_cz_path, make_averaged_data):
    """Solving with a mix of bound and unbound paths in fit.paths."""
    f0, f1 = 0.9, 0.8
    model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI", "IZ"]), **_PM_GENS})
    unbound0 = make_cz_path("IX")  # row = {CZ:IZ: 4.0}
    unbound1 = make_cz_path("XI")
    bound1 = unbound1.bind_at(2)  # row = {CZ:ZI: 8.0, P:XI: 2.0, M:XI: 2.0}

    fit = Fit(model=model, paths=[unbound0, bound1])
    fit[AveragedData] = make_averaged_data([(unbound0, -1, f0), (unbound1, 2, f1)])
    result = solver.run(fit)

    assert np.isclose(_get_rate_from_fit(result, "CZ", "IZ"), -np.log(f0) / 4, atol=1e-6)

    r_cz = _get_rate_from_fit(result, "CZ", "ZI")
    r_p = _get_rate_from_fit(result, "P", "XI")
    r_m = _get_rate_from_fit(result, "M", "XI")
    assert r_cz >= 0
    assert r_p >= 0
    assert r_m >= 0
    assert np.isclose(8 * r_cz + 2 * r_p + 2 * r_m, -np.log(f1), atol=1e-6)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_no_paths_uses_all_data(solver, gate_set_cz, make_cz_path, make_averaged_data):
    """When fit.paths is not specified, all data is used."""
    model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI"]), **_PM_GENS})
    path = make_cz_path("XI")

    # Same path present both unbound (decay; repeatable fragment only) and bound at depth 2 (which
    # also picks up the SPAM generators).
    fit = Fit(model=model)
    fit[AveragedData] = make_averaged_data([(path, -1, 0.85), (path, 2, 0.7)])
    result = solver.run(fit)

    params = list(result.model_data.dataset["parameter"].values)
    assert GeneratorIndex("CZ", QubitSparsePauli("ZI")) in params
    assert GeneratorIndex("P", QubitSparsePauli("XI")) in params
    assert GeneratorIndex("M", QubitSparsePauli("XI")) in params
