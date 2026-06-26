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

from qiskit_noise_learning.analysis import Fit, PositivityMinSolve
from qiskit_noise_learning.data import AveragedData
from qiskit_noise_learning.math import IndexedMatrix, IndexedVector
from qiskit_noise_learning.models._legacy import (
    CompleteFidelityModel,
    GeneratorIndex,
    PauliLindbladModel,
)

cvxpy = pytest.importorskip("cvxpy")

# Prep and measurement generators required to build the model; they only appear in rows of paths
# that carry SPAM fragments (i.e. bound paths).
_PM_GENS = {"P": QubitSparsePauliList(["XI"]), "M": QubitSparsePauliList(["XI"])}


def _get_rate_from_fit(fit, gate_name, label):
    """Read a fitted rate by its generator label."""
    gen = GeneratorIndex(gate_name, QubitSparsePauli(label))
    return fit.model_data.dataset["parameter_values"].sel(parameter=gen).item()


class TestPositivityMinSolve:
    """Tests for PositivityMinSolve."""

    def test_type_check_rejects_non_pauli_lindblad(
        self, gate_set_cz, make_cz_path, make_averaged_data
    ):
        """Raises TypeError if model is not a PauliLindbladModel."""
        # CompleteFidelityModel is a real FidelityModel that is not a PauliLindbladModel.
        model = CompleteFidelityModel(gate_set_cz)
        path = make_cz_path("XI")
        solver = PositivityMinSolve(coefficients={"CZ": 1.0}, epsilon=1.0, deltas={path: 1.0})

        fit = Fit(model=model)
        fit[AveragedData] = make_averaged_data([(path, -1, 0.8)])

        with pytest.raises(TypeError, match="PauliLindbladModel"):
            solver.run(fit)

    def test_loose_constraints_minimize_positivity(
        self, gate_set_cz, make_cz_path, make_averaged_data
    ):
        """With loose constraints, the optimizer makes the parameter non-positive."""
        model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI"]), **_PM_GENS})
        path = make_cz_path("XI")  # row = {CZ:ZI: 4.0}
        solver = PositivityMinSolve(
            coefficients={"CZ": 1.0},
            epsilon=10.0,
            deltas={path: 10.0},
        )

        fit = Fit(model=model)
        fit[AveragedData] = make_averaged_data([(path, -1, 0.8)])
        result = solver.run(fit)

        # The objective max(0, x) is minimized by pushing x <= 0.
        assert _get_rate_from_fit(result, "CZ", "ZI") <= 1e-6

    def test_tight_constraints_recover_solution(
        self, gate_set_cz, make_cz_path, make_averaged_data
    ):
        """With tight epsilon and delta, the solution is forced near the LS solution."""
        f_true = 0.8
        model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI"]), **_PM_GENS})
        path = make_cz_path("XI")  # row = {CZ:ZI: 4.0}
        solver = PositivityMinSolve(
            coefficients={"CZ": 1.0},
            epsilon=1e-6,
            deltas={path: 1e-6},
        )

        fit = Fit(model=model)
        fit[AveragedData] = make_averaged_data([(path, -1, f_true)])
        result = solver.run(fit)

        # A = [[4]], b = -log(f), so the rate is -log(f) / 4.
        assert np.isclose(_get_rate_from_fit(result, "CZ", "ZI"), -np.log(f_true) / 4, atol=1e-4)

    def test_multiple_parameters_same_gate(self, gate_set_cz, make_cz_path, make_averaged_data):
        """Test with multiple parameters belonging to the same gate."""
        f0, f1 = 0.9, 0.85
        model = PauliLindbladModel(
            gate_set_cz, {"CZ": QubitSparsePauliList(["ZI", "IZ"]), **_PM_GENS}
        )
        path0 = make_cz_path("XI")  # row = {CZ:ZI: 4.0}
        path1 = make_cz_path("IX")  # row = {CZ:IZ: 4.0}
        solver = PositivityMinSolve(
            coefficients={"CZ": 1.0},
            epsilon=1e-6,
            deltas={path0: 1e-6, path1: 1e-6},
        )

        fit = Fit(model=model)
        fit[AveragedData] = make_averaged_data([(path0, -1, f0), (path1, -1, f1)])
        result = solver.run(fit)

        assert np.isclose(_get_rate_from_fit(result, "CZ", "ZI"), -np.log(f0) / 4, atol=1e-4)
        assert np.isclose(_get_rate_from_fit(result, "CZ", "IZ"), -np.log(f1) / 4, atol=1e-4)

    def test_multiple_gates_different_coefficients(
        self, gate_set_cz, make_cz_path, make_averaged_data
    ):
        """Parameters from different gates get different objective coefficients."""
        f0, f1 = 0.9, 0.9
        model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI"]), **_PM_GENS})
        unbound = make_cz_path("XI")
        bound = unbound.bind_at(1)
        # unbound row = {CZ:ZI: 4.0}; bound row = {CZ:ZI: 4.0, P:XI: 2.0, M:XI: 2.0}.
        solver = PositivityMinSolve(
            coefficients={"CZ": 10.0, "P": 1.0, "M": 1.0},
            epsilon=1e-6,
            deltas={unbound: 1e-6, bound: 1e-6},
        )

        fit = Fit(model=model)
        fit[AveragedData] = make_averaged_data([(unbound, -1, f1), (unbound, 1, f0)])
        result = solver.run(fit)

        # The unbound path pins CZ:ZI = -log(f1) / 4; with f0 == f1 the bound path then forces the
        # prep/measurement rates to zero.
        assert np.isclose(_get_rate_from_fit(result, "CZ", "ZI"), -np.log(f1) / 4, atol=1e-4)
        assert np.isclose(_get_rate_from_fit(result, "P", "XI"), 0.0, atol=1e-4)
        assert np.isclose(_get_rate_from_fit(result, "M", "XI"), 0.0, atol=1e-4)

    def test_non_negative_constraint(self, gate_set_cz, make_cz_path, make_averaged_data):
        """With non_negative=True, all parameters are >= 0 even under loose constraints."""
        model = PauliLindbladModel(
            gate_set_cz, {"CZ": QubitSparsePauliList(["ZI", "IZ"]), **_PM_GENS}
        )
        path0 = make_cz_path("XI")  # row = {CZ:ZI: 4.0}
        path1 = make_cz_path("IX")  # row = {CZ:IZ: 4.0}
        # Loose constraints would let the positivity objective drive the rates negative; the
        # non_negative flag pins them at the zero boundary instead.
        solver = PositivityMinSolve(
            coefficients={"CZ": 1.0},
            epsilon=10.0,
            deltas={path0: 10.0, path1: 10.0},
            non_negative=True,
        )

        fit = Fit(model=model)
        fit[AveragedData] = make_averaged_data([(path0, -1, 0.9), (path1, -1, 0.8)])
        result = solver.run(fit)

        assert _get_rate_from_fit(result, "CZ", "ZI") >= -1e-8
        assert _get_rate_from_fit(result, "CZ", "IZ") >= -1e-8

    def test_weight_matrix(self, gate_set_cz, make_cz_path, make_averaged_data):
        """Test that a weight matrix is applied to the L2 constraint."""
        model = PauliLindbladModel(
            gate_set_cz, {"CZ": QubitSparsePauliList(["ZI", "IZ"]), **_PM_GENS}
        )
        path0 = make_cz_path("XI")  # row = {CZ:ZI: 4.0}
        path1 = make_cz_path("IX")  # row = {CZ:IZ: 4.0}

        weights = IndexedMatrix()
        weights.add_rows(
            row_indices=[path0, path1],
            rows=[
                IndexedVector({path0: 2.0, path1: 0.0}),
                IndexedVector({path0: 0.0, path1: 1.0}),
            ],
        )

        solver = PositivityMinSolve(
            coefficients={"CZ": 1.0},
            epsilon=1e-6,
            deltas={path0: 10.0, path1: 10.0},
            weights=weights,
        )

        fit = Fit(model=model)
        fit[AveragedData] = make_averaged_data([(path0, -1, 0.9), (path1, -1, 0.8)])
        result = solver.run(fit)

        # Tight weighted L2 forces close to the LS solution.
        assert np.isclose(_get_rate_from_fit(result, "CZ", "ZI"), -np.log(0.9) / 4, atol=1e-4)
        assert np.isclose(_get_rate_from_fit(result, "CZ", "IZ"), -np.log(0.8) / 4, atol=1e-4)

    def test_metadata_contains_problem(self, gate_set_cz, make_cz_path, make_averaged_data):
        """Test that metadata contains the cvxpy Problem object."""
        model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI"]), **_PM_GENS})
        path = make_cz_path("XI")
        solver = PositivityMinSolve(
            coefficients={"CZ": 1.0},
            epsilon=10.0,
            deltas={path: 10.0},
        )

        fit = Fit(model=model)
        fit[AveragedData] = make_averaged_data([(path, -1, 0.8)])
        result = solver.run(fit)

        assert "problem" in result.model_data.metadata
        assert isinstance(result.model_data.metadata["problem"], cvxpy.Problem)

    def test_covariance_is_zeros(self, gate_set_cz, make_cz_path, make_averaged_data):
        """PositivityMinSolve returns zero covariance."""
        model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI"]), **_PM_GENS})
        path = make_cz_path("XI")
        solver = PositivityMinSolve(
            coefficients={"CZ": 1.0},
            epsilon=10.0,
            deltas={path: 10.0},
        )

        fit = Fit(model=model)
        fit[AveragedData] = make_averaged_data([(path, -1, 0.8)])
        result = solver.run(fit)

        cov = result.model_data.dataset["covariance"].values
        assert cov.shape == (1, 1)
        assert np.allclose(cov, 0.0)

    def test_underdetermined_minimizes_positivity(
        self, gate_set_cz, make_cz_path, make_averaged_data
    ):
        """In an underdetermined system, the solver minimizes the sum of max(0, x_i)."""
        model = PauliLindbladModel(
            gate_set_cz, {"CZ": QubitSparsePauliList(["ZI", "IZ"]), **_PM_GENS}
        )
        path = make_cz_path("XX")  # one row onto two generators: {CZ:ZI: 4.0, CZ:IZ: 4.0}
        solver = PositivityMinSolve(
            coefficients={"CZ": 1.0},
            epsilon=1e-6,
            deltas={path: 1e-6},
        )

        fit = Fit(model=model)
        fit[AveragedData] = make_averaged_data([(path, -1, 0.8)])
        result = solver.run(fit)

        x0 = _get_rate_from_fit(result, "CZ", "ZI")
        x1 = _get_rate_from_fit(result, "CZ", "IZ")
        # 4 * (x0 + x1) = -log(0.8), so the sum is -log(0.8) / 4.
        assert np.isclose(x0 + x1, -np.log(0.8) / 4, atol=1e-4)
        # The minimum of max(0,a) + max(0,b) subject to a + b = c > 0 is exactly c.
        pos_sum = max(0, x0) + max(0, x1)
        assert np.isclose(pos_sum, -np.log(0.8) / 4, atol=1e-4)
