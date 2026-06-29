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

from dataclasses import dataclass

import numpy as np
import pytest

from qiskit_noise_learning.analysis import Fit, PositivityMinSolve
from qiskit_noise_learning.data import AveragedData
from qiskit_noise_learning.math import IndexedMatrix, IndexedVector
from qiskit_noise_learning.models.pauli_lindblad_model import PauliLindbladModel

cvxpy = pytest.importorskip("cvxpy")


@dataclass(frozen=True)
class MockGeneratorIndex:
    """Minimal mock of GeneratorIndex with gate_name attribute."""

    gate_name: str
    label: str

    def __repr__(self):
        return f"{self.gate_name}:{self.label}"


class MockPauliLindbladModel(PauliLindbladModel):
    """A mock that passes isinstance checks but uses a simple row dictionary."""

    def __new__(cls, rows):
        instance = object.__new__(cls)
        return instance

    def __init__(self, rows):
        self._rows = rows

    def row_from_path(self, path):
        return self._rows[path]


def _make_decay_data(f_values, f_std_values=None):
    """Build AveragedData with depth=-1 entries."""
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


class TestPositivityMinSolve:
    """Tests for PositivityMinSolve."""

    def test_type_check_rejects_non_pauli_lindblad(self):
        """Raises TypeError if model is not a PauliLindbladModel."""

        class FakeModel:
            def row_from_path(self, path):
                return IndexedVector({"x": 1.0})

        solver = PositivityMinSolve(
            coefficients={"g": 1.0},
            epsilon=1.0,
            deltas={"pp0": 1.0},
        )
        fit = Fit(model=FakeModel())
        fit[AveragedData] = _make_decay_data({"pp0": 0.8})

        with pytest.raises(TypeError, match="PauliLindbladModel"):
            solver.run(fit)

    def test_loose_constraints_minimize_positivity(self):
        """With loose constraints, the optimizer makes the parameter non-positive."""
        pp = "pp0"
        gen = MockGeneratorIndex("G1", "r0")

        model = MockPauliLindbladModel({pp: IndexedVector({gen: 1.0})})
        solver = PositivityMinSolve(
            coefficients={"G1": 1.0},
            epsilon=10.0,
            deltas={pp: 10.0},
        )

        fit = Fit(model=model)
        fit[AveragedData] = _make_decay_data({pp: 0.8})
        result = solver.run(fit)

        x = result.model_data.dataset["parameter_values"].sel(parameter=gen).item()
        # The objective max(0, x) is minimized by pushing x <= 0
        assert x <= 1e-6

    def test_tight_constraints_recover_solution(self):
        """With tight epsilon and delta, the solution is forced near the LS solution."""
        pp = "pp0"
        f_true = 0.8
        b_expected = -np.log(f_true)
        gen = MockGeneratorIndex("G1", "r0")

        model = MockPauliLindbladModel({pp: IndexedVector({gen: 1.0})})
        solver = PositivityMinSolve(
            coefficients={"G1": 1.0},
            epsilon=1e-6,
            deltas={pp: 1e-6},
        )

        fit = Fit(model=model)
        fit[AveragedData] = _make_decay_data({pp: f_true})
        result = solver.run(fit)

        x = result.model_data.dataset["parameter_values"].sel(parameter=gen).item()
        assert np.isclose(x, b_expected, atol=1e-4)

    def test_multiple_parameters_same_gate(self):
        """Test with multiple parameters belonging to the same gate."""
        pp0, pp1 = "pp0", "pp1"
        f0, f1 = 0.9, 0.85
        gen0 = MockGeneratorIndex("G1", "r0")
        gen1 = MockGeneratorIndex("G1", "r1")

        model = MockPauliLindbladModel(
            {
                pp0: IndexedVector({gen0: 1.0}),
                pp1: IndexedVector({gen1: 1.0}),
            }
        )
        solver = PositivityMinSolve(
            coefficients={"G1": 1.0},
            epsilon=1e-6,
            deltas={pp0: 1e-6, pp1: 1e-6},
        )

        fit = Fit(model=model)
        fit[AveragedData] = _make_decay_data({pp0: f0, pp1: f1})
        result = solver.run(fit)

        model_data = result.model_data
        x0 = model_data.dataset["parameter_values"].sel(parameter=gen0).item()
        x1 = model_data.dataset["parameter_values"].sel(parameter=gen1).item()
        assert np.isclose(x0, -np.log(f0), atol=1e-4)
        assert np.isclose(x1, -np.log(f1), atol=1e-4)

    def test_multiple_gates_different_coefficients(self):
        """Parameters from different gates get different objective coefficients."""
        pp0, pp1 = "pp0", "pp1"
        f0, f1 = 0.9, 0.9
        gen_a = MockGeneratorIndex("A", "ra")
        gen_b = MockGeneratorIndex("B", "rb")

        # pp0: gen_a + gen_b = -log(f0), pp1: gen_a = -log(f1)
        model = MockPauliLindbladModel(
            {
                pp0: IndexedVector({gen_a: 1.0, gen_b: 1.0}),
                pp1: IndexedVector({gen_a: 1.0}),
            }
        )

        solver = PositivityMinSolve(
            coefficients={"A": 10.0, "B": 1.0},
            epsilon=1e-6,
            deltas={pp0: 1e-6, pp1: 1e-6},
        )

        fit = Fit(model=model)
        fit[AveragedData] = _make_decay_data({pp0: f0, pp1: f1})
        result = solver.run(fit)

        model_data = result.model_data
        xa = model_data.dataset["parameter_values"].sel(parameter=gen_a).item()
        xb = model_data.dataset["parameter_values"].sel(parameter=gen_b).item()
        # pp1 constrains gen_a = -log(f1), so gen_b = -log(f0) - (-log(f1)) = 0
        assert np.isclose(xa, -np.log(f1), atol=1e-4)
        assert np.isclose(xb, 0.0, atol=1e-4)

    def test_non_negative_constraint(self):
        """With non_negative=True, all parameters are >= 0."""
        pp0, pp1 = "pp0", "pp1"
        gen0 = MockGeneratorIndex("G1", "r0")
        gen1 = MockGeneratorIndex("G1", "r1")

        model = MockPauliLindbladModel(
            {
                pp0: IndexedVector({gen0: 1.0, gen1: -1.0}),
                pp1: IndexedVector({gen1: 1.0}),
            }
        )

        solver = PositivityMinSolve(
            coefficients={"G1": 1.0},
            epsilon=10.0,
            deltas={pp0: 10.0, pp1: 10.0},
            non_negative=True,
        )

        fit = Fit(model=model)
        fit[AveragedData] = _make_decay_data({pp0: 0.9, pp1: 0.8})
        result = solver.run(fit)

        model_data = result.model_data
        x0 = model_data.dataset["parameter_values"].sel(parameter=gen0).item()
        x1 = model_data.dataset["parameter_values"].sel(parameter=gen1).item()
        assert x0 >= -1e-8
        assert x1 >= -1e-8

    def test_weight_matrix(self):
        """Test that a weight matrix is applied to the L2 constraint."""
        pp0, pp1 = "pp0", "pp1"
        gen0 = MockGeneratorIndex("G1", "r0")
        gen1 = MockGeneratorIndex("G1", "r1")

        model = MockPauliLindbladModel(
            {
                pp0: IndexedVector({gen0: 1.0}),
                pp1: IndexedVector({gen1: 1.0}),
            }
        )

        weights = IndexedMatrix()
        weights.add_rows(
            row_indices=[pp0, pp1],
            rows=[
                IndexedVector({pp0: 2.0, pp1: 0.0}),
                IndexedVector({pp0: 0.0, pp1: 1.0}),
            ],
        )

        solver = PositivityMinSolve(
            coefficients={"G1": 1.0},
            epsilon=1e-6,
            deltas={pp0: 10.0, pp1: 10.0},
            weights=weights,
        )

        fit = Fit(model=model)
        fit[AveragedData] = _make_decay_data({pp0: 0.9, pp1: 0.8})
        result = solver.run(fit)

        model_data = result.model_data
        x0 = model_data.dataset["parameter_values"].sel(parameter=gen0).item()
        x1 = model_data.dataset["parameter_values"].sel(parameter=gen1).item()
        # Tight weighted L2 forces close to the LS solution
        assert np.isclose(x0, -np.log(0.9), atol=1e-4)
        assert np.isclose(x1, -np.log(0.8), atol=1e-4)

    def test_metadata_contains_problem(self):
        """Test that metadata contains the cvxpy Problem object."""
        pp = "pp0"
        gen = MockGeneratorIndex("G1", "r0")

        model = MockPauliLindbladModel({pp: IndexedVector({gen: 1.0})})
        solver = PositivityMinSolve(
            coefficients={"G1": 1.0},
            epsilon=10.0,
            deltas={pp: 10.0},
        )

        fit = Fit(model=model)
        fit[AveragedData] = _make_decay_data({pp: 0.8})
        result = solver.run(fit)

        assert "problem" in result.model_data.metadata
        assert isinstance(result.model_data.metadata["problem"], cvxpy.Problem)

    def test_covariance_is_zeros(self):
        """PositivityMinSolve returns zero covariance."""
        pp = "pp0"
        gen = MockGeneratorIndex("G1", "r0")

        model = MockPauliLindbladModel({pp: IndexedVector({gen: 1.0})})
        solver = PositivityMinSolve(
            coefficients={"G1": 1.0},
            epsilon=10.0,
            deltas={pp: 10.0},
        )

        fit = Fit(model=model)
        fit[AveragedData] = _make_decay_data({pp: 0.8})
        result = solver.run(fit)

        cov = result.model_data.dataset["covariance"].values
        assert cov.shape == (1, 1)
        assert np.allclose(cov, 0.0)

    def test_underdetermined_minimizes_positivity(self):
        """In an underdetermined system, the solver minimizes the sum of max(0, x_i)."""
        pp0 = "pp0"
        gen0 = MockGeneratorIndex("G1", "r0")
        gen1 = MockGeneratorIndex("G1", "r1")

        # One equation, two unknowns: r0 + r1 = -log(0.8)
        model = MockPauliLindbladModel(
            {
                pp0: IndexedVector({gen0: 1.0, gen1: 1.0}),
            }
        )

        solver = PositivityMinSolve(
            coefficients={"G1": 1.0},
            epsilon=1e-6,
            deltas={pp0: 1e-6},
        )

        fit = Fit(model=model)
        fit[AveragedData] = _make_decay_data({pp0: 0.8})
        result = solver.run(fit)

        model_data = result.model_data
        x0 = model_data.dataset["parameter_values"].sel(parameter=gen0).item()
        x1 = model_data.dataset["parameter_values"].sel(parameter=gen1).item()

        # The sum must equal -log(0.8)
        assert np.isclose(x0 + x1, -np.log(0.8), atol=1e-4)
        # The minimum of max(0,a) + max(0,b) subject to a+b=c>0 is exactly c
        pos_sum = max(0, x0) + max(0, x1)
        assert np.isclose(pos_sum, -np.log(0.8), atol=1e-4)
