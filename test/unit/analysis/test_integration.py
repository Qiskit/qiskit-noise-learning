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

from qiskit_noise_learning.analysis import (
    AnalysisPipeline,
    CurveFitObservables,
    Fit,
    NNLSSolve,
)
from qiskit_noise_learning.data import ModelData, ObservableData
from qiskit_noise_learning.math import IndexedVector


@dataclass(frozen=True)
class MockPath:
    name: str


class MockFidelityModel:
    def __init__(self, rows: dict):
        self._rows = rows

    def row_from_path(self, path):
        return self._rows[path]


def _make_observable_data(
    path_depths: dict[str, list[int]],
    a_values: dict[str, float],
    f_values: dict[str, float],
    n_rand: int = 20,
    noise_std: float = 0.005,
    seed: int = 42,
) -> tuple[ObservableData, dict[str, MockPath]]:
    rng = np.random.default_rng(seed)

    all_observables = []
    all_unbound_paths = []
    all_depths = []
    unbound_paths = {}

    for pp_name, depths in path_depths.items():
        pp = MockPath(pp_name)
        unbound_paths[pp_name] = pp
        for d in depths:
            true_val = a_values[pp_name] * f_values[pp_name] ** d
            values = true_val + rng.normal(0, noise_std, size=n_rand)
            all_observables.append(values)
            all_unbound_paths.append(pp)
            all_depths.append(d)

    n = len(all_observables)
    observables = np.stack(all_observables)
    return ObservableData.from_arrays(
        unbound_paths=all_unbound_paths,
        depths=all_depths,
        observables=observables,
        time_lbs=np.empty((n, n_rand), dtype="datetime64[us]"),
        time_ubs=np.empty((n, n_rand), dtype="datetime64[us]"),
    ), unbound_paths


class TestAnalysisPipelineIntegration:
    """Tests analysis pipeline on synthetic data."""

    def test_curve_fit_then_nnls_single_unbound_path(self):
        """Test curve fitting then linear solving."""
        a_true, f_true = 0.9, 0.8
        pp_name = "pp0"
        obs, unbound_paths = _make_observable_data(
            path_depths={pp_name: [1, 2, 3, 4, 5]},
            a_values={pp_name: a_true},
            f_values={pp_name: f_true},
        )
        model = MockFidelityModel({unbound_paths[pp_name]: IndexedVector({"theta": 1.0})})

        pipeline = AnalysisPipeline(CurveFitObservables(), NNLSSolve())
        assert pipeline.input_level is ObservableData
        assert pipeline.output_level is ModelData

        fit = Fit(model=model)
        fit[ObservableData] = obs
        result = pipeline.run(fit)

        assert isinstance(result.model_data, ModelData)
        assert np.isclose(
            result.model_data.dataset["parameter_values"].sel(parameter="theta").item(),
            -np.log(f_true),
            atol=0.05,
        )

    def test_curve_fit_then_nnls_multiple_unbound_paths(self):
        """Test curve fitting then linear solving with multiple unbound paths."""
        pp0, pp1 = "pp0", "pp1"
        f0, f1 = 0.9, 0.7
        obs, unbound_paths = _make_observable_data(
            path_depths={pp0: [1, 2, 3, 4, 5], pp1: [1, 2, 3, 4, 5]},
            a_values={pp0: 0.95, pp1: 0.85},
            f_values={pp0: f0, pp1: f1},
        )
        model = MockFidelityModel(
            {
                unbound_paths[pp0]: IndexedVector({"r0": 1.0}),
                unbound_paths[pp1]: IndexedVector({"r1": 1.0}),
            }
        )

        pipeline = AnalysisPipeline(CurveFitObservables(), NNLSSolve())
        fit = Fit(model=model)
        fit[ObservableData] = obs
        result = pipeline.run(fit)

        assert np.isclose(
            result.model_data.dataset["parameter_values"].sel(parameter="r0").item(),
            -np.log(f0),
            atol=0.05,
        )
        assert np.isclose(
            result.model_data.dataset["parameter_values"].sel(parameter="r1").item(),
            -np.log(f1),
            atol=0.05,
        )
