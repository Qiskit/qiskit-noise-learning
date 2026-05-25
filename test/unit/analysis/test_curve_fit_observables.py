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

from qiskit_noise_learning.analysis import CurveFitObservables, Fit
from qiskit_noise_learning.analysis.curve_fit_observables import fit_exponential
from qiskit_noise_learning.data import ObservableData


@dataclass(frozen=True)
class MockPath:
    name: str


def _make_observable_data(
    unbound_path_depths: dict,
    a_values: dict,
    f_values: dict,
    n_rand: int = 20,
    noise_std: float = 0.005,
    seed: int = 42,
) -> tuple[ObservableData, dict[str, MockPath]]:
    """Build synthetic ObservableData with MockPath keys.

    Returns the ObservableData and a dict mapping path name to MockPath.
    """
    rng = np.random.default_rng(seed)

    all_observables = []
    all_unbound_paths = []
    all_depths = []
    unbound_paths = {}

    for pp_name, depths in unbound_path_depths.items():
        pp = MockPath(pp_name)
        unbound_paths[pp_name] = pp
        for d in depths:
            true_val = a_values[pp_name] * f_values[pp_name] ** d
            noise = rng.normal(0, noise_std, size=n_rand)
            values = true_val + noise
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


def _run_stage(obs_data):
    """Run CurveFitObservables on the given ObservableData."""
    fit = Fit()
    fit[ObservableData] = obs_data
    return CurveFitObservables().run(fit)


class TestCurveFitObservables:
    """Tests for the CurveFitObservables analysis stage."""

    def test_single_unbound_path(self):
        """Test fitting a single unbound path."""
        a_true, f_true = 0.9, 0.8
        pp = "pp0"
        obs, unbound_paths = _make_observable_data(
            unbound_path_depths={pp: [1, 2, 3, 4, 5]},
            a_values={pp: a_true},
            f_values={pp: f_true},
        )
        result = _run_stage(obs)
        ds = result.averaged_data.dataset
        mask = (ds["unbound_path"].data == unbound_paths[pp]) & (ds["depth"].data == -1)
        assert np.isclose(ds["metadata"].data[mask][0]["spam_fidelity"], a_true, atol=0.02)
        assert np.isclose(ds["observables"].data[mask][0], f_true, atol=0.02)

    def test_multiple_unbound_paths(self):
        """Test fitting multiple unbound paths."""
        pp0, pp1 = "pp0", "pp1"
        obs, unbound_paths = _make_observable_data(
            unbound_path_depths={pp0: [1, 2, 3, 4, 5], pp1: [1, 2, 3, 4, 5]},
            a_values={pp0: 0.95, pp1: 0.85},
            f_values={pp0: 0.9, pp1: 0.7},
        )
        result = _run_stage(obs)
        ds = result.averaged_data.dataset
        decay_mask = ds["depth"].data == -1

        mask0 = (ds["unbound_path"].data == unbound_paths[pp0]) & decay_mask
        assert np.isclose(ds["observables"].data[mask0][0], 0.9, atol=0.02)

        mask1 = (ds["unbound_path"].data == unbound_paths[pp1]) & decay_mask
        assert np.isclose(ds["observables"].data[mask1][0], 0.7, atol=0.02)

    def test_decay_fidelity_value(self):
        """Test the decay fidelity is close to the true value."""
        pp = "pp0"
        obs, unbound_paths = _make_observable_data(
            unbound_path_depths={pp: [1, 2, 3, 4, 5]},
            a_values={pp: 0.9},
            f_values={pp: 0.8},
        )
        result = _run_stage(obs)
        ds = result.averaged_data.dataset
        mask = (ds["unbound_path"].data == unbound_paths[pp]) & (ds["depth"].data == -1)
        fidelity = ds["observables"].data[mask][0]
        assert np.isclose(fidelity, 0.8, atol=0.02)


class TestFitExponential:
    """Tests for the fit_exponential helper function."""

    def test_chi_squared_with_zero_sigma(self):
        """Test that Chi-squared is nan when all uncertainties are zero."""
        depths = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = 0.9 * 0.8**depths
        y_err = np.zeros_like(y_data)

        _, _, _, _, chi_sq = fit_exponential(depths, y_data, y_err)
        assert np.isnan(chi_sq)

    def test_fallback_without_weights(self):
        """Test that when sigma causes curve_fit to fail, fallback still produces a fit."""
        depths = np.array([1.0, 2.0, 3.0])
        y_data = np.array([0.72, 0.576, 0.4608])  # a=0.9, f=0.8
        y_err = np.array([1e-15, 1e-15, 1e-15])

        a, f, _, _, _ = fit_exponential(depths, y_data, y_err)
        assert 0 < a <= 1
        assert 0 < f <= 1
