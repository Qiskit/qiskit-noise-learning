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

from qiskit_noise_learning.analysis import CurveFitObservables, Fit
from qiskit_noise_learning.analysis.curve_fit_observables import fit_exponential
from qiskit_noise_learning.data import ObservableData


def _run_stage(obs_data, paths=None):
    """Run CurveFitObservables on the given ObservableData."""
    fit = Fit(paths=paths or [])
    fit[ObservableData] = obs_data
    return CurveFitObservables().run(fit)


class TestCurveFitObservables:
    """Tests for the CurveFitObservables analysis stage."""

    def test_single_unbound_path(self, make_cz_path, make_observable_data):
        """Test fitting a single unbound path."""
        a_true, f_true = 0.9, 0.8
        pp = make_cz_path("IX")
        obs = make_observable_data([(pp, a_true, f_true, [1, 2, 3, 4, 5])])
        result = _run_stage(obs)
        ds = result.averaged_data.dataset
        mask = (ds["unbound_path"].data == pp) & (ds["fragment_depth"].data == -1)
        assert np.isclose(ds["metadata"].data[mask][0]["spam_fidelity"], a_true, atol=0.02)
        assert np.isclose(ds["observables"].data[mask][0], f_true, atol=0.02)

    def test_multiple_unbound_paths(self, make_cz_path, make_observable_data):
        """Test fitting multiple unbound paths."""
        pp0, pp1 = make_cz_path("IX"), make_cz_path("XI")
        obs = make_observable_data(
            [
                (pp0, 0.95, 0.9, [1, 2, 3, 4, 5]),
                (pp1, 0.85, 0.7, [1, 2, 3, 4, 5]),
            ]
        )
        result = _run_stage(obs)
        ds = result.averaged_data.dataset
        decay_mask = ds["fragment_depth"].data == -1

        mask0 = (ds["unbound_path"].data == pp0) & decay_mask
        assert np.isclose(ds["observables"].data[mask0][0], 0.9, atol=0.02)

        mask1 = (ds["unbound_path"].data == pp1) & decay_mask
        assert np.isclose(ds["observables"].data[mask1][0], 0.7, atol=0.02)

    def test_decay_fidelity_value(self, make_cz_path, make_observable_data):
        """Test the decay fidelity is close to the true value."""
        pp = make_cz_path("IX")
        obs = make_observable_data([(pp, 0.9, 0.8, [1, 2, 3, 4, 5])])
        result = _run_stage(obs)
        ds = result.averaged_data.dataset
        mask = (ds["unbound_path"].data == pp) & (ds["fragment_depth"].data == -1)
        fidelity = ds["observables"].data[mask][0]
        assert np.isclose(fidelity, 0.8, atol=0.02)

    def test_unbound_path_in_fit_paths_triggers_curve_fit(self, make_cz_path, make_observable_data):
        """Test that an unbound path in fit.paths is curve-fit."""
        pp = make_cz_path("IX")
        obs = make_observable_data([(pp, 0.9, 0.8, [1, 2, 3, 4, 5])])
        result = _run_stage(obs, paths=[pp])
        ds = result.averaged_data.dataset
        mask = (ds["unbound_path"].data == pp) & (ds["fragment_depth"].data == -1)
        assert np.isclose(ds["observables"].data[mask][0], 0.8, atol=0.02)

    def test_bound_path_in_fit_paths_triggers_average(self, make_cz_path, make_observable_data):
        """Test that a bound path (not in curve_fit_paths) is averaged, not curve-fit."""
        pp0, pp1 = make_cz_path("IX"), make_cz_path("XI")
        obs = make_observable_data(
            [
                (pp0, 0.9, 0.8, [1, 2, 3, 4, 5]),
                (pp1, 0.85, 0.7, [1, 2, 3]),
            ]
        )
        # Only pp0 is unbound (curve-fit); pp1 is not in fit.paths unbound set → averaged
        result = _run_stage(obs, paths=[pp0])
        ds = result.averaged_data.dataset

        # pp0 was curve-fit (fragment_depth = -1)
        mask0 = (ds["unbound_path"].data == pp0) & (ds["fragment_depth"].data == -1)
        assert mask0.any()

        # pp1 was averaged (fragment_depths 1, 2, 3 present, not fragment_depth -1)
        mask1_decay = (ds["unbound_path"].data == pp1) & (ds["fragment_depth"].data == -1)
        assert not mask1_decay.any()
        for d in [1, 2, 3]:
            mask1_d = (ds["unbound_path"].data == pp1) & (ds["fragment_depth"].data == d)
            assert mask1_d.any()

    def test_unbound_path_single_depth_raises(self, make_cz_path, make_observable_data):
        """Test that an unbound path in fit.paths with only 1 fragment_depth raises ValueError."""
        pp = make_cz_path("IX")
        obs = make_observable_data([(pp, 0.9, 0.8, [3])])
        with pytest.raises(ValueError, match="At least 2 fragment depths are required"):
            _run_stage(obs, paths=[pp])

    def test_reduced_chi_squared_recorded(self, make_cz_path, make_observable_data):
        """The per-row metadata records reduced chi-squared as the raw value over the dof."""
        pp = make_cz_path("IX")
        obs = make_observable_data([(pp, 0.9, 0.8, [1, 2, 3, 4, 5])])
        result = _run_stage(obs)
        ds = result.averaged_data.dataset
        mask = (ds["unbound_path"].data == pp) & (ds["depth"].data == -1)
        meta = ds["metadata"].data[mask][0]
        # dof = M - 2 = 5 - 2 = 3 fit parameters (a, f).
        assert np.isclose(meta["reduced_chi_squared"], meta["chi_squared"] / 3)

    def test_reduced_chi_squared_nan_without_dof(self, make_cz_path, make_observable_data):
        """Reduced chi-squared is nan when there are no degrees of freedom (2 depths)."""
        pp = make_cz_path("IX")
        obs = make_observable_data([(pp, 0.9, 0.8, [1, 2])])
        result = _run_stage(obs)
        ds = result.averaged_data.dataset
        mask = (ds["unbound_path"].data == pp) & (ds["depth"].data == -1)
        assert np.isnan(ds["metadata"].data[mask][0]["reduced_chi_squared"])


class TestFitExponential:
    """Tests for the fit_exponential helper function."""

    def test_chi_squared_with_zero_sigma(self):
        """Test that Chi-squared is nan when all uncertainties are zero."""
        fragment_depths = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = 0.9 * 0.8**fragment_depths
        y_err = np.zeros_like(y_data)

        _, _, _, _, chi_sq = fit_exponential(fragment_depths, y_data, y_err)
        assert np.isnan(chi_sq)

    def test_fallback_without_weights(self):
        """Test that when sigma causes curve_fit to fail, fallback still produces a fit."""
        fragment_depths = np.array([1.0, 2.0, 3.0])
        y_data = np.array([0.72, 0.576, 0.4608])  # a=0.9, f=0.8
        y_err = np.array([1e-15, 1e-15, 1e-15])

        a, f, _, _, _ = fit_exponential(fragment_depths, y_data, y_err)
        assert 0 < a <= 1
        assert 0 < f <= 1
