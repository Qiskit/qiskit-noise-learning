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

from qiskit_noise_learning.sequences import Path
from qiskit_noise_learning.visualizations.path_data.data_adapters import (
    _dataset_paths,
    averaged_data_points,
    exponential_fit_curves,
    model_curves,
    observable_data_points,
)

# --------------------------------------------------------------------------------------------------
# exponential_fit_curves
# --------------------------------------------------------------------------------------------------


def test_exponential_fit_curves_extracts_base_and_spam_intercept(make_cz_path, make_averaged_data):
    p = make_cz_path("XI")
    averaged = make_averaged_data([(p, -1, 0.8, {"spam_fidelity": 0.95})])
    bases, intercepts = exponential_fit_curves(averaged)
    assert bases[p] == pytest.approx(0.8)
    assert intercepts[p] == pytest.approx(0.95)


def test_exponential_fit_curves_intercept_defaults_to_one(make_cz_path, make_averaged_data):
    p = make_cz_path("XI")
    averaged = make_averaged_data([(p, -1, 0.8)])
    _, intercepts = exponential_fit_curves(averaged)
    assert intercepts[p] == pytest.approx(1.0)


def test_exponential_fit_curves_ignores_point_rows(make_cz_path, make_averaged_data):
    p = make_cz_path("XI")
    # A depth>=0 point row is not an exponential-fit (depth == -1) entry.
    averaged = make_averaged_data([(p, 0, 0.9), (p, -1, 0.8)])
    bases, _ = exponential_fit_curves(averaged)
    assert list(bases) == [p]
    assert bases[p] == pytest.approx(0.8)


def test_exponential_fit_curves_restricts_to_paths(make_cz_path, make_averaged_data):
    p0, p1 = make_cz_path("XI"), make_cz_path("IX")
    averaged = make_averaged_data([(p0, -1, 0.8), (p1, -1, 0.7)])
    bases, _ = exponential_fit_curves(averaged, paths=[p0])
    assert list(bases) == [p0]


# --------------------------------------------------------------------------------------------------
# averaged_data_points
# --------------------------------------------------------------------------------------------------


def test_averaged_data_points_extracts_point_rows_with_stds(make_cz_path, make_averaged_data):
    p = make_cz_path("XI")
    averaged = make_averaged_data([(p, -1, 0.8), (p, 0, 0.9, 0.01), (p, 2, 0.7, 0.02)])
    points = averaged_data_points(averaged)
    series = points[p]
    # Only the two depth>=0 rows, not the depth == -1 base row.
    assert sorted(series.xs) == [0.0, 2.0]
    assert series.stds is not None
    assert set(np.round(series.stds, 3)) == {0.01, 0.02}


# --------------------------------------------------------------------------------------------------
# observable_data_points
# --------------------------------------------------------------------------------------------------


def test_observable_data_points_flattens_randomizations(make_cz_path, make_observable_data):
    p = make_cz_path("XI")
    obs = make_observable_data([(p, 1.0, 0.9, [0, 1, 2])], n_rand=5)
    points = observable_data_points(obs)
    series = points[p]
    # 3 depths x 5 randomizations, all retained (no nan padding here).
    assert series.xs.size == 15
    assert series.stds is None


# --------------------------------------------------------------------------------------------------
# model_curves
# --------------------------------------------------------------------------------------------------


def test_model_curves_base_and_intercept(make_cz_path, make_fidelity_model_data):
    p = make_cz_path("XI")
    rate = 0.05
    model, model_data = make_fidelity_model_data([p], rate_default=rate)
    bases, intercepts = model_curves(model, model_data, [p])
    # base = product of repeatable-fragment fidelities = exp(-rate * n_repeatable).
    assert bases[p] == pytest.approx(np.exp(-rate * len(p.repeatable_fragment)))
    # intercept = product over start+end fragments = exp(-rate * (n_start + n_end)).
    n_spam = len(p.start_fragment) + len(p.end_fragment)
    assert intercepts[p] == pytest.approx(np.exp(-rate * n_spam))


def test_model_curves_raises_on_bound_path(make_cz_path, make_fidelity_model_data):
    p = make_cz_path("XI")
    model, model_data = make_fidelity_model_data([p])
    with pytest.raises(ValueError, match="unbound decay paths"):
        model_curves(model, model_data, [p.bind_at(3)])


def test_model_curves_raises_on_empty_repeatable(make_cz_path, make_fidelity_model_data):
    p = make_cz_path("XI")
    model, model_data = make_fidelity_model_data([p])
    non_decay = Path(
        start_fragment=p.start_fragment, repeatable_fragment=[], end_fragment=p.end_fragment
    )
    with pytest.raises(ValueError, match="unbound decay paths"):
        model_curves(model, model_data, [non_decay])


def test_model_curves_raises_on_non_fidelity_model(make_cz_path, make_fidelity_model_data):
    p = make_cz_path("XI")
    _, model_data = make_fidelity_model_data([p])

    class _NotFidelity:
        output_space = object()

    with pytest.raises(ValueError, match="fidelity model"):
        model_curves(_NotFidelity(), model_data, [p])


# --------------------------------------------------------------------------------------------------
# _dataset_paths
# --------------------------------------------------------------------------------------------------


def test_dataset_paths_unique_first_seen_order(make_cz_path, make_averaged_data):
    p0, p1 = make_cz_path("XI"), make_cz_path("IX")
    averaged = make_averaged_data([(p0, -1, 0.8), (p1, -1, 0.7), (p0, 0, 0.9)])
    assert _dataset_paths(averaged) == [p0, p1]


def test_dataset_paths_skips_none():
    assert _dataset_paths(None) == []
