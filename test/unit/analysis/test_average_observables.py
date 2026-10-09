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

import warnings

import numpy as np
import pytest

from qiskit_noise_learning.analysis.average_observables import average_observables
from qiskit_noise_learning.data import ObservableData


def _single_group(path, observable_values, time_lbs=None, time_ubs=None):
    """Observable data whose observables all share one path and fragment depth."""
    observable_values = np.asarray(observable_values, dtype=float)
    unset = np.full(observable_values.shape, np.datetime64("NaT"), dtype="datetime64[us]")
    return ObservableData.from_arrays(
        unbound_paths=[path] * len(observable_values),
        fragment_depths=[1] * len(observable_values),
        observable_values=observable_values,
        time_lbs=unset if time_lbs is None else np.asarray(time_lbs, dtype="datetime64[us]"),
        time_ubs=unset if time_ubs is None else np.asarray(time_ubs, dtype="datetime64[us]"),
    )


class TestAverageObservables:
    """Tests for the ``average_observables`` function."""

    def test_output_order_follows_first_seen_paths(self, make_cz_path, make_observable_data):
        """Observables are ordered by first appearance of their path, then by fragment depth.

        The fragment depths are supplied in descending order, so an ascending output order
        can only have come from this function.
        """
        paths = [make_cz_path(label) for label in ("XI", "IX", "XX", "YI", "IY")]
        obs = make_observable_data([(path, 0.9, 0.8, [2, 1]) for path in paths])

        dataset = average_observables(obs).dataset

        assert list(dataset["unbound_path"].data) == [path for path in paths for _ in range(2)]
        assert list(dataset["fragment_depth"].data) == [1, 2] * len(paths)

    def test_requested_path_order_is_respected(self, make_cz_path, make_observable_data):
        """The output follows the order of the requested paths, not that of the data."""
        paths = [make_cz_path(label) for label in ("XI", "IX", "XX")]
        obs = make_observable_data([(path, 0.9, 0.8, [1]) for path in paths])

        requested = [paths[2], paths[0]]
        dataset = average_observables(obs, unique_unbound_paths=requested).dataset

        assert list(dataset["unbound_path"].data) == requested

    @pytest.mark.parametrize(
        "observable_values, expected_estimate, expected_std",
        [
            pytest.param([[0.2, 0.4], [0.6, 0.8]], 0.5, 0.12909944487358055, id="pooled"),
            pytest.param([[0.2, 0.4, np.nan]], 0.3, 0.1, id="padding-excluded"),
            pytest.param([[0.5, np.nan]], 0.5, 0.4330127018922193, id="one-usable-value"),
            pytest.param([[np.nan, np.nan]], np.nan, np.nan, id="no-usable-value"),
        ],
    )
    def test_estimate_pools_the_group(
        self, make_cz_path, observable_values, expected_estimate, expected_std
    ):
        """One estimate per path and depth, pooling the usable values of every observable in it.

        Two observables share a path and depth when data from more than one classical layout has
        been merged, and ``nan`` marks padding of the ragged randomization dimension. A single
        usable value has no sample spread, so its uncertainty comes from the estimate itself. None
        of these warns, including the case with nothing usable to average.
        """
        obs = _single_group(make_cz_path("XI"), observable_values)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dataset = average_observables(obs).dataset

        assert len(dataset["estimate_values"].data) == 1
        assert dataset["estimate_values"].data[0] == pytest.approx(expected_estimate, nan_ok=True)
        assert dataset["estimate_std"].data[0] == pytest.approx(expected_std, nan_ok=True)

    def test_time_bounds_span_the_group(self, make_cz_path):
        """The bounds are the earliest lower and latest upper bound over the group's observables."""
        obs = _single_group(
            make_cz_path("XI"),
            [[0.2, 0.4], [0.6, 0.8]],
            time_lbs=[["2026-01-03", "2026-01-02"], ["2026-01-05", "2026-01-01"]],
            time_ubs=[["2026-01-04", "2026-01-06"], ["2026-01-02", "2026-01-03"]],
        )

        dataset = average_observables(obs).dataset

        assert dataset["time_lbs"].data[0] == np.datetime64("2026-01-01")
        assert dataset["time_ubs"].data[0] == np.datetime64("2026-01-06")
