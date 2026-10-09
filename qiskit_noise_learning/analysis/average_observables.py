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

from collections.abc import Iterable

import numpy as np

from qiskit_noise_learning.analysis import AnalysisStage
from qiskit_noise_learning.data import AggregatedObservableData, ObservableData
from qiskit_noise_learning.data.xarray_utils import time_bound
from qiskit_noise_learning.sequences import Path


class AverageObservables(AnalysisStage):
    """Average observables over randomizations for each unbound path and fragment depth pair."""

    @property
    def input_level(self):
        return ObservableData

    @property
    def output_level(self):
        return AggregatedObservableData

    def _run(self, fit):
        fit[AggregatedObservableData] = average_observables(fit.observable_data)


def _group_rows_by_path_and_depth(
    unbound_paths: np.ndarray, fragment_depths: np.ndarray
) -> dict[Path, dict[int, np.ndarray]]:
    """Group observable row indices by unbound path, and then by fragment depth.

    A single pass over the label columns, replacing a boolean-mask dataset selection per path. The
    paths are keyed in the order they first appear and each path's fragment depths are ascending,
    which are the two orderings the aggregated output follows.

    Args:
        unbound_paths: The unbound path of each observable.
        fragment_depths: The fragment depth of each observable.

    Returns:
        A mapping from unbound path to a mapping from fragment depth to the indices of the
        observables having that path and depth.
    """
    rows: dict[Path, dict[int, list[int]]] = {}
    for row, (unbound_path, fragment_depth) in enumerate(zip(unbound_paths, fragment_depths)):
        rows.setdefault(unbound_path, {}).setdefault(int(fragment_depth), []).append(row)

    return {
        unbound_path: {
            fragment_depth: np.array(depth_rows, dtype=int)
            for fragment_depth, depth_rows in sorted(by_depth.items())
        }
        for unbound_path, by_depth in rows.items()
    }


def _group_estimate(observable_values: np.ndarray) -> tuple[float, float]:
    """Estimate one group of observables, pooling their randomizations.

    Args:
        observable_values: Values of every observable and randomization in the group. Entries of
            ``nan`` are padding of the ragged randomization dimension, and are excluded.

    Returns:
        The mean of the usable values, and its standard deviation. A single usable value has no
        sample spread, so its uncertainty is read from the mean as a binomial proportion, and a
        group with no usable values gives ``(nan, nan)``.
    """
    values = observable_values.flatten()
    values = values[~np.isnan(values)]

    if values.size == 0:
        return float("nan"), float("nan")

    mean = float(np.mean(values))
    if values.size == 1:
        p = (mean + 1) / 2
        return mean, float(np.sqrt(p * (1 - p)))

    return mean, float(np.std(values, ddof=1) / np.sqrt(values.size))


def average_observables(
    observable_data: ObservableData, unique_unbound_paths: Iterable[Path] | None = None
) -> AggregatedObservableData:
    """Compute averaged observables for the paths.

    The returned observables are ordered by ``unique_unbound_paths``, and by ascending fragment
    depth within each path.

    Args:
        observable_data: The observable data.
        unique_unbound_paths: The unbound paths to compute the averaged observables for, whose
            order the output follows. Defaults to all unbound paths in the observable data, in the
            order they first appear in it.
    """

    dataset = observable_data.dataset
    rows_by_path = _group_rows_by_path_and_depth(
        dataset["unbound_path"].data, dataset["fragment_depth"].data
    )
    if unique_unbound_paths is None:
        unique_unbound_paths = list(rows_by_path)

    observable_values = dataset["observable_values"].data
    all_time_lbs = dataset["time_lbs"].data
    all_time_ubs = dataset["time_ubs"].data

    obs_unbound_paths = []
    obs_fragment_depths = []
    obs_means = []
    obs_stds = []
    obs_time_lbs = []
    obs_time_ubs = []

    # A requested path absent from the data contributes no observables, rather than raising.
    for unbound_path in unique_unbound_paths:
        for fragment_depth, rows in rows_by_path.get(unbound_path, {}).items():
            mean, std = _group_estimate(observable_values[rows])

            obs_unbound_paths.append(unbound_path)
            obs_fragment_depths.append(fragment_depth)
            obs_means.append(mean)
            obs_stds.append(std)
            obs_time_lbs.append(time_bound(all_time_lbs[rows], "min"))
            obs_time_ubs.append(time_bound(all_time_ubs[rows], "max"))

    return AggregatedObservableData.from_arrays(
        unbound_paths=np.array(obs_unbound_paths, dtype=object),
        fragment_depths=np.array(obs_fragment_depths, dtype=int),
        estimate_values=np.array(obs_means, dtype=float),
        estimate_std=np.array(obs_stds, dtype=float),
        time_lbs=np.array(obs_time_lbs, dtype="datetime64[us]"),
        time_ubs=np.array(obs_time_ubs, dtype="datetime64[us]"),
    )
