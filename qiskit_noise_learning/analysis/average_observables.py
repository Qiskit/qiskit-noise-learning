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

from qiskit_noise_learning.analysis import AnalysisStage
from qiskit_noise_learning.data import AveragedData, ObservableData
from qiskit_noise_learning.data.xarray_utils import time_bound
from qiskit_noise_learning.sequences import PathPattern


class AverageObservables(AnalysisStage):
    """Average observables over randomizations for each unique (path_pattern, depth) pair."""

    @property
    def input_level(self):
        return ObservableData

    @property
    def output_level(self):
        return AveragedData

    def _run(self, fit):
        fit[AveragedData] = average_observables(fit.observable_data)


def average_observables(
    observable_data: ObservableData, unique_path_patterns: set[PathPattern] | None = None
) -> AveragedData:
    """Compute averaged observables for the given set of path patterns.

    Args:
        observable_data: The observable data.
        unique_path_patterns: A collection of unique patterns to compute the averaged observables
            for. Defaults to all of the patterns in the observable data.
    """

    dataset = observable_data.dataset
    if unique_path_patterns is None:
        unique_path_patterns = set(dataset["path_pattern"].data)

    obs_path_patterns = []
    obs_depths = []
    obs_means = []
    obs_stds = []
    obs_time_lbs = []
    obs_time_ubs = []

    for pp in unique_path_patterns:
        pp_mask = dataset["path_pattern"].data == pp
        pp_dataset = dataset.sel({"observable": pp_mask})

        for depth in sorted(set(pp_dataset["depth"].data)):
            depth_mask = pp_dataset["depth"].data == depth
            values = pp_dataset["observables"].data[depth_mask].flatten()
            values = values[~np.isnan(values)]

            obs_path_patterns.append(pp)
            obs_depths.append(depth)
            obs_means.append(float(np.nanmean(values)))
            if values.size <= 1:
                p = (obs_means[-1] + 1) / 2
                obs_stds.append(np.sqrt(p * (1 - p)))
            else:
                obs_stds.append(float(np.std(values, ddof=1) / np.sqrt(values.size)))

            obs_time_lbs.append(time_bound(pp_dataset["time_lbs"].data[depth_mask], "min"))
            obs_time_ubs.append(time_bound(pp_dataset["time_ubs"].data[depth_mask], "max"))

    return AveragedData.from_arrays(
        path_patterns=np.array(obs_path_patterns, dtype=object),
        depths=np.array(obs_depths, dtype=int),
        observables=np.array(obs_means, dtype=float),
        std=np.array(obs_stds, dtype=float),
        time_lbs=np.array(obs_time_lbs, dtype="datetime64[us]"),
        time_ubs=np.array(obs_time_ubs, dtype="datetime64[us]"),
    )
