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
from qiskit_noise_learning.sequences import Path


class AverageObservables(AnalysisStage):
    """Average observables over randomizations for each unbound path and depth pair."""

    @property
    def input_level(self):
        return ObservableData

    @property
    def output_level(self):
        return AveragedData

    def _run(self, fit):
        fit[AveragedData] = average_observables(fit.observable_data)


def average_observables(
    observable_data: ObservableData, unique_unbound_paths: set[Path] | None = None
) -> AveragedData:
    """Compute averaged observables for the paths.

    Args:
        observable_data: The observable data.
        unique_unbound_paths: A set of unbound paths to compute the averaged
            observables for. Defaults to all unbound paths in the observable data.
    """

    dataset = observable_data.dataset
    if unique_unbound_paths is None:
        unique_unbound_paths = set(dataset["unbound_path"].data)

    obs_unbound_paths = []
    obs_depths = []
    obs_means = []
    obs_stds = []
    obs_time_lbs = []
    obs_time_ubs = []

    for unbound_path in unique_unbound_paths:
        path_mask = dataset["unbound_path"].data == unbound_path
        path_dataset = dataset.sel({"observable": path_mask})

        for depth in sorted(set(path_dataset["depth"].data)):
            depth_mask = path_dataset["depth"].data == depth
            values = path_dataset["observables"].data[depth_mask].flatten()
            values = values[~np.isnan(values)]

            obs_unbound_paths.append(unbound_path)
            obs_depths.append(depth)
            obs_means.append(float(np.nanmean(values)))
            if values.size <= 1:
                p = (obs_means[-1] + 1) / 2
                obs_stds.append(np.sqrt(p * (1 - p)))
            else:
                obs_stds.append(float(np.std(values, ddof=1) / np.sqrt(values.size)))

            obs_time_lbs.append(time_bound(path_dataset["time_lbs"].data[depth_mask], "min"))
            obs_time_ubs.append(time_bound(path_dataset["time_ubs"].data[depth_mask], "max"))

    return AveragedData.from_arrays(
        unbound_paths=np.array(obs_unbound_paths, dtype=object),
        depths=np.array(obs_depths, dtype=int),
        observables=np.array(obs_means, dtype=float),
        std=np.array(obs_stds, dtype=float),
        time_lbs=np.array(obs_time_lbs, dtype="datetime64[us]"),
        time_ubs=np.array(obs_time_ubs, dtype="datetime64[us]"),
    )
