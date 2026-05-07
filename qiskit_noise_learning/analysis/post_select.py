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

from typing import Callable
import numpy as np
import xarray as xr

from qiskit_noise_learning.analysis import AnalysisStage
from qiskit_noise_learning.data import RawData
from qiskit_noise_learning.data.xarray_utils import time_bound
from qiskit_noise_learning.sequences import PathPattern


class PostSelect(AnalysisStage):
    """Apply a mask to raw data based on post-selection metadata."""

    def __init__(self, selector: Callable):
        """
        Args:
            selector: creg_bit_boundaries x (shot, bit) array -> bool
        
        Needs:
        - measure map (which qubit indices are measured in each classical register): could be
        included in raw data.
        - edges: coupling map basically --- needed for the "edge" mode post selection strategy.
        - primary_cregs: Can be deduced from raw data? Doesn't seem necessary. Could be an optoinal
        argument, and if not specified, will default to any creg with the suffix (then removed).

        More generically, could make this take a generic vectorized boolean function on classical
        registers, and have a 
        
        """
        self._selector = selector
    
    @staticmethod
    def flip_post_selection(mode: str, measure_map: dict[str, list[int]], edges: set[frozenset[int]], post_selection_suffix: str = "_ps", primary_cregs: list[str] | None = None):
        """Post select according to the "node" or "edge" mode.

        This could also alternatively be a subclass of a more generic PostSelect.
        
        """

        if mode == "node":

            def selector(creg_bit_boundaries, data):
                if primary_cregs is None:
                    primary_cregs = [
                        name[: -len(post_selection_suffix)]
                        for name in creg_bit_boundaries
                        if name.endswith(post_selection_suffix)
                    ]

                keep = np.ones(data.shape[0], dtype=bool)
                for creg in primary_cregs:
                    ps_creg = creg + post_selection_suffix
                    if creg not in creg_bit_boundaries or ps_creg not in creg_bit_boundaries:
                        continue
                    start, end = creg_bit_boundaries[creg]
                    ps_start, ps_end = creg_bit_boundaries[ps_creg]
                    keep &= np.all(data[:, start:end] == data[:, ps_start:ps_end], axis=1)
                return keep

            return PostSelect(selector)

        elif mode == "edge":
            # this requires knowing the measure_map and the edges
            # measure_map is natural to include in the dataset, so maybe not a required arg here
            # the edges can be grabbed from the model (couplign map in gate set)
            raise ValueError("not implemented")
        else:
            raise ValueError("Unrecognized mode")

    @property
    def input_level(self):
        return RawData

    @property
    def output_level(self):
        return RawData

    def _run(self, fit):
        """Note that a limitation here is that we need to check the creg data for every single
        entry. Maybe it makes sense to partition the datasets based on the creg information, rather
        than purely the number of bits.
        """
        def _dataset_selector(dataset: xr.Dataset) -> xr.Dataset:
            data = dataset["data"].data
            mask = dataset["data_mask"].data.copy()
            boundaries = dataset["creg_bit_boundaries"].data

            for rand_idx in range(data.shape[0]):
                keep = self._selector(boundaries[rand_idx], data[rand_idx])
                mask[rand_idx] |= ~keep

            new_data_mask = xr.DataArray(data=mask, dims=["randomization", "shot"])
            return dataset.assign(data_mask=new_data_mask)

        fit[RawData] = RawData(fit.raw_data.datatree.map_over_datasets(_dataset_selector))