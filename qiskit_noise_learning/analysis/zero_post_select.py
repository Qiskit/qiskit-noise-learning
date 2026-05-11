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

from collections.abc import Callable, Iterator
from typing import Literal

import xarray as xr

from qiskit_noise_learning.analysis import AnalysisStage
from qiskit_noise_learning.data import RawData


class ZeroPostSelect(AnalysisStage):
    """Apply a mask to raw data based on whether bit values are all False.

    This post-selection stage identifies cregs and masks shots whose bit patterns indicate
    failure. It can be configured to operate in one of two modes:

    * ``"node"``: Shots are discarded if any bit in the identified creg is True.
    * ``"edge"``: Shots are discarded if there exists a pair of neighbouring qubits in the
        coupling map for which both bits are True.

    Args:
        creg_identifier: A callable that, given a list of present creg names, returns an
            iterator over creg names to post-select on. Defaults to identifying cregs with naming
            pattern ``"*_ps"``.
        mode: Post-selection mode; either ``"node"`` or ``"edge"``.
    """

    def __init__(
        self,
        creg_identifier: Callable[[list[str]], Iterator[str]] | None = None,
        mode: Literal["node", "edge"] = "edge",
    ):
        self._creg_identifier = creg_identifier or suffix_creg_identifier()
        self._mode = mode

    @property
    def input_level(self):
        return RawData

    @property
    def output_level(self):
        return RawData

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def creg_identifier(self) -> Callable[[list[str]], Iterator[str]]:
        return self._creg_identifier

    def _run(self, fit):
        coupling_map = fit.model.gate_set.coupling_map

        def _dataset_masker(dataset: xr.Dataset) -> xr.Dataset:
            if "data" not in dataset:
                return dataset
            data = dataset["data"].values
            mask = dataset["data_mask"].values.copy()
            boundaries = dataset.attrs["creg_bit_boundaries"]
            creg_names = dataset.attrs["creg_names"]
            measurement_map = dataset.attrs["measurement_map"]

            for name in self._creg_identifier(creg_names):
                start, end = boundaries[name]
                creg_bits = data[:, :, start:end]

                if self._mode == "node":
                    mask |= creg_bits.any(axis=-1)
                elif self._mode == "edge":
                    qubit_indices = measurement_map[name]
                    for i, qi in enumerate(qubit_indices):
                        for j, qj in enumerate(qubit_indices):
                            if j <= i:
                                continue
                            if coupling_map.graph.has_edge(qi, qj):
                                mask |= creg_bits[:, :, i] & creg_bits[:, :, j]

            new_data_mask = xr.DataArray(data=mask, dims=["randomization", "shot"])
            return dataset.assign(data_mask=new_data_mask)

        fit[RawData] = RawData(fit.raw_data.datatree.map_over_datasets(_dataset_masker))


def suffix_creg_identifier(suffix: str = "ps") -> Callable[[list[str]], Iterator[str]]:
    def creg_identifier(creg_names):
        suffix_tag = f"_{suffix}"
        for name in creg_names:
            if name.endswith(suffix_tag):
                yield name

    return creg_identifier
