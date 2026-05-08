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

from typing import Callable, Iterator, Literal, Self

import numpy as np
import xarray as xr

from qiskit_noise_learning.analysis import AnalysisStage
from qiskit_noise_learning.data import RawData


class PostSelect(AnalysisStage):
    """Apply a mask to raw data based on whether bit values are all False.

    This post-selection stage identifies cregs and masks shots whose bit patterns indicate
    failure. It can be configured to operate in one of two modes:

    * ``"node"``: Shots are discarded if any bit in the identified creg is True.
    * ``"edge"``: Shots are discarded if there exists a pair of neighbouring qubits in the
        coupling map for which both bits are True.

    Args:
        creg_identifier: A callable that, given a list of present creg names, returns an
            iterator over creg names to post-select on.
        mode: Post-selection mode; either ``"node"`` or ``"edge"``.
    """

    def __init__(
        self,
        creg_identifier: Callable[[list[str]], Iterator[str]] | None = None,
        mode: Literal["node"] | Literal["edge"] = "edge",
    ):
        if creg_identifier is None:
            creg_identifier = PostSelect.from_suffix().creg_identifier
        self._creg_identifier = creg_identifier
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

    @staticmethod
    def from_list(
        creg_names_list: list[str],
        mode: Literal["node"] | Literal["edge"] = "edge",
    ) -> Self:
        """Create from a pre-defined list of creg names.

        If a given creg is not found in the dataset's creg names, it will be skipped.

        Args:
            creg_names_list: A list of creg names to post-select on.
            mode: Post-selection mode.
        """

        def creg_identifier(creg_names):
            for name in creg_names_list:
                if name in creg_names:
                    yield name

        return PostSelect(creg_identifier=creg_identifier, mode=mode)

    @staticmethod
    def from_suffix(
        suffix: str = "ps",
        mode: Literal["node"] | Literal["edge"] = "edge",
    ) -> Self:
        """Identify cregs by suffix for post-selection.

        Cregs whose names end with ``f"_{suffix}"`` are identified for post-selection.
        Cregs without this suffix are ignored.

        Args:
            suffix: The suffix for creg name pattern matching.
            mode: Post-selection mode.
        """

        def creg_identifier(creg_names):
            suffix_tag = f"_{suffix}"
            for name in creg_names:
                if name.endswith(suffix_tag):
                    yield name

        return PostSelect(creg_identifier=creg_identifier, mode=mode)

    def _run(self, fit):
        coupling_map = fit.model.gate_set.coupling_map

        def _dataset_selector(dataset: xr.Dataset) -> xr.Dataset:
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

        fit[RawData] = RawData(fit.raw_data.datatree.map_over_datasets(_dataset_selector))
