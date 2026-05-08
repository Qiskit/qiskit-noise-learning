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
from qiskit_noise_learning.data.xarray_utils import time_bound
from qiskit_noise_learning.sequences import PathPattern


class FlipPostSelect(AnalysisStage):
    """Apply a mask to raw data based on bit flips across measurement outcomes.

    This post selection stage is based on identifying successful bit flips on the same qubit(s)
    between two measurements. It can be configured to operate in one of two modes:
    * ``"node"``: Shots are discarded if at least one bit failed to flip.
    * ``"edge"``: Shots are discarded if there exists a pair of neighbouring qubits in the
        measurement for which both bits failed to flip.

    Args:
        creg_pair_identifier: A callable that, given a list of present creg names, returns an
            iterator over pairs of creg names for which to do the flip-based post selection on. 
            Defaults to returning pairs of cregs with names of the form ``"*"`` and ``"*_ps"``.
        mode: Post-selection mode; either ``"node"`` or ``"edge"``.
    """

    def __init__(
        self, 
        creg_pair_identifier: Callable[[list[str]], Iterator[tuple[str, str]]] | None = None,
        mode: Literal["node"] | Literal["edge"] = "edge",
    ):
        if creg_pair_identifier is None:
            creg_pair_identifier = FlipPostSelect.from_suffix().creg_pair_identifier
        self._creg_pair_identifier = creg_pair_identifier
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
    def creg_pair_identifier(self) -> Callable[[list[str]], Iterator[tuple[str, str]]]:
        return self._creg_pair_identifier
    
    @staticmethod
    def from_list(creg_pairs: list[tuple[str, str]], mode: Literal["node"] | Literal["edge"] = "edge") -> Self:
        """Create from a pre-defined list of pairs of creg names.

        If a given pair is not found in the supplied ``creg_names``, it will be skipped.
        
        Args:
            creg_pairs: A list of pairs of creg names.
            mode: Flip post-selection mode.
        """

        def creg_pair_identifier(creg_names):
            for creg_pair in creg_pairs:
                if creg_pair[0] in creg_names and creg_pair[1] in creg_names:
                    yield creg_pair
        
        return FlipPostSelect(creg_pair_identifier=creg_pair_identifier, mode=mode)
    
    @staticmethod
    def from_suffix(suffix: str = "ps", mode:  Literal["node"] | Literal["edge"] = "edge") -> Self:
        """Defines the creg pair identifier to find pairs with names ``"*"`` and ``f"*_{suffix}"``.

        Any cregs that do not have a corresponding pair according to this rule are ignored.

        Args:
            suffix: The suffix for creg name pattern matching.
            mode: The post selection mode.
        """

        def creg_pair_identifier(creg_names):
            suffix_tag = f"_{suffix}"
            for name in creg_names:
                if name.endswith(suffix_tag):
                    base = name[: -len(suffix_tag)]
                    if base in creg_names:
                        yield (base, name)

        return FlipPostSelect(creg_pair_identifier=creg_pair_identifier, mode=mode)


    def _run(self, fit):
        coupling_map = fit.model.gate_set.coupling_map

        def _dataset_selector(dataset: xr.Dataset) -> xr.Dataset:
            if "data" not in dataset:
                return dataset
            data = dataset["data"].values
            mask = dataset["data_mask"].values.copy()
            boundaries = dataset.attrs["creg_bit_boundaries"]
            creg_names = dataset.attrs["creg_names"]
            measurement_map = dataset.attrs["measurement_map"]

            for base_name, ps_name in self._creg_pair_identifier(creg_names):
                base_qubits = measurement_map[base_name]
                ps_qubits = measurement_map[ps_name]
                if not np.array_equal(base_qubits, ps_qubits):
                    raise ValueError(
                        f"Cregs '{base_name}' and '{ps_name}' do not measure the same qubits."
                    )

                base_start, base_end = boundaries[base_name]
                ps_start, ps_end = boundaries[ps_name]

                base_bits = data[:, :, base_start:base_end]
                ps_bits = data[:, :, ps_start:ps_end]

                failed_to_flip = base_bits == ps_bits

                if self._mode == "node":
                    mask |= failed_to_flip.any(axis=-1)
                elif self._mode == "edge":
                    qubit_indices = measurement_map[base_name]
                    for i, qi in enumerate(qubit_indices):
                        for j, qj in enumerate(qubit_indices):
                            if j <= i:
                                continue
                            if coupling_map.graph.has_edge(qi, qj):
                                mask |= failed_to_flip[:, :, i] & failed_to_flip[:, :, j]

            new_data_mask = xr.DataArray(data=mask, dims=["randomization", "shot"])
            return dataset.assign(data_mask=new_data_mask)

        fit[RawData] = RawData(fit.raw_data.datatree.map_over_datasets(_dataset_selector))
