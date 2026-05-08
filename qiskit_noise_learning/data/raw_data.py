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

from typing import Self

import numpy as np
import xarray as xr

from qiskit_noise_learning.sequences import InstructionSequence

from .leveled_data import LeveledData
from .xarray_utils import filter_time, ragged_concat


def _measurement_maps_equal(
    map1: dict[str, np.ndarray], map2: dict[str, np.ndarray]
) -> bool:
    """Check equality of two measurement maps (dicts of str -> np.ndarray)."""
    return maps1.keys() == maps2.keys() and all(np.array_equal(map1[k], map2[k]) for k in map1)


class RawData(LeveledData):
    """Raw experimental outcome data associated with instruction sequences and classical registers.

    This class is a wrapper around a 1-layer deep XArray ``DataTree`` with arbitrary string keys.
    Each leaf dataset contains:

    - Data variables:
        - ``data``: The raw boolean data with dimensions ``("randomization", "shot", "bit")``.
        - ``data_mask``: A boolean mask with dimensions ``("randomization", "shot")``. Handles
          potential raggedness in the ``"shot"`` dimension across different randomizations.
        - ``measurement_flips``: A boolean array of measurement flips with dimensions
          ``("randomization", "bit")``.
        - ``time_lbs``: Lower bound on data acquisition times, with dimensions
          ``("randomization",)``, of type ``"datetime64[us]"``.
        - ``time_ubs``: Upper bound on data acquisition times, with dimensions
          ``("randomization",)``, of type ``"datetime64[us]"``.
    - Coordinates:
        - ``instruction_pattern``: The instruction pattern for the data, along dimension
          ``("randomization",)``, of type :class:`InstructionPattern`.
        - ``depth``: Integer array of depths along dimension ``("randomization",)``.
    - Attrs:
        - ``creg_names``: Ordered list of classical register names.
        - ``measurement_map``: Dictionary mapping creg names to arrays of measured qubit indices.
        - ``creg_bit_boundaries``: Dictionary mapping creg names to ``(start_idx, end_idx)`` tuples
          indicating the slice of the ``"bit"`` dimension for that register.

    Datasets are grouped by creg metadata: two datasets with the same ``creg_names`` and
    ``measurement_map`` are merged along the ``"randomization"`` dimension.

    Args:
        datatree: A datatree in the above format.
    """

    def __init__(self, datatree: xr.DataTree):
        self._datatree = datatree

    @property
    def datatree(self) -> xr.DataTree:
        """The data tree."""
        return self._datatree

    @classmethod
    def from_arrays(
        cls,
        creg_names: list[str],
        measurement_map: dict[str, np.ndarray],
        instruction_sequences: list[InstructionSequence],
        data: list[np.ndarray[np.bool_]],
        measurement_flips: list[np.ndarray[np.bool_]],
        time_lbs: list[np.ndarray[np.datetime64]],
        time_ubs: list[np.ndarray[np.datetime64]],
    ):
        """Instantiate from data specified as arrays.

        All instruction sequences must share the same creg structure (same ``creg_names`` and
        ``measurement_map``). The resulting ``RawData`` contains a single-leaf datatree.

        Args:
            creg_names: Ordered list of classical register names.
            measurement_map: Dictionary mapping creg names to arrays of measured physical qubit indices.
            instruction_sequences: The list of instruction sequences used to generate the
                experiments.
            data: A list of outcome data for each instruction sequence for all classical registers.
                The data has dimensions ``("randomization", "shot", "bit")``. Bits are ordered
                according to ``creg_names`` order, with each creg's bits contiguous.
            measurement_flips: A list of measurement flips to be applied to the data for each
                instruction sequence. Dimensions are ``("randomization", "bit")``.
            time_lbs: A lower bound on the data collection time for each randomization for a given
                instruction sequence. The dimensions are ``("randomization",)``.
            time_ubs: An upper bound on the data collection time for each randomization for a given
                instruction sequence. The dimensions are ``("randomization",)``.
        """
        creg_bit_boundaries = {}
        offset = 0
        for creg in creg_names:
            length = len(measurement_map[creg])
            creg_bit_boundaries[creg] = (offset, offset + length)
            offset += length

        raw_data = cls(datatree=xr.DataTree())
        for inst_sequence, inst_data, inst_meas_flips, inst_time_lbs, inst_time_ubs in zip(
            instruction_sequences, data, measurement_flips, time_lbs, time_ubs
        ):
            new_dataset = xr.Dataset(
                data_vars={
                    "data": xr.DataArray(data=inst_data, dims=["randomization", "shot", "bit"]),
                    "data_mask": xr.DataArray(
                        data=np.zeros(inst_data.shape[:2], dtype=bool),
                        dims=["randomization", "shot"],
                    ),
                    "measurement_flips": xr.DataArray(
                        data=inst_meas_flips, dims=["randomization", "bit"]
                    ),
                    "time_lbs": xr.DataArray(inst_time_lbs, dims=["randomization"]),
                    "time_ubs": xr.DataArray(inst_time_ubs, dims=["randomization"]),
                },
                coords={
                    "instruction_pattern": (
                        ("randomization",),
                        np.array([inst_sequence.pattern] * len(inst_data), dtype=object),
                    ),
                    "depth": (
                        ("randomization",),
                        np.array([inst_sequence.depth] * len(inst_data), dtype=int),
                    ),
                },
                attrs={
                    "creg_names": creg_names,
                    "measurement_map": measurement_map,
                    "creg_bit_boundaries": creg_bit_boundaries,
                },
            )

            raw_data = raw_data.merge(cls(xr.DataTree.from_dict({"0": new_dataset})))

        return raw_data

    def merge(self, other: Self) -> Self:
        """Merge with another raw data set.

        Datasets with matching creg metadata (``creg_names`` and ``measurement_map``) are
        concatenated along the ``"randomization"`` dimension. Potential raggedness of the
        ``"shot"`` dimension is handled via the ``"data_mask"`` data variable.

        Args:
            other: The other raw dataset.

        Returns:
            The merged data.
        """

        new_datatree = self.datatree.copy(deep=True)

        for _, other_node in other.datatree.items():
            other_ds = other_node.dataset
            matched_key = self._find_matching_key(new_datatree, other_ds.attrs)
            if matched_key is not None:
                new_datatree[matched_key] = ragged_concat(
                    datasets=[new_datatree[matched_key].dataset, other_ds],
                    concat_dim="randomization",
                    ragged_dim="shot",
                )
            else:
                new_key = str(len(new_datatree))
                new_datatree[new_key] = other_node

        return RawData(datatree=new_datatree)

    @staticmethod
    def _find_matching_key(datatree: xr.DataTree, attrs: dict) -> str | None:
        """Find a key in the datatree whose dataset has matching creg metadata."""
        for key, node in datatree.items():
            node_attrs = node.dataset.attrs
            if (
                node_attrs.get("creg_names") == attrs.get("creg_names")
                and _measurement_maps_equal(
                    node_attrs.get("measurement_map", {}), attrs.get("measurement_map", {})
                )
            ):
                return key
        return None

    def filter_time(self, lb: np.datetime64, ub: np.datetime64) -> Self:
        """Filter to data gathered within the time bounds.

        Args:
            lb: The time lower bound (inclusive).
            ub: The time upper bound (inclusive).

        Returns:
            The time filtered version of self.
        """
        return RawData(filter_time(self.datatree, lb=lb, ub=ub))
