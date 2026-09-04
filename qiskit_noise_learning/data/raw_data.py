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
from .measurement_register import MeasurementRegister
from .xarray_utils import filter_time, ragged_concat

# The coordinates over the "bit" dimension that give each bit its identity. Two datasets may be
# merged only if they agree on all of them.
_BIT_COORD_NAMES = ("creg_name", "qubit_idx", "measuring_gate_idx")


def _bit_coords(registers: list[MeasurementRegister]) -> dict[str, tuple[tuple[str], np.ndarray]]:
    """Return the ``"bit"`` coordinates for registers laid out contiguously in the given order."""
    names = [register.name for register in registers for _ in range(register.num_bits)]
    qubit_idxs = [idx for register in registers for idx in register.qubit_idxs]
    gate_idxs = [register.measuring_gate_idx for register in registers for _ in register.qubit_idxs]
    return {
        "creg_name": (("bit",), np.array(names, dtype=np.str_)),
        "qubit_idx": (("bit",), np.array(qubit_idxs, dtype=int)),
        "measuring_gate_idx": (("bit",), np.array(gate_idxs, dtype=int)),
    }


class RawData(LeveledData):
    """Raw experimental outcome data associated with instruction sequences and classical registers.

    This class is a wrapper around a 1-layer deep XArray ``DataTree`` with arbitrary string keys.
    Each leaf dataset contains:

    - Data variables:

        - ``data``: The raw boolean data with dimensions ``("randomization", "shot", "bit")``.
        - ``data_mask``: A boolean mask with dimensions ``("randomization", "shot")``, where
          ``True`` marks a shot to exclude, following the ``numpy.ma`` convention. Handles
          potential raggedness in the ``"shot"`` dimension across different randomizations, as
          well as the outcome of any post-selection.
        - ``measurement_flips``: A boolean array of measurement flips with dimensions
          ``("randomization", "bit")``.
        - ``time_lbs``: Lower bound on data acquisition times, with dimensions
          ``("randomization",)``, of type ``"datetime64[us]"``.
        - ``time_ubs``: Upper bound on data acquisition times, with dimensions
          ``("randomization",)``, of type ``"datetime64[us]"``.
    - Coordinates:

        - ``unbound_instruction_sequence``: The unbound instruction sequence for the data, along
          dimension ``("randomization",)``, of type :class:`InstructionSequence`.
        - ``fragment_depth``: Integer array of fragment depths along dimension
          ``("randomization",)``.
        - ``creg_name``: The name of the classical register each bit belongs to, along dimension
          ``("bit",)``.
        - ``qubit_idx``: The physical qubit index whose outcome each bit holds, along dimension
          ``("bit",)``.
        - ``measuring_gate_idx``: The measuring gate whose outcomes this bit's register holds,
          counted among the measuring gates of the instruction sequences only, in the order those
          gates are traversed, or ``-1`` if the register has no such gate. Along dimension
          ``("bit",)``.

    Together these three describe the ``"bit"`` dimension bit by bit, so a consumer never needs to
    know how the bits were laid out. Note that a register need not measure in ascending qubit order,
    and that the same physical qubit may be measured by more than one register.

    Only measuring gates are counted by ``measuring_gate_idx``, because one leaf may hold data from
    several fragment depths, so a position among *all* gates would not be a property of a bit.

    Datasets are grouped by their ``"bit"`` coordinates: two datasets whose bits carry the same
    ``creg_name``, ``qubit_idx`` and ``measuring_gate_idx`` values are merged along the
    ``"randomization"`` dimension.

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
        registers: list[MeasurementRegister],
        instruction_sequences: list[InstructionSequence],
        data: list[np.ndarray[np.bool_]],
        measurement_flips: list[np.ndarray[np.bool_]],
        time_lbs: list[np.ndarray[np.datetime64]],
        time_ubs: list[np.ndarray[np.datetime64]],
    ):
        """Instantiate from data specified as arrays.

        All instruction sequences must share the same registers. The resulting ``RawData`` contains
        a single-leaf datatree.

        Args:
            registers: The classical registers holding the data, in the order their bits are laid
                out along the ``"bit"`` dimension.
            instruction_sequences: The list of instruction sequences used to generate the
                experiments.
            data: A list of outcome data for each instruction sequence for all classical registers.
                The data has dimensions ``("randomization", "shot", "bit")``. Bits are ordered
                according to ``registers`` order, with each register's bits contiguous.
            measurement_flips: A list of measurement flips to be applied to the data for each
                instruction sequence. Dimensions are ``("randomization", "bit")``.
            time_lbs: A lower bound on the data collection time for each randomization for a given
                instruction sequence. The dimensions are ``("randomization",)``.
            time_ubs: An upper bound on the data collection time for each randomization for a given
                instruction sequence. The dimensions are ``("randomization",)``.

        Raises:
            ValueError: If the register names are not unique, if the per-sequence arguments do not
                all have the same length, or if an array's ``"bit"`` dimension does not match the
                total number of classical bits.
        """
        names = [register.name for register in registers]
        if len(set(names)) != len(names):
            raise ValueError(f"The register names must be unique, but got {names}.")

        lengths = {
            "instruction_sequences": len(instruction_sequences),
            "data": len(data),
            "measurement_flips": len(measurement_flips),
            "time_lbs": len(time_lbs),
            "time_ubs": len(time_ubs),
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"The per-sequence arguments must all have the same length, but got {lengths}."
            )

        bit_coords = _bit_coords(registers)
        num_bits = sum(register.num_bits for register in registers)

        raw_data = cls(datatree=xr.DataTree())
        for inst_sequence, inst_data, inst_meas_flips, inst_time_lbs, inst_time_ubs in zip(
            instruction_sequences, data, measurement_flips, time_lbs, time_ubs
        ):
            if inst_data.shape[-1] != num_bits:
                raise ValueError(
                    f"The cregs hold {num_bits} classical bits in total, but an entry of data has "
                    f"{inst_data.shape[-1]} along its 'bit' dimension."
                )
            if inst_meas_flips.shape[-1] != num_bits:
                raise ValueError(
                    f"The cregs hold {num_bits} classical bits in total, but an entry of "
                    f"measurement_flips has {inst_meas_flips.shape[-1]} along its 'bit' dimension."
                )

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
                    "unbound_instruction_sequence": (
                        ("randomization",),
                        np.array([inst_sequence.unbind()] * len(inst_data), dtype=object),
                    ),
                    "fragment_depth": (
                        ("randomization",),
                        np.array([inst_sequence.fragment_depth] * len(inst_data), dtype=int),
                    ),
                    **bit_coords,
                },
            )

            raw_data = raw_data.merge(cls(xr.DataTree.from_dict({"0": new_dataset})))

        return raw_data

    def merge(self, other: Self) -> Self:
        """Merge with another raw data set.

        Datasets whose ``"bit"`` coordinates agree are concatenated along the ``"randomization"``
        dimension. Potential raggedness of the ``"shot"`` dimension is handled via the
        ``"data_mask"`` data variable.

        Args:
            other: The other raw dataset.

        Returns:
            The merged data.
        """

        new_datatree = self.datatree.copy(deep=True)

        for _, other_node in other.datatree.items():
            other_ds = other_node.dataset
            matched_key = self._find_matching_key(new_datatree, other_ds)
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
    def _find_matching_key(datatree: xr.DataTree, dataset: xr.Dataset) -> str | None:
        """Find a key in the datatree whose dataset has matching ``"bit"`` coordinates."""
        for key, node in datatree.items():
            if all(_variables_equal(node.dataset, dataset, name) for name in _BIT_COORD_NAMES):
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


def _variables_equal(dataset1: xr.Dataset, dataset2: xr.Dataset, name: str) -> bool:
    """Check whether two datasets hold an equal variable under ``name``, or neither holds one."""
    variable1 = dataset1.variables.get(name)
    variable2 = dataset2.variables.get(name)
    if variable1 is None or variable2 is None:
        return variable1 is None and variable2 is None
    return variable1.equals(variable2)
