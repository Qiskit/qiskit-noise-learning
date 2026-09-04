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

from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import numpy as np
import xarray as xr

from qiskit_noise_learning.analysis import AnalysisStage
from qiskit_noise_learning.data import RawData


class FlipPostSelect(AnalysisStage):
    """Apply a mask to raw data based on bit-flip failures across classical registers.

    This post-selection stage identifies groups of one or two cregs, marks the *failed bits* of each
    group, and masks shots based on the structure of the failures. What counts as a failure depends
    on the size of the group:

    * Two cregs ``(base, ps)``: a qubit failed if its bit holds the same value in both, i.e. it
      did not flip between the two measurements. The two registers must measure the same set of
      qubits, but need not measure them in the same classical bit order; their bits are paired up
      by the qubit they hold.
    * One creg ``(base,)``: bit ``j`` failed if it is True. This is the natural rule when a creg
      is expected to read out all-zeros, and coincides with the two-creg rule for a ``ps`` register
      of all ones.

    Given the failed bits, the mode determines which shots are discarded:

    * ``"node"``: Shots are discarded if at least one bit failed.
    * ``"edge"``: Shots are discarded if there exists a pair of neighbouring qubits in the
      coupling map for which both bits failed.

    Args:
        creg_identifier: A callable that, given a list of present creg names, returns an iterator
            over tuples of creg names to post-select on. Each tuple holds either one or two names,
            selecting the corresponding rule above. Defaults to pairing each creg named ``"*_ps"``
            with ``"*"`` when the latter is present, and treating it on its own otherwise.
        mode: Post-selection mode; either ``"node"`` or ``"edge"``.

    Raises:
        ValueError: If ``mode`` is not ``"node"`` or ``"edge"``.
    """

    def __init__(
        self,
        creg_identifier: Callable[[list[str]], Iterator[tuple[str, ...]]] | None = None,
        mode: Literal["node", "edge"] = "edge",
    ):
        if mode not in ("node", "edge"):
            raise ValueError(f"The mode must be 'node' or 'edge', but got {mode!r}.")
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
    def creg_identifier(self) -> Callable[[list[str]], Iterator[tuple[str, ...]]]:
        return self._creg_identifier

    def _run(self, fit):
        coupling_map = fit.model.gate_set.coupling_map

        def _dataset_masker(dataset: xr.Dataset) -> xr.Dataset:
            if "data" not in dataset:
                return dataset
            mask = dataset["data_mask"].values.copy()

            for names in self._creg_identifier(_creg_names(dataset)):
                failed, qubit_idxs = _failed_bits(names, dataset)

                if self._mode == "node":
                    mask |= failed.any(axis=-1)
                elif self._mode == "edge":
                    for i, qi in enumerate(qubit_idxs):
                        for j, qj in enumerate(qubit_idxs):
                            if j <= i:
                                continue
                            if coupling_map.graph.has_edge(qi, qj):
                                mask |= failed[:, :, i] & failed[:, :, j]

            new_data_mask = xr.DataArray(data=mask, dims=["randomization", "shot"])
            return dataset.assign(data_mask=new_data_mask)

        fit[RawData] = RawData(fit.raw_data.datatree.map_over_datasets(_dataset_masker))


def _creg_names(dataset: xr.Dataset) -> list[str]:
    """Return the dataset's creg names, in order of first appearance along the ``"bit"`` dim."""
    return list(dict.fromkeys(str(name) for name in dataset["creg_name"].values))


def _failed_bits(names: Sequence[str], dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return the failed bits of the creg group ``names``, and the qubits those bits measure.

    The returned array has shape ``(randomization, shot, bit)``, spanning only the bits of the
    group, and entry ``j`` of the returned qubit indices is the physical qubit measured into bit
    ``j`` of that array. A register's bits are located by the dataset's ``"creg_name"``
    coordinate, so the result does not depend on where along the ``"bit"`` dimension they sit.

    A two-creg group is paired up by qubit rather than by classical bit position, so the two
    registers may measure their common qubits in different orders. Both are put in ascending qubit
    order, which is the order of the returned arrays.

    Args:
        names: The one or two creg names making up the group.
        dataset: The leaf dataset holding the bits.

    Returns:
        The failed bits, and the physical qubits those bits measure.

    Raises:
        ValueError: If ``names`` does not hold one or two names, if a name matches no bit of the
            dataset, or if a two-creg group's registers do not measure the same set of qubits.
    """
    if len(names) not in (1, 2):
        raise ValueError(
            f"The creg identifier must yield tuples of one or two creg names, but got "
            f"{tuple(names)}."
        )

    bit_creg_names = dataset["creg_name"].values
    bit_qubit_idxs = dataset["qubit_idx"].values

    selections = []
    for name in names:
        selection = bit_creg_names == name
        if not selection.any():
            raise ValueError(
                f"The creg identifier yielded '{name}', but no bit of the dataset belongs to a "
                f"register of that name. The registers present are {_creg_names(dataset)}."
            )
        selections.append(selection)

    data = dataset["data"].values

    if len(names) == 1:
        (selection,) = selections
        return data[:, :, selection], bit_qubit_idxs[selection]

    base_selection, ps_selection = selections
    base_order = np.argsort(bit_qubit_idxs[base_selection], kind="stable")
    ps_order = np.argsort(bit_qubit_idxs[ps_selection], kind="stable")
    base_qubits = bit_qubit_idxs[base_selection][base_order]
    ps_qubits = bit_qubit_idxs[ps_selection][ps_order]
    if not np.array_equal(base_qubits, ps_qubits):
        raise ValueError(
            f"Cregs '{names[0]}' and '{names[1]}' must measure the same qubits, but "
            f"'{names[0]}' measures {base_qubits.tolist()} and '{names[1]}' measures "
            f"{ps_qubits.tolist()}."
        )

    base_bits = data[:, :, base_selection][:, :, base_order]
    ps_bits = data[:, :, ps_selection][:, :, ps_order]
    return base_bits == ps_bits, base_qubits


def suffix_creg_identifier(suffix: str = "ps") -> Callable[[list[str]], Iterator[tuple[str, ...]]]:
    def creg_identifier(creg_names):
        suffix_tag = f"_{suffix}"
        for name in creg_names:
            if name.endswith(suffix_tag):
                base = name[: -len(suffix_tag)]
                yield (base, name) if base in creg_names else (name,)

    return creg_identifier
