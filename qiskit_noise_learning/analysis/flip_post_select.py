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

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Literal

import numpy as np
import xarray as xr

from qiskit_noise_learning.analysis import AnalysisStage
from qiskit_noise_learning.data import RawData


class FlipPostSelect(AnalysisStage):
    """Apply a mask to raw data based on bit-flip failures across classical registers.

    This post-selection stage identifies groups of one or two cregs, marks the *failed bits* of each
    group, and masks shots based on the structure of the failures:

    * Two cregs ``(base, ps)``: bit ``j`` failed if its outcome is the same in both, i.e. it did
      not flip between the two measurements.
    * One creg ``(base,)``: bit ``j`` failed if its outcome is True. This is the natural rule when a
      creg is expected to read out all-zeros, and coincides with the two-creg rule for a ``ps``
      register of all ones.

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
        apply_measurement_flips: Whether or not to include measurement flips in the comparison.
            Defaults to ``False``, i.e. comparing the physical measurement outcomes.

    Raises:
        ValueError: If ``mode`` is not ``"node"`` or ``"edge"``.
    """

    def __init__(
        self,
        creg_identifier: Callable[[list[str]], Iterator[tuple[str, ...]]] | None = None,
        mode: Literal["node", "edge"] = "edge",
        apply_measurement_flips: bool = False,
    ):
        if mode not in ("node", "edge"):
            raise ValueError(f"The mode must be 'node' or 'edge', but got {mode!r}.")
        self._creg_identifier = creg_identifier or suffix_creg_identifier()
        self._mode = mode
        self._apply_measurement_flips = apply_measurement_flips

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

    @property
    def apply_measurement_flips(self) -> bool:
        return self._apply_measurement_flips

    def _run(self, fit):
        coupling_map = fit.model.gate_set.coupling_map

        def _dataset_masker(dataset: xr.Dataset) -> xr.Dataset:
            if "data" not in dataset:
                return dataset
            outcomes = dataset["data"].values
            if self._apply_measurement_flips:
                outcomes = outcomes ^ dataset["measurement_flips"].values[:, np.newaxis, :]
            mask = dataset["data_mask"].values.copy()
            boundaries = dataset.attrs["creg_bit_boundaries"]
            creg_names = dataset.attrs["creg_names"]
            clbit_qubit_idxs = dataset.attrs["clbit_qubit_idxs"]

            for names in self._creg_identifier(creg_names):
                failed, qubit_idxs = _failed_bits(names, outcomes, boundaries, clbit_qubit_idxs)

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


def _failed_bits(
    names: Sequence[str],
    outcomes: np.ndarray,
    boundaries: Mapping[str, tuple[int, int]],
    clbit_qubit_idxs: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the failed bits of the creg group ``names``, and the qubits those bits measure.

    The ``outcomes`` argument is the measured bits with their measurement twirling flips already
    undone.

    The returned array has shape ``(randomization, shot, bit)``, and entry ``j`` of the returned
    qubit indices is the physical qubit measured into bit ``j``. Note that the one-creg branch
    returns a view into ``outcomes`` rather than a fresh array.
    """
    if len(names) == 1:
        (name,) = names
        start, end = boundaries[name]
        return outcomes[:, :, start:end], clbit_qubit_idxs[name]

    if len(names) == 2:
        base_name, ps_name = names
        base_qubits = clbit_qubit_idxs[base_name]
        ps_qubits = clbit_qubit_idxs[ps_name]
        if not np.array_equal(base_qubits, ps_qubits):
            raise ValueError(
                f"Cregs '{base_name}' and '{ps_name}' must measure the same qubits in "
                "the same classical bit order."
            )

        base_start, base_end = boundaries[base_name]
        ps_start, ps_end = boundaries[ps_name]

        base_bits = outcomes[:, :, base_start:base_end]
        ps_bits = outcomes[:, :, ps_start:ps_end]

        return base_bits == ps_bits, base_qubits

    raise ValueError(
        f"The creg identifier must yield tuples of one or two creg names, but got {tuple(names)}."
    )


def suffix_creg_identifier(
    suffix: str = "_ps",
) -> Callable[[list[str]], Iterator[tuple[str, ...]]]:
    """Return a creg identifier that pairs each creg named ``"*<suffix>"`` with ``"*"``.

    The returned callable is suitable as the ``creg_identifier`` argument of
    :class:`~.FlipPostSelect`, and is the default that class uses. Given the names of the cregs
    present in a dataset, it yields one tuple for every name ending in ``suffix``: the pair
    ``(base, name)`` when the corresponding ``base`` name is also present, and the singleton
    ``(name,)`` when it is not. Names that do not end in ``suffix`` are not post-selected on.

    Args:
        suffix: The complete creg name suffix marking a post-selection register, including any
            separator. The default pairs ``"meas_ps"`` with ``"meas"``.

    Returns:
        A callable mapping a list of creg names to an iterator over groups of creg names.

    Raises:
        ValueError: If *suffix* is empty, which would pair every creg with itself.
    """
    if not suffix:
        raise ValueError("The suffix must not be empty.")

    def creg_identifier(creg_names):
        for name in creg_names:
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                yield (base, name) if base in creg_names else (name,)

    return creg_identifier
