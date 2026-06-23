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

from itertools import chain

import numpy as np
import xarray as xr

from qiskit_noise_learning.analysis import AnalysisStage
from qiskit_noise_learning.data import ObservableData, RawData
from qiskit_noise_learning.sequences import InstructionSequence, Path


class ComputeObservables(AnalysisStage):
    """Compute observable data from raw data.

    This analysis stage utilizes existing relations between paths and instruction sequences in the
    :class:`Fit` to determine how to compute observables. If relations are not specified, they are
    greedily constructed by exhaustively comparing paths in ``fit.paths`` and the instruction
    sequences in ``fit.raw_data``. Note that in this latter case, ``fit.paths`` is not updated and
    will therefore remain ``None``.

    .. note:
        This analysis stage does not currently support midcircuit measurements.
    """

    @property
    def input_level(self):
        return RawData

    @property
    def output_level(self):
        return ObservableData

    def _run(self, fit):
        # note that in_bit_indices could technically be empty with a measurement
        if any(
            fidelity_index.in_bit_indices
            for path in fit.paths
            for fidelity_index in chain(
                path.start_fragment,
                path.repeatable_fragment,
                path.end_fragment[:-1],
            )
        ):
            raise NotImplementedError("Encountered a path with a midcircuit measurement.")

        raw_data = fit.raw_data

        # map from unique unbound paths to accepted depths.
        # None means accept any depth (for unbound paths in fit.paths).
        unbound_path_depths: dict[Path, set[int] | None] = dict()
        for path in fit.paths:
            if path.is_unbound:
                unbound_path_depths[path] = None
            else:
                # only add something if path wasn't unbound in fit.paths
                existing = unbound_path_depths.get(path.without_depth())
                if existing is not None or path.without_depth() not in unbound_path_depths:
                    unbound_path_depths.setdefault(path.without_depth(), set()).add(path.depth)

        # mapping from unbound paths into data
        # nested mapping: unbound_path -> dt_key -> depth ->
        # {"array_indices": list[int], "signs": list[int]}
        unbound_path_to_data = dict()

        if fit.relations is not None and fit.instruction_sequences is not None:
            # Strategy A: use precomputed relations
            unique_uis_list: list[InstructionSequence] = []
            uis_path_signs_list: list[dict[Path, tuple[bool, bool]]] = []
            for path_idx, seq_idx in fit.relations:
                path = fit.paths[path_idx]
                unbound_path = path.without_depth()
                seq = fit.instruction_sequences[seq_idx]
                uis = seq.without_depth()

                if uis not in unique_uis_list:
                    unique_uis_list.append(uis)
                    uis_path_signs_list.append({})

                uis_paths = uis_path_signs_list[unique_uis_list.index(uis)]
                if unbound_path not in uis_paths:
                    uis_paths[unbound_path] = unbound_path.fragment_sign_flips(seq)

            for dt_key, datasubtree in raw_data.datatree.items():
                for array_idx, (uis, depth) in enumerate(
                    zip(
                        datasubtree.dataset["unbound_instruction_sequence"].data,
                        datasubtree.dataset["depth"].data,
                    )
                ):
                    if uis not in unique_uis_list:
                        continue

                    for unbound_path, signs in uis_path_signs_list[
                        unique_uis_list.index(uis)
                    ].items():
                        accepted = unbound_path_depths[unbound_path]
                        if accepted is not None and depth not in accepted:
                            continue

                        unbound_path_dt_depth_dict = (
                            unbound_path_to_data.setdefault(unbound_path, dict())
                            .setdefault(dt_key, dict())
                            .setdefault(depth, {"array_indices": [], "signs": []})
                        )
                        unbound_path_dt_depth_dict["array_indices"].append(array_idx)
                        unbound_path_dt_depth_dict["signs"].append(
                            (-1) ** (signs[0] + depth * signs[1])
                        )
        else:
            # Strategy B: greedy discovery fallback
            unique_unbound_instruction_sequences: list[InstructionSequence] = []
            unbound_instruction_sequence_path_signs: list[dict[Path, tuple[bool, bool]]] = []

            for dt_key, datasubtree in raw_data.datatree.items():
                for array_idx, (uis, depth) in enumerate(
                    zip(
                        datasubtree.dataset["unbound_instruction_sequence"].data,
                        datasubtree.dataset["depth"].data,
                    )
                ):
                    uis_path_signs = None
                    if uis not in unique_unbound_instruction_sequences:
                        unique_unbound_instruction_sequences.append(uis)
                        uis_path_signs = dict()
                        for unbound_path in unbound_path_depths:
                            if unbound_path.is_traversed_by(uis):
                                uis_path_signs[unbound_path] = unbound_path.fragment_sign_flips(uis)
                        unbound_instruction_sequence_path_signs.append(uis_path_signs)
                    else:
                        uis_path_signs = unbound_instruction_sequence_path_signs[
                            unique_unbound_instruction_sequences.index(uis)
                        ]

                    for unbound_path, signs in uis_path_signs.items():
                        accepted = unbound_path_depths[unbound_path]
                        if accepted is not None and depth not in accepted:
                            continue

                        unbound_path_dt_depth_dict = (
                            unbound_path_to_data.setdefault(unbound_path, dict())
                            .setdefault(dt_key, dict())
                            .setdefault(depth, {"array_indices": [], "signs": []})
                        )
                        unbound_path_dt_depth_dict["array_indices"].append(array_idx)
                        unbound_path_dt_depth_dict["signs"].append(
                            (-1) ** (signs[0] + depth * signs[1])
                        )

        # determine dimension sizes for observable data
        observable_count = 0
        max_num_randomizations = 0
        for unbound_path, datatree_mapping in unbound_path_to_data.items():
            for dt_key, depth_mapping in datatree_mapping.items():
                for depth, dataset_mapping in depth_mapping.items():
                    observable_count += 1
                    max_num_randomizations = max(
                        max_num_randomizations, len(dataset_mapping["array_indices"])
                    )

        # initialize arrays
        observable_array = np.empty((observable_count, max_num_randomizations), dtype=float)
        observable_array[:] = np.nan
        time_lbs = np.empty((observable_count, max_num_randomizations), dtype="datetime64[us]")
        time_lbs[:] = np.datetime64("NaT")
        time_ubs = np.empty((observable_count, max_num_randomizations), dtype="datetime64[us]")
        time_ubs[:] = np.datetime64("NaT")
        unbound_path_coord = np.empty(observable_count, dtype=object)
        depth_coord = np.empty(observable_count, dtype=int)

        observable_idx = 0
        for unbound_path, datatree_mapping in unbound_path_to_data.items():
            bit_mask = unbound_path.end_fragment[-1].mask
            for dt_key, depth_mapping in datatree_mapping.items():
                raw_dataset = raw_data.datatree[dt_key].dataset
                for depth, dataset_mapping in depth_mapping.items():
                    randomization_mask = np.array(dataset_mapping["array_indices"])

                    new_observables = compute_expectation_value(
                        bits=raw_dataset["data"].data[randomization_mask],
                        flips=raw_dataset["measurement_flips"].data[randomization_mask],
                        shot_mask=raw_dataset["data_mask"].data[randomization_mask],
                        bit_mask=bit_mask,
                        signs=np.array(dataset_mapping["signs"]),
                    )
                    new_time_lbs = raw_dataset["time_lbs"].data[randomization_mask]
                    new_time_ubs = raw_dataset["time_ubs"].data[randomization_mask]

                    observable_array[observable_idx, 0 : len(new_observables)] = new_observables
                    time_lbs[observable_idx, 0 : len(new_observables)] = new_time_lbs
                    time_ubs[observable_idx, 0 : len(new_observables)] = new_time_ubs
                    unbound_path_coord[observable_idx] = unbound_path
                    depth_coord[observable_idx] = depth
                    observable_idx += 1

        fit[ObservableData] = ObservableData(
            dataset=xr.Dataset(
                data_vars={
                    "observables": xr.DataArray(
                        data=observable_array, dims=["observable", "randomization"]
                    ),
                    "time_lbs": xr.DataArray(data=time_lbs, dims=["observable", "randomization"]),
                    "time_ubs": xr.DataArray(data=time_ubs, dims=["observable", "randomization"]),
                },
                coords={
                    "unbound_path": (("observable",), unbound_path_coord),
                    "depth": (("observable",), depth_coord),
                },
            )
        )


def compute_expectation_value(
    bits: np.ndarray[np.bool_],
    flips: np.ndarray[np.bool_],
    shot_mask: np.ndarray[np.bool_],
    bit_mask: np.ndarray[np.bool_],
    signs: np.ndarray[int],
) -> np.ndarray[np.float64]:
    """Compute expectation value from given data.

    Args:
        bits: A record of the measured bits, with dimensions ``(randomization, shots, bits)``.
        flips: Specification of required flips on bits ``(randomization, bits)``.
        shot_mask: A boolean mask on the ``(randomization, shots)`` dimensions.
        bit_mask: A boolean mask on the ``(bit,)`` dimension. Note that the dimension is first
            truncated to ``0:len(bit_mask)`` under the assumption that the expectation-value
            relevant bits are at the beginning of the dimension.
        signs: Signs for the computed observables along the ``(randomization,)`` dimension.

    Returns:
        Expectation values with dimension ``(randomization,)``.
    """
    corrected_bits = (bits ^ flips[:, np.newaxis, :])[..., : len(bit_mask)][..., bit_mask]
    broadcasted_shot_mask = np.broadcast_to(shot_mask[:, :, np.newaxis], corrected_bits.shape)
    masked_arr = np.ma.array(corrected_bits, mask=broadcasted_shot_mask)
    per_sample = 1 - 2 * np.mod(np.sum(masked_arr, axis=-1), 2)
    return signs * per_sample.mean(axis=-1)


def observable_bit_mask(unbound_path: Path, depth: int) -> np.ndarray[bool]:
    """Return the observable bit mask corresponding to the unbound path at the given depth."""
    mask_array = np.array([], dtype=bool)

    start_masks = [x.mask for x in unbound_path.start_fragment]
    for mask_fragment in start_masks:
        mask_array = np.append(mask_array, mask_fragment)

    repeatable_masks = [x.mask for x in unbound_path.repeatable_fragment]
    repeatable_mask = np.array([], dtype=bool)
    for mask_fragment in repeatable_masks:
        repeatable_mask = np.append(repeatable_mask, mask_fragment)
    mask_array = np.append(
        mask_array, np.repeat(np.array([repeatable_mask]), depth, axis=0).flatten()
    )

    end_masks = [x.mask for x in unbound_path.end_fragment]
    for mask_fragment in end_masks:
        mask_array = np.append(mask_array, mask_fragment)

    return mask_array
