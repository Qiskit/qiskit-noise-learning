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
from qiskit_noise_learning.sequences import InstructionPattern, PathPattern


class ComputeObservables(AnalysisStage):
    """Compute observable data from raw data.

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
        if any(
            fidelity_index.gate.meas_idxs
            for path_pattern in [p.pattern for p in fit.paths]
            for fidelity_index in chain(
                path_pattern.start_fragment,
                path_pattern.repeatable_fragment,
                path_pattern.end_fragment[:-1],
            )
        ):
            raise NotImplementedError("Encountered a path with a midcircuit measurement.")

        raw_data = fit.raw_data

        # map from unique path patterns to depths
        path_pattern_depths: dict[PathPattern, list[int]] = dict()
        for path in fit.paths:
            path_pattern_depths.setdefault(path.pattern, set()).add(path.depth)

        # mapping from path patterns into data
        # nested mapping: path pattern -> bit_count_str -> depth ->
        # {"array_indices": list[int], "signs": list[int]}
        path_pattern_to_data = dict()

        # mapping from path patterns to the instruction sequences that measure them, and signs
        # just used in the process of constructing the above
        unique_instruction_patterns: list[InstructionPattern] = []
        instruction_pattern_path_signs: list[dict[PathPattern, tuple[bool, bool]]] = []

        for bit_count_str, datasubtree in raw_data.datatree.items():
            for array_idx, (ip, depth) in enumerate(
                zip(
                    datasubtree.dataset["instruction_pattern"].data,
                    datasubtree.dataset["depth"].data,
                )
            ):
                # get the path patterns traversed by this path and the sign information,
                # constructing if not yet encountered
                ip_path_signs = None
                if ip not in unique_instruction_patterns:
                    unique_instruction_patterns.append(ip)
                    ip_path_signs = dict()
                    for path_pattern in path_pattern_depths:
                        if path_pattern.is_traversed_by(ip):
                            ip_path_signs[path_pattern] = path_pattern.sign_flips(ip)
                    instruction_pattern_path_signs.append(ip_path_signs)
                else:
                    ip_path_signs = instruction_pattern_path_signs[
                        unique_instruction_patterns.index(ip)
                    ]

                for path_pattern, signs in ip_path_signs.items():
                    # if depth not relevant, continue
                    if depth not in path_pattern_depths[path_pattern]:
                        continue

                    # get the dictionary for this path pattern, bit count, and depth
                    path_pattern_bit_count_depth_dict = (
                        path_pattern_to_data.setdefault(path_pattern, dict())
                        .setdefault(bit_count_str, dict())
                        .setdefault(depth, {"array_indices": [], "signs": []})
                    )
                    path_pattern_bit_count_depth_dict["array_indices"].append(array_idx)
                    path_pattern_bit_count_depth_dict["signs"] = (-1) ** (
                        signs[0] + depth * signs[1]
                    )

        # determine dimension sizes for observable data
        observable_count = 0
        max_num_randomizations = 0
        for path_pattern, datatree_mapping in path_pattern_to_data.items():
            for bit_count_str, depth_mapping in datatree_mapping.items():
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
        path_pattern_coord = np.empty(observable_count, dtype=object)
        depth_coord = np.empty(observable_count, dtype=int)

        observable_idx = 0
        for path_pattern, datatree_mapping in path_pattern_to_data.items():
            bit_mask = path_pattern.end_fragment[-1].mask
            for bit_count_str, depth_mapping in datatree_mapping.items():
                raw_dataset = raw_data.datatree[bit_count_str].dataset
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
                    path_pattern_coord[observable_idx] = path_pattern
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
                    "path_pattern": (("observable",), path_pattern_coord),
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
        bit_mask: A boolean mask on the ``(bit,)`` dimension.
        signs: Signs for the computed observables along the ``(randomization,)`` dimension.

    Returns:
        Expectation values with dimension ``(randomization,)``.
    """
    corrected_bits = (bits ^ flips[:, np.newaxis, :])[..., bit_mask]
    broadcasted_shot_mask = np.broadcast_to(shot_mask[:, :, np.newaxis], corrected_bits.shape)
    masked_arr = np.ma.array(corrected_bits, mask=broadcasted_shot_mask)
    per_sample = 1 - 2 * np.mod(np.sum(masked_arr, axis=-1), 2)
    return signs * per_sample.mean(axis=-1)


def observable_bit_mask(path_pattern: PathPattern, depth: int) -> np.ndarray[bool]:
    """Return the observable bit mask corresponding to the path pattern at the given depth."""
    mask_array = np.array([], dtype=bool)

    start_masks = [x.mask for x in path_pattern.start_fragment]
    for mask_fragment in start_masks:
        mask_array = np.append(mask_array, mask_fragment)

    repeatable_masks = [x.mask for x in path_pattern.repeatable_fragment]
    repeatable_mask = np.array([], dtype=bool)
    for mask_fragment in repeatable_masks:
        repeatable_mask = np.append(repeatable_mask, mask_fragment)
    mask_array = np.append(
        mask_array, np.repeat(np.array([repeatable_mask]), depth, axis=0).flatten()
    )

    end_masks = [x.mask for x in path_pattern.end_fragment]
    for mask_fragment in end_masks:
        mask_array = np.append(mask_array, mask_fragment)

    return mask_array
