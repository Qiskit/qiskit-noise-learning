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

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import xarray as xr
from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.analysis import Fit, ZeroPostSelect
from qiskit_noise_learning.data import RawData


@dataclass(frozen=True)
class MockInstructionPattern:
    name: str


@dataclass(frozen=True)
class MockInstructionSequence:
    pattern: MockInstructionPattern
    depth: int


def make_fit(raw_data, coupling_map):
    fit = Fit(model=SimpleNamespace(gate_set=SimpleNamespace(coupling_map=coupling_map)))
    fit[RawData] = raw_data
    return fit


def make_raw_data(creg_names, measurement_map, data):
    """Build a RawData from a single data array (1 sequence)."""
    seq = MockInstructionSequence(pattern=MockInstructionPattern("p0"), depth=1)
    num_rand = data.shape[0]
    num_bits = data.shape[2]
    return RawData.from_arrays(
        creg_names=creg_names,
        measurement_map=measurement_map,
        instruction_sequences=[seq],
        data=[data],
        measurement_flips=[np.zeros((num_rand, num_bits), dtype=bool)],
        time_lbs=[np.array(["2026-01-01"] * num_rand, dtype="datetime64[us]")],
        time_ubs=[np.array(["2026-01-02"] * num_rand, dtype="datetime64[us]")],
    )


def test_zero_post_select_node_masks_shots_with_any_true_bit():
    """ZeroPostSelect node mode masks shots with any True bit in the post-selection creg."""
    data = np.array(
        [
            [
                [False, False, False, False],
                [True, False, False, False],
                [False, False, True, False],
                [False, False, False, False],
            ]
        ],
        dtype=bool,
    )
    raw = make_raw_data(
        creg_names=["meas0_ps"],
        measurement_map={"meas0_ps": np.array([0, 1, 2, 3])},
        data=data,
    )
    fit = make_fit(raw, CouplingMap.from_line(4))

    result = ZeroPostSelect(mode="node").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    np.testing.assert_array_equal(mask, [[False, True, True, False]])


def test_zero_post_select_node_no_masking_when_all_false():
    """ZeroPostSelect node mode produces no masking when all bits are False."""
    data = np.zeros((2, 3, 4), dtype=bool)
    raw = make_raw_data(
        creg_names=["meas0_ps"],
        measurement_map={"meas0_ps": np.array([0, 1, 2, 3])},
        data=data,
    )
    fit = make_fit(raw, CouplingMap.from_line(4))

    result = ZeroPostSelect(mode="node").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    np.testing.assert_array_equal(mask, np.zeros((2, 3), dtype=bool))


def test_zero_post_select_edge_masks_adjacent_pair():
    """ZeroPostSelect edge mode masks shots with True on adjacent qubits."""
    # coupling map line: 0-1, 1-2, 2-3
    # shot 0: bits 0,1 True → adjacent → mask
    # shot 1: bits 0,2 True → not adjacent → keep
    # shot 2: bits 2,3 True → adjacent → mask
    data = np.array(
        [
            [
                [True, True, False, False],
                [True, False, True, False],
                [False, False, True, True],
            ]
        ],
        dtype=bool,
    )
    raw = make_raw_data(
        creg_names=["meas0_ps"],
        measurement_map={"meas0_ps": np.array([0, 1, 2, 3])},
        data=data,
    )
    fit = make_fit(raw, CouplingMap.from_line(4))

    result = ZeroPostSelect(mode="edge").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    np.testing.assert_array_equal(mask, [[True, False, True]])


def test_zero_post_select_edge_non_adjacent_not_masked():
    """ZeroPostSelect edge mode does not mask shots with True only on non-adjacent qubits."""
    data = np.array(
        [
            [
                [True, False, True, False],
                [True, False, False, True],
            ]
        ],
        dtype=bool,
    )
    raw = make_raw_data(
        creg_names=["meas0_ps"],
        measurement_map={"meas0_ps": np.array([0, 1, 2, 3])},
        data=data,
    )
    fit = make_fit(raw, CouplingMap.from_line(4))

    result = ZeroPostSelect(mode="edge").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    np.testing.assert_array_equal(mask, [[False, False]])


def test_zero_post_select_multiple_randomizations():
    """ZeroPostSelect correctly handles multiple randomizations independently."""
    # 3 randomizations, 2 shots each, 4 bits
    # rand 0: shot 0 has True bit → mask; shot 1 all False → keep
    # rand 1: both shots all False → keep both
    # rand 2: shot 0 all False → keep; shot 1 has True bit → mask
    data = np.array(
        [
            [[True, False, False, False], [False, False, False, False]],
            [[False, False, False, False], [False, False, False, False]],
            [[False, False, False, False], [False, True, False, False]],
        ],
        dtype=bool,
    )
    raw = make_raw_data(
        creg_names=["meas0_ps"],
        measurement_map={"meas0_ps": np.array([0, 1, 2, 3])},
        data=data,
    )
    fit = make_fit(raw, CouplingMap.from_line(4))

    result = ZeroPostSelect(mode="node").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    expected = np.array(
        [
            [True, False],
            [False, False],
            [False, True],
        ]
    )
    np.testing.assert_array_equal(mask, expected)


def test_zero_post_select_preserves_existing_mask():
    """ZeroPostSelect preserves pre-existing True entries in data_mask (OR semantics)."""
    data = np.zeros((1, 3, 4), dtype=bool)
    data[0, 1, 0] = True  # shot 1 will be masked by node mode

    raw = make_raw_data(
        creg_names=["meas0_ps"],
        measurement_map={"meas0_ps": np.array([0, 1, 2, 3])},
        data=data,
    )
    # Manually set shot 2 as already masked
    ds = raw.datatree["0"].dataset
    existing_mask = np.zeros((1, 3), dtype=bool)
    existing_mask[0, 2] = True
    new_ds = ds.assign(data_mask=xr.DataArray(existing_mask, dims=["randomization", "shot"]))
    raw = RawData(xr.DataTree.from_dict({"0": new_ds}))

    fit = make_fit(raw, CouplingMap.from_line(4))

    result = ZeroPostSelect(mode="node").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    np.testing.assert_array_equal(mask, [[False, True, True]])
