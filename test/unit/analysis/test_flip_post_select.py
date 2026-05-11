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
import pytest
import xarray as xr
from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.analysis import Fit, FlipPostSelect
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


def test_flip_post_select_node_masks_unchanged_bits():
    """FlipPostSelect node mode masks shots where any bit is unchanged between cregs."""
    # data layout: [meas0 (4 bits), meas0_ps (4 bits)]
    # shot 0: all bits flipped → keep
    # shot 1: bit 2 same (both False) → mask
    # shot 2: all bits flipped → keep
    data = np.array(
        [
            [
                [False, False, False, False, True, True, True, True],
                [False, False, False, False, True, True, False, True],
                [True, False, True, False, False, True, False, True],
            ]
        ],
        dtype=bool,
    )
    raw = make_raw_data(
        creg_names=["meas0", "meas0_ps"],
        measurement_map={
            "meas0": np.array([0, 1, 2, 3]),
            "meas0_ps": np.array([0, 1, 2, 3]),
        },
        data=data,
    )
    fit = make_fit(raw, CouplingMap.from_line(4))

    result = FlipPostSelect.from_suffix(mode="node").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    np.testing.assert_array_equal(mask, [[False, True, False]])


def test_flip_post_select_node_no_masking_when_all_flipped():
    """FlipPostSelect node mode produces no masking when all bits flip."""
    data = np.array(
        [
            [
                [False, False, False, False, True, True, True, True],
                [True, True, True, True, False, False, False, False],
            ]
        ],
        dtype=bool,
    )
    raw = make_raw_data(
        creg_names=["meas0", "meas0_ps"],
        measurement_map={
            "meas0": np.array([0, 1, 2, 3]),
            "meas0_ps": np.array([0, 1, 2, 3]),
        },
        data=data,
    )
    fit = make_fit(raw, CouplingMap.from_line(4))

    result = FlipPostSelect.from_suffix(mode="node").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    np.testing.assert_array_equal(mask, np.zeros((1, 2), dtype=bool))


def test_flip_post_select_edge_masks_adjacent_pair_failures():
    """FlipPostSelect edge mode masks shots with adjacent qubits both failing to flip."""
    # coupling map line: 0-1, 1-2, 2-3
    # data layout: [meas0 (4 bits), meas0_ps (4 bits)]
    # shot 0: bits 0,1 fail to flip (both same) → adjacent → mask
    # shot 1: bits 0,2 fail to flip → not adjacent → keep
    # shot 2: only bit 1 fails → no pair → keep
    data = np.array(
        [
            [
                [False, False, False, False, False, False, True, True],
                [False, False, False, False, False, True, False, True],
                [False, False, False, False, True, False, True, True],
            ]
        ],
        dtype=bool,
    )
    raw = make_raw_data(
        creg_names=["meas0", "meas0_ps"],
        measurement_map={
            "meas0": np.array([0, 1, 2, 3]),
            "meas0_ps": np.array([0, 1, 2, 3]),
        },
        data=data,
    )
    fit = make_fit(raw, CouplingMap.from_line(4))

    result = FlipPostSelect.from_suffix(mode="edge").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    np.testing.assert_array_equal(mask, [[True, False, False]])


def test_flip_post_select_mismatched_qubits_raises():
    """FlipPostSelect raises ValueError when cregs measure different qubits."""
    data = np.zeros((1, 2, 4), dtype=bool)
    raw = make_raw_data(
        creg_names=["meas0", "meas0_ps"],
        measurement_map={
            "meas0": np.array([0, 1]),
            "meas0_ps": np.array([2, 3]),
        },
        data=data,
    )
    fit = make_fit(raw, CouplingMap.from_line(4))

    with pytest.raises(ValueError, match="do not measure the same qubits"):
        FlipPostSelect.from_suffix(mode="node").run(fit)


def test_flip_post_select_multiple_randomizations():
    """FlipPostSelect correctly handles multiple randomizations independently."""
    # 3 randomizations, 2 shots each
    # data layout: [meas0 (4 bits), meas0_ps (4 bits)]
    # rand 0: shot 0 bit 0 fails to flip → mask; shot 1 all flip → keep
    # rand 1: both shots all flip → keep both
    # rand 2: shot 0 all flip → keep; shot 1 bit 3 fails to flip → mask
    data = np.array(
        [
            [
                [False, False, False, False, False, True, True, True],
                [False, False, False, False, True, True, True, True],
            ],
            [
                [False, False, False, False, True, True, True, True],
                [True, True, True, True, False, False, False, False],
            ],
            [
                [False, False, False, False, True, True, True, True],
                [False, False, False, True, True, True, True, True],
            ],
        ],
        dtype=bool,
    )
    raw = make_raw_data(
        creg_names=["meas0", "meas0_ps"],
        measurement_map={
            "meas0": np.array([0, 1, 2, 3]),
            "meas0_ps": np.array([0, 1, 2, 3]),
        },
        data=data,
    )
    fit = make_fit(raw, CouplingMap.from_line(4))

    result = FlipPostSelect.from_suffix(mode="node").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    expected = np.array(
        [
            [True, False],
            [False, False],
            [False, True],
        ]
    )
    np.testing.assert_array_equal(mask, expected)


def test_flip_post_select_from_list_targets_specific_pair():
    """FlipPostSelect.from_list targets only the specified creg pair."""
    # Two pairs: ("a","b") and ("c","d"), only target ("a","b")
    # data layout: [a (2 bits), b (2 bits), c (2 bits), d (2 bits)]
    # shot 0: a and b bits same (fail to flip) → mask
    # shot 1: c and d bits same but a/b flipped → keep (c/d not targeted)
    data = np.array(
        [
            [
                [False, False, False, False, True, True, True, True],
                [False, False, True, True, False, False, False, False],
            ]
        ],
        dtype=bool,
    )
    raw = make_raw_data(
        creg_names=["a", "b", "c", "d"],
        measurement_map={
            "a": np.array([0, 1]),
            "b": np.array([0, 1]),
            "c": np.array([2, 3]),
            "d": np.array([2, 3]),
        },
        data=data,
    )
    fit = make_fit(raw, CouplingMap.from_line(4))

    result = FlipPostSelect.from_list([("a", "b")], mode="node").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    np.testing.assert_array_equal(mask, [[True, False]])


def test_flip_post_select_preserves_existing_mask():
    """FlipPostSelect preserves pre-existing True entries in data_mask."""
    # All bits flip → no new masking
    data = np.array(
        [
            [
                [False, False, False, False, True, True, True, True],
                [False, False, False, False, True, True, True, True],
                [False, False, False, False, True, True, True, True],
            ]
        ],
        dtype=bool,
    )
    raw = make_raw_data(
        creg_names=["meas0", "meas0_ps"],
        measurement_map={
            "meas0": np.array([0, 1, 2, 3]),
            "meas0_ps": np.array([0, 1, 2, 3]),
        },
        data=data,
    )
    # Set shot 1 as already masked
    ds = raw.datatree["0"].dataset
    existing_mask = np.zeros((1, 3), dtype=bool)
    existing_mask[0, 1] = True
    new_ds = ds.assign(data_mask=xr.DataArray(existing_mask, dims=["randomization", "shot"]))
    raw = RawData(xr.DataTree.from_dict({"0": new_ds}))

    fit = make_fit(raw, CouplingMap.from_line(4))

    result = FlipPostSelect.from_suffix(mode="node").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    np.testing.assert_array_equal(mask, [[False, True, False]])
