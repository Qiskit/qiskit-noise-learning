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

from dataclasses import replace

import numpy as np
import pytest

from qiskit_noise_learning.data import MeasurementRegister, RawData


def test_from_arrays(make_instruction_sequence):
    """Test constructing RawData from arrays."""
    seq = make_instruction_sequence(name="CZ", fragment_depth=1)
    num_randomizations = 3
    num_shots = 10
    num_bits = 2

    raw = RawData.from_arrays(
        registers=[MeasurementRegister("meas0", (0, 1), measuring_gate_idx=0)],
        instruction_sequences=[seq],
        data=[np.zeros((num_randomizations, num_shots, num_bits), dtype=bool)],
        measurement_flips=[np.zeros((num_randomizations, num_bits), dtype=bool)],
        time_lbs=[np.array(["2026-01-01"] * num_randomizations, dtype="datetime64[us]")],
        time_ubs=[np.array(["2026-01-02"] * num_randomizations, dtype="datetime64[us]")],
    )
    dt = raw.datatree
    assert "0" in dt
    ds = dt["0"].dataset
    assert ds["data"].shape == (num_randomizations, num_shots, num_bits)
    assert ds["measurement_flips"].shape == (num_randomizations, num_bits)
    np.testing.assert_array_equal(ds["creg_name"].values, ["meas0", "meas0"])
    np.testing.assert_array_equal(ds["qubit_idx"].values, [0, 1])
    np.testing.assert_array_equal(ds["measuring_gate_idx"].values, [0, 0])


def test_filter_time(make_instruction_sequence):
    """Test that filter_time keeps only randomizations within the time window."""
    seq = make_instruction_sequence(name="CZ", fragment_depth=1)
    num_shots = 5
    num_bits = 2
    t_lbs = np.array(["2026-01-01", "2026-01-03", "2026-01-05"], dtype="datetime64[us]")
    t_ubs = np.array(["2026-01-02", "2026-01-04", "2026-01-06"], dtype="datetime64[us]")

    raw = RawData.from_arrays(
        registers=[MeasurementRegister("meas0", (0, 1), measuring_gate_idx=0)],
        instruction_sequences=[seq],
        data=[np.ones((3, num_shots, num_bits), dtype=bool)],
        measurement_flips=[np.zeros((3, num_bits), dtype=bool)],
        time_lbs=[t_lbs],
        time_ubs=[t_ubs],
    )
    filtered = raw.filter_time(lb=np.datetime64("2026-01-03"), ub=np.datetime64("2026-01-04"))
    ds = filtered.datatree["0"].dataset
    time_lbs_out = ds["time_lbs"].values
    assert np.isnat(time_lbs_out[0])
    assert not np.isnat(time_lbs_out[1])
    assert np.isnat(time_lbs_out[2])


MEAS0 = MeasurementRegister("meas0", (0, 1), measuring_gate_idx=0)
FLAG_PS = MeasurementRegister("flag_ps", (2, 3), measuring_gate_idx=-1)


def _raw_data(registers, seq):
    """Build a single-leaf ``RawData`` over ``registers`` holding one randomization of zeros."""
    num_bits = sum(register.num_bits for register in registers)
    return RawData.from_arrays(
        registers=registers,
        instruction_sequences=[seq],
        data=[np.zeros((1, 4, num_bits), dtype=bool)],
        measurement_flips=[np.zeros((1, num_bits), dtype=bool)],
        time_lbs=[np.array(["2026-01-01"], dtype="datetime64[us]")],
        time_ubs=[np.array(["2026-01-02"], dtype="datetime64[us]")],
    )


def test_merge_matching_bit_coords_gives_one_leaf(make_instruction_sequence):
    """Data sharing a bit layout concatenates into a single leaf.

    This is the control for the mismatch cases below: without it, a bug that split *every* merge
    into two leaves would satisfy them all.
    """
    seq = make_instruction_sequence(name="CZ", fragment_depth=1)
    registers = [MEAS0, FLAG_PS]

    merged = _raw_data(registers, seq).merge(_raw_data(registers, seq))

    assert list(merged.datatree) == ["0"]
    assert merged.datatree["0"].dataset.sizes["randomization"] == 2


@pytest.mark.parametrize(
    "other_registers",
    [
        [MEAS0, replace(FLAG_PS, name="flag2_ps")],
        [replace(MEAS0, qubit_idxs=(1, 0)), FLAG_PS],
        [MEAS0, replace(FLAG_PS, measuring_gate_idx=1)],
        [FLAG_PS, MEAS0],
    ],
    ids=["creg_name", "qubit_idx", "measuring_gate_idx", "register_order"],
)
def test_merge_mismatched_bit_coords_gives_two_leaves(other_registers, make_instruction_sequence):
    """Data whose bit layout differs in any single coordinate lands in its own leaf.

    A shared leaf would silently pair up bits that mean different things, so each of the three
    coordinates on its own, and a permutation of the registers along the bit axis, must split.
    """
    seq = make_instruction_sequence(name="CZ", fragment_depth=1)

    merged = _raw_data([MEAS0, FLAG_PS], seq).merge(_raw_data(other_registers, seq))

    assert list(merged.datatree) == ["0", "1"]
    for node in merged.datatree.values():
        assert node.dataset.sizes["randomization"] == 1


def test_from_arrays_duplicate_register_names_raises(make_instruction_sequence):
    """Two registers of the same name cannot be told apart by the ``creg_name`` coordinate."""
    seq = make_instruction_sequence(name="CZ", fragment_depth=1)

    with pytest.raises(ValueError, match="register names must be unique"):
        _raw_data([MEAS0, replace(FLAG_PS, name="meas0")], seq)


def test_from_arrays_short_per_sequence_list_raises(make_instruction_sequence):
    """A per-sequence list shorter than the others raises rather than being silently truncated."""
    seq = make_instruction_sequence(name="CZ", fragment_depth=1)

    with pytest.raises(ValueError, match="must all have the same length"):
        RawData.from_arrays(
            registers=[MEAS0],
            instruction_sequences=[seq, seq],
            data=[np.zeros((1, 4, 2), dtype=bool)] * 2,
            measurement_flips=[np.zeros((1, 2), dtype=bool)] * 2,
            time_lbs=[np.array(["2026-01-01"], dtype="datetime64[us]")],
            time_ubs=[np.array(["2026-01-02"], dtype="datetime64[us]")] * 2,
        )


@pytest.mark.parametrize("name", ["data", "measurement_flips"])
def test_from_arrays_wrong_bit_size_raises(name, make_instruction_sequence):
    """An array whose ``"bit"`` dimension disagrees with the registers' total bit count raises."""
    seq = make_instruction_sequence(name="CZ", fragment_depth=1)
    arrays = {
        "data": [np.zeros((1, 4, 2), dtype=bool)],
        "measurement_flips": [np.zeros((1, 2), dtype=bool)],
    }
    arrays[name] = [array[..., :1] for array in arrays[name]]

    with pytest.raises(ValueError, match=f"an entry of {name} has 1 along its 'bit' dimension"):
        RawData.from_arrays(
            registers=[MEAS0],
            instruction_sequences=[seq],
            time_lbs=[np.array(["2026-01-01"], dtype="datetime64[us]")],
            time_ubs=[np.array(["2026-01-02"], dtype="datetime64[us]")],
            **arrays,
        )
