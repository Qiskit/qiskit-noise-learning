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

import numpy as np

from qiskit_noise_learning.data import RawData


def test_from_arrays(make_instruction_sequence):
    """Test constructing RawData from arrays."""
    seq = make_instruction_sequence(name="CZ", fragment_depth=1)
    num_randomizations = 3
    num_shots = 10
    num_bits = 2

    raw = RawData.from_arrays(
        creg_names=["meas0"],
        measurement_map={"meas0": np.array([0, 1])},
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
    assert ds.attrs["creg_names"] == ["meas0"]
    assert ds.attrs["creg_bit_boundaries"] == {"meas0": (0, 2)}


def test_filter_time(make_instruction_sequence):
    """Test that filter_time keeps only randomizations within the time window."""
    seq = make_instruction_sequence(name="CZ", fragment_depth=1)
    num_shots = 5
    num_bits = 2
    t_lbs = np.array(["2026-01-01", "2026-01-03", "2026-01-05"], dtype="datetime64[us]")
    t_ubs = np.array(["2026-01-02", "2026-01-04", "2026-01-06"], dtype="datetime64[us]")

    raw = RawData.from_arrays(
        creg_names=["meas0"],
        measurement_map={"meas0": np.array([0, 1])},
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
