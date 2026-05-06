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

from qiskit_noise_learning.data import ModelData


def test_from_arrays():
    """Test constructing ModelData from arrays."""
    md = ModelData.from_arrays(
        parameter_indices=["r0", "r1"],
        parameter_values=np.array([0.1, 0.2]),
        covariance=np.eye(2) * 0.01,
        time_lbs=np.array(["2026-01-01", "2026-01-01"], dtype="datetime64[us]"),
        time_ubs=np.array(["2026-01-02", "2026-01-02"], dtype="datetime64[us]"),
    )
    ds = md.dataset
    assert ds["parameter_values"].sel(parameter="r0").item() == 0.1
    assert ds["parameter_values"].sel(parameter="r1").item() == 0.2
    assert ds["covariance"].shape == (2, 2)


def test_filter_time():
    """Test that filter_time keeps only parameters within the time window."""
    md = ModelData.from_arrays(
        parameter_indices=["r0", "r1"],
        parameter_values=np.array([0.1, 0.2]),
        covariance=np.eye(2) * 0.01,
        time_lbs=np.array(["2026-01-01", "2026-01-05"], dtype="datetime64[us]"),
        time_ubs=np.array(["2026-01-02", "2026-01-06"], dtype="datetime64[us]"),
    )
    filtered = md.filter_time(lb=np.datetime64("2026-01-04"), ub=np.datetime64("2026-01-07"))
    vals = filtered.dataset["parameter_values"].values
    assert np.isnan(vals[0])
    assert vals[1] == 0.2
