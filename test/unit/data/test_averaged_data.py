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

from qiskit_noise_learning.data import AveragedData


def test_from_arrays(make_cz_path):
    """Test constructing AveragedData from arrays."""
    p = make_cz_path("IX")
    avg = AveragedData.from_arrays(
        unbound_paths=[p],
        fragment_depths=[-1],
        observables=np.array([0.8]),
        std=np.array([0.01]),
        time_lbs=np.array(["2026-01-01"], dtype="datetime64[us]"),
        time_ubs=np.array(["2026-01-02"], dtype="datetime64[us]"),
    )
    ds = avg.dataset
    assert ds["observables"].shape == (1,)
    assert ds["unbound_path"].values[0] == p
    assert ds["fragment_depth"].values[0] == -1
    assert float(ds["observables"].values[0]) == 0.8
    assert float(ds["std"].values[0]) == 0.01


def test_filter_time(make_cz_path):
    """Test that filter_time keeps only data within the time window."""
    p0 = make_cz_path("IX")
    p1 = make_cz_path("XI")
    avg = AveragedData.from_arrays(
        unbound_paths=[p0, p1],
        fragment_depths=[-1, -1],
        observables=np.array([0.8, 0.7]),
        std=np.array([0.01, 0.02]),
        time_lbs=np.array(["2026-01-01", "2026-01-05"], dtype="datetime64[us]"),
        time_ubs=np.array(["2026-01-02", "2026-01-06"], dtype="datetime64[us]"),
    )
    filtered = avg.filter_time(lb=np.datetime64("2026-01-04"), ub=np.datetime64("2026-01-07"))
    vals = filtered.dataset["observables"].values
    assert np.isnan(vals[0])
    assert vals[1] == 0.7
