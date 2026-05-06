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

import numpy as np

from qiskit_noise_learning.data import ObservableData


@dataclass(frozen=True)
class MockPathPattern:
    name: str


def test_from_arrays():
    """Test constructing ObservableData with a single observable."""
    pp = MockPathPattern("pp0")
    obs = ObservableData.from_arrays(
        path_patterns=[pp],
        depths=[3],
        observables=np.array([[0.9, 0.85, 0.92]]),
        time_lbs=np.array([["2026-01-01", "2026-01-01", "2026-01-01"]], dtype="datetime64[us]"),
        time_ubs=np.array([["2026-01-02", "2026-01-02", "2026-01-02"]], dtype="datetime64[us]"),
    )
    ds = obs.dataset
    assert ds["observables"].shape == (1, 3)
    assert ds["path_pattern"].values[0] == pp
    assert ds["depth"].values[0] == 3


def test_filter_time():
    """Test that filter_time keeps only data within the time window."""
    pp = MockPathPattern("pp0")
    t_lbs = np.array([["2026-01-01", "2026-01-03", "2026-01-05"]], dtype="datetime64[us]")
    t_ubs = np.array([["2026-01-02", "2026-01-04", "2026-01-06"]], dtype="datetime64[us]")
    obs = ObservableData.from_arrays(
        path_patterns=[pp],
        depths=[1],
        observables=np.array([[0.9, 0.8, 0.7]]),
        time_lbs=t_lbs,
        time_ubs=t_ubs,
    )
    filtered = obs.filter_time(lb=np.datetime64("2026-01-03"), ub=np.datetime64("2026-01-04"))
    vals = filtered.dataset["observables"].values[0]
    assert np.isnan(vals[0])
    assert vals[1] == 0.8
    assert np.isnan(vals[2])
