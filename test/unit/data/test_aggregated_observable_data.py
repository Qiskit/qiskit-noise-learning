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

from qiskit_noise_learning.data import AggregatedObservableData


def test_from_arrays(make_cz_path):
    """Test constructing AggregatedObservableData from arrays."""
    p = make_cz_path("IX")
    avg = AggregatedObservableData.from_arrays(
        unbound_paths=[p],
        fragment_depths=[-1],
        estimate_values=np.array([0.8]),
        estimate_std=np.array([0.01]),
        time_lbs=np.array(["2026-01-01"], dtype="datetime64[us]"),
        time_ubs=np.array(["2026-01-02"], dtype="datetime64[us]"),
    )
    ds = avg.dataset
    assert ds["estimate_values"].shape == (1,)
    assert ds["unbound_path"].values[0] == p
    assert ds["fragment_depth"].values[0] == -1
    assert float(ds["estimate_values"].values[0]) == 0.8
    assert float(ds["estimate_std"].values[0]) == 0.01
