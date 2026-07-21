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

"""Shared helpers for analysis tests."""

import numpy as np
import pytest

from qiskit_noise_learning.analysis import Fit
from qiskit_noise_learning.data import RawData
from qiskit_noise_learning.gate_sets import ModelGateSet
from qiskit_noise_learning.models import IdentityFidelityModel


@pytest.fixture()
def make_fit():
    """Return a builder ``(raw_data, coupling_map) -> Fit``.

    The fit wraps a real :class:`~.IdentityFidelityModel` over a real
    :class:`~.ModelGateSet` carrying the requested coupling map, since the
    post-select stages read only ``fit.model.gate_set.coupling_map``.
    """

    def _make(raw_data, coupling_map):
        gate_set = ModelGateSet(coupling_map.size(), coupling_map=coupling_map)
        fit = Fit(model=IdentityFidelityModel(gate_set))
        fit[RawData] = raw_data
        return fit

    return _make


@pytest.fixture()
def make_raw_data(make_instruction_sequence):
    """Return a builder ``(creg_names, measurement_map, data) -> RawData`` (1 sequence)."""

    def _make(creg_names, measurement_map, data):
        seq = make_instruction_sequence(name="p0", fragment_depth=1)
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

    return _make
