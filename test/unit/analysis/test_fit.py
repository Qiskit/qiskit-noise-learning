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

import pytest

from qiskit_noise_learning.analysis import Fit
from qiskit_noise_learning.models import IdentityFidelityModel, LogFidelitySpace
from qiskit_noise_learning.sequences import LogPathMap


def test_accepts_fidelity_model_and_none(gate_set_cz):
    assert Fit(model=IdentityFidelityModel(gate_set_cz)).model is not None
    assert Fit(model=None).model is None


def test_rejects_non_linear_map(gate_set_cz):
    # a gate set is not a LinearMap at all
    with pytest.raises(TypeError, match="fidelity model"):
        Fit(model=gate_set_cz)


def test_rejects_non_fidelity_linear_map(gate_set_cz):
    # a LinearMap whose output space is a LogPathSpace, not a LogFidelitySpace
    with pytest.raises(TypeError, match="fidelity model"):
        Fit(model=LogPathMap(LogFidelitySpace(gate_set_cz)))
