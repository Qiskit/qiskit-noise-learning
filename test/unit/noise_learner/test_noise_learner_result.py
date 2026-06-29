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
import pytest
from qiskit.quantum_info import PauliLindbladMap, QubitSparsePauli, QubitSparsePauliList

from qiskit_noise_learning.analysis import Fit
from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.models import GeneratorIndex, PauliLindbladModel
from qiskit_noise_learning.noise_learner import NoiseLearnerResult


@pytest.fixture()
def generators_cz():
    return {
        "CZ": QubitSparsePauliList(["ZI", "IZ"]),
        "P": QubitSparsePauliList(["XI"]),
        "M": QubitSparsePauliList(["IX"]),
    }


@pytest.fixture()
def model(gate_set_cz, generators_cz):
    return PauliLindbladModel(gate_set_cz, generators_cz)


@pytest.fixture()
def model_data(model):
    indices = [
        GeneratorIndex(gate_name="CZ", generator=QubitSparsePauli(gen_label))
        for gen_label in ["ZI", "IZ"]
    ]
    values = np.array([0.01, 0.01])
    return ModelData.from_arrays(
        parameter_indices=indices,
        parameter_values=values,
        covariance=np.zeros((2, 2)),
        time_lbs=np.empty(2, dtype="datetime64[us]"),
        time_ubs=np.empty(2, dtype="datetime64[us]"),
    )


@pytest.fixture()
def fit_with_model_data(model, model_data):
    fit = Fit(model=model)
    fit[ModelData] = model_data
    return fit


@pytest.fixture()
def result(fit_with_model_data):
    return NoiseLearnerResult(fit_with_model_data)


def test_noise_learner_result_to_dict(result):
    """Test the NoiseLearnerResult.to_dict method."""
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "CZ" in d
    assert "P" not in d
    assert "M" not in d

    for value in d.values():
        assert isinstance(value, PauliLindbladMap)


def test_noise_learner_to_dict_raises_when_model_data_absent():
    """Test NoiseLearnerResult.to_dict raises when no model data is present."""
    fit = Fit()
    result = NoiseLearnerResult(fit)
    with pytest.raises(RuntimeError, match="analysis may not have completed"):
        result.to_dict()


def test_noise_learner_result_to_dict_zero_rate_parameters(model):
    """Test NoiseLearnerResult.to_dict with simple data."""
    indices = [GeneratorIndex(gate_name="CZ", generator=QubitSparsePauli("ZI"))]
    model_data = ModelData.from_arrays(
        parameter_indices=indices,
        parameter_values=np.array([0.0]),
        covariance=np.zeros((1, 1)),
        time_lbs=np.empty(1, dtype="datetime64[us]"),
        time_ubs=np.empty(1, dtype="datetime64[us]"),
    )
    fit = Fit(model=model)
    fit[ModelData] = model_data
    result = NoiseLearnerResult(fit)
    assert isinstance(result.to_dict(), dict)
