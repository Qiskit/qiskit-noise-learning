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
from qiskit.quantum_info import QubitSparsePauli, QubitSparsePauliList

from qiskit_noise_learning.analysis import (
    AnalysisPipeline,
    CurveFitObservables,
    Fit,
    NNLSSolve,
)
from qiskit_noise_learning.data import ModelData, ObservableData
from qiskit_noise_learning.models._legacy import GeneratorIndex, PauliLindbladModel

# Prep and measurement generators required to build the model; they do not appear in the rows of the
# unbound (decay) paths used here.
_PM_GENS = {"P": QubitSparsePauliList(["XI"]), "M": QubitSparsePauliList(["XI"])}


def _get_rate_from_fit(fit, gate_name, label):
    """Read a fitted rate by its generator label."""
    gen = GeneratorIndex(gate_name, QubitSparsePauli(label))
    return fit.model_data.dataset["parameter_values"].sel(parameter=gen).item()


class TestAnalysisPipelineIntegration:
    """Tests analysis pipeline on synthetic data."""

    def test_curve_fit_then_nnls_single_unbound_path(
        self, gate_set_cz, make_cz_path, make_observable_data
    ):
        """Test curve fitting then linear solving."""
        a_true, f_true = 0.9, 0.8
        model = PauliLindbladModel(gate_set_cz, {"CZ": QubitSparsePauliList(["ZI"]), **_PM_GENS})
        path = make_cz_path("XI")  # row = {CZ:ZI: 4.0}
        obs = make_observable_data([(path, a_true, f_true, [1, 2, 3, 4, 5])])

        pipeline = AnalysisPipeline(CurveFitObservables(), NNLSSolve())
        assert pipeline.input_level is ObservableData
        assert pipeline.output_level is ModelData

        fit = Fit(model=model)
        fit[ObservableData] = obs
        result = pipeline.run(fit)

        assert isinstance(result.model_data, ModelData)
        # The curve fit recovers -log(f); NNLS divides by the coefficient 4.
        assert np.isclose(_get_rate_from_fit(result, "CZ", "ZI"), -np.log(f_true) / 4, atol=0.05)

    def test_curve_fit_then_nnls_multiple_unbound_paths(
        self, gate_set_cz, make_cz_path, make_observable_data
    ):
        """Test curve fitting then linear solving with multiple unbound paths."""
        f0, f1 = 0.9, 0.7
        model = PauliLindbladModel(
            gate_set_cz, {"CZ": QubitSparsePauliList(["ZI", "IZ"]), **_PM_GENS}
        )
        path0 = make_cz_path("XI")  # row = {CZ:ZI: 4.0}
        path1 = make_cz_path("IX")  # row = {CZ:IZ: 4.0}
        obs = make_observable_data(
            [
                (path0, 0.95, f0, [1, 2, 3, 4, 5]),
                (path1, 0.85, f1, [1, 2, 3, 4, 5]),
            ]
        )

        pipeline = AnalysisPipeline(CurveFitObservables(), NNLSSolve())
        fit = Fit(model=model)
        fit[ObservableData] = obs
        result = pipeline.run(fit)

        assert np.isclose(_get_rate_from_fit(result, "CZ", "ZI"), -np.log(f0) / 4, atol=0.05)
        assert np.isclose(_get_rate_from_fit(result, "CZ", "IZ"), -np.log(f1) / 4, atol=0.05)
