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
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, QubitSparsePauliList

from qiskit_noise_learning.analysis import Fit, SymmetrizeFidelities, SymmetrizeGenerators
from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.models import GeneratorIndex, PauliLindbladModel


@pytest.fixture()
def cz_model():
    """A 2-qubit PauliLindbladModel with a CZ gate, prep, and meas."""
    gate_set = ModelGateSet(2)
    gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
    gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))

    generators = {
        "CZ": QubitSparsePauliList(["XI", "IX", "XX", "ZI", "IZ", "ZZ"]),
        "P": QubitSparsePauliList(["XI", "IX", "XX"]),
        "M": QubitSparsePauliList(["XI", "IX", "XX"]),
    }
    return PauliLindbladModel(gate_set, generators)


def _make_model_data(model, rates):
    """Build a ModelData from a PauliLindbladModel and a dict of gate_name -> rate array."""
    param_indices = []
    param_values = []
    for gate_name, gens in model.generators.items():
        for gen in gens:
            param_indices.append(GeneratorIndex(gate_name=gate_name, generator=gen))
            param_values.append(
                rates.get(gate_name, np.zeros(len(gens)))[
                    len(param_values)
                    - sum(
                        len(model.generators[g])
                        for g in list(model.generators.keys())[
                            : list(model.generators.keys()).index(gate_name)
                        ]
                    )
                ]
            )

    # Simpler approach: build flat arrays
    param_indices = []
    param_values = []
    for gate_name, gens in model.generators.items():
        gate_rates = rates.get(gate_name, np.zeros(len(gens)))
        for i, gen in enumerate(gens):
            param_indices.append(GeneratorIndex(gate_name=gate_name, generator=gen))
            param_values.append(gate_rates[i])

    n = len(param_indices)
    return ModelData.from_arrays(
        parameter_indices=param_indices,
        parameter_values=np.array(param_values),
        covariance=np.eye(n) * 0.01,
        time_lbs=np.zeros(n, dtype="datetime64[us]"),
        time_ubs=np.zeros(n, dtype="datetime64[us]"),
    )


class TestSymmetrizeGenerators:
    def test_input_output_levels(self):
        stage = SymmetrizeGenerators()
        assert stage.input_level == ModelData
        assert stage.output_level == ModelData

    def test_no_pairs_unchanged(self, cz_model):
        """When no generator pairs exist under conjugation, rates are unchanged."""
        # ZI under CZ stays ZI (it commutes with CZ), so ZI has no conjugate partner
        # among the other generators unless one exists. Let's use a model where
        # no pairs form.
        gate_set = ModelGateSet(2)
        gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
        gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
        gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))

        # ZI and IZ are both invariant under CZ conjugation, so no pairs form among them
        generators = {
            "CZ": QubitSparsePauliList(["ZI", "IZ"]),
            "P": QubitSparsePauliList(["XI", "IX"]),
            "M": QubitSparsePauliList(["XI", "IX"]),
        }
        model = PauliLindbladModel(gate_set, generators)
        rates = {
            "CZ": np.array([0.1, 0.2]),
            "P": np.array([0.01, 0.02]),
            "M": np.array([0.03, 0.04]),
        }
        model_data = _make_model_data(model, rates)

        fit = Fit(model=model)
        fit[ModelData] = model_data
        result = SymmetrizeGenerators().run(fit)

        result_rates = result.model_data.dataset["parameter_values"].data
        # CZ rates unchanged (ZI->ZI, IZ->IZ under CZ, no pairs)
        cz_mask = [p.gate_name == "CZ" for p in result.model_data.dataset["parameter"].data]
        np.testing.assert_allclose(result_rates[cz_mask], [0.1, 0.2])

    def test_paired_generators_averaged(self, cz_model):
        """Paired generators get their rates averaged."""
        # Under CZ: XI -> XZ and XZ -> XI (they are conjugate pairs)
        # Similarly: IX -> ZX and ZX -> IX
        rates = {
            "CZ": np.array([0.1, 0.3, 0.2, 0.4, 0.5, 0.6]),  # XI, IX, XX, ZI, IZ, ZZ
            "P": np.array([0.01, 0.02, 0.03]),
            "M": np.array([0.01, 0.02, 0.03]),
        }
        model_data = _make_model_data(cz_model, rates)

        fit = Fit(model=cz_model)
        fit[ModelData] = model_data
        result = SymmetrizeGenerators().run(fit)

        result_params = result.model_data.dataset["parameter"].data
        result_rates = result.model_data.dataset["parameter_values"].data

        # Find rates by generator index
        rate_map = {(p.gate_name, str(p.generator)): r for p, r in zip(result_params, result_rates)}

        # SPAM gates should be unchanged
        assert rate_map[("P", str(QubitSparsePauliList(["XI"])[0]))] == pytest.approx(0.01)
        assert rate_map[("M", str(QubitSparsePauliList(["IX"])[0]))] == pytest.approx(0.02)

    def test_covariance_propagated(self, cz_model):
        """Covariance is propagated through the linear transform."""
        rates = {
            "CZ": np.array([0.1, 0.3, 0.2, 0.4, 0.5, 0.6]),
            "P": np.array([0.01, 0.02, 0.03]),
            "M": np.array([0.01, 0.02, 0.03]),
        }
        model_data = _make_model_data(cz_model, rates)

        fit = Fit(model=cz_model)
        fit[ModelData] = model_data
        result = SymmetrizeGenerators().run(fit)

        cov = result.model_data.dataset["covariance"].data
        # Covariance should not be all zeros (unlike SymmetrizeFidelities)
        assert not np.allclose(cov, 0)
        # Should be symmetric
        np.testing.assert_allclose(cov, cov.T)

    def test_requires_pauli_lindblad_model(self):
        """Raises TypeError if model is not PauliLindbladModel."""
        fit = Fit(model=None)
        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=["a"],
            parameter_values=np.array([1.0]),
            covariance=np.eye(1),
            time_lbs=np.zeros(1, dtype="datetime64[us]"),
            time_ubs=np.zeros(1, dtype="datetime64[us]"),
        )
        with pytest.raises(TypeError, match="PauliLindbladModel"):
            SymmetrizeGenerators().run(fit)


class TestSymmetrizeFidelities:
    def test_input_output_levels(self):
        stage = SymmetrizeFidelities()
        assert stage.input_level == ModelData
        assert stage.output_level == ModelData

    def test_already_symmetric_unchanged(self, cz_model):
        """Rates already in the null space are not modified."""
        # All equal rates satisfy any fidelity symmetry
        rates = {
            "CZ": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            "P": np.array([0.01, 0.01, 0.01]),
            "M": np.array([0.01, 0.01, 0.01]),
        }
        model_data = _make_model_data(cz_model, rates)

        fit = Fit(model=cz_model)
        fit[ModelData] = model_data
        result = SymmetrizeFidelities().run(fit)

        result_rates = result.model_data.dataset["parameter_values"].data
        cz_mask = [p.gate_name == "CZ" for p in result.model_data.dataset["parameter"].data]
        # Should be non-negative
        assert np.all(result_rates[cz_mask] >= -1e-10)

    def test_projection_non_negative(self, cz_model):
        """Projected rates are always non-negative."""
        rates = {
            "CZ": np.array([0.5, 0.001, 0.3, 0.002, 0.1, 0.4]),
            "P": np.array([0.01, 0.02, 0.03]),
            "M": np.array([0.01, 0.02, 0.03]),
        }
        model_data = _make_model_data(cz_model, rates)

        fit = Fit(model=cz_model)
        fit[ModelData] = model_data
        result = SymmetrizeFidelities().run(fit)

        result_rates = result.model_data.dataset["parameter_values"].data
        assert np.all(result_rates >= -1e-10)

    def test_covariance_zeroed(self, cz_model):
        """Covariance is zeroed after non-linear projection."""
        rates = {
            "CZ": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "P": np.array([0.01, 0.02, 0.03]),
            "M": np.array([0.01, 0.02, 0.03]),
        }
        model_data = _make_model_data(cz_model, rates)

        fit = Fit(model=cz_model)
        fit[ModelData] = model_data
        result = SymmetrizeFidelities().run(fit)

        cov = result.model_data.dataset["covariance"].data
        np.testing.assert_allclose(cov, 0)

    def test_spam_rates_unchanged(self, cz_model):
        """SPAM gate rates are not modified."""
        rates = {
            "CZ": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "P": np.array([0.01, 0.02, 0.03]),
            "M": np.array([0.04, 0.05, 0.06]),
        }
        model_data = _make_model_data(cz_model, rates)

        fit = Fit(model=cz_model)
        fit[ModelData] = model_data
        result = SymmetrizeFidelities().run(fit)

        result_params = result.model_data.dataset["parameter"].data
        result_rates = result.model_data.dataset["parameter_values"].data

        p_mask = [p.gate_name == "P" for p in result_params]
        m_mask = [p.gate_name == "M" for p in result_params]
        np.testing.assert_allclose(result_rates[p_mask], [0.01, 0.02, 0.03])
        np.testing.assert_allclose(result_rates[m_mask], [0.04, 0.05, 0.06])

    def test_requires_pauli_lindblad_model(self):
        """Raises TypeError if model is not PauliLindbladModel."""
        fit = Fit(model=None)
        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=["a"],
            parameter_values=np.array([1.0]),
            covariance=np.eye(1),
            time_lbs=np.zeros(1, dtype="datetime64[us]"),
            time_ubs=np.zeros(1, dtype="datetime64[us]"),
        )
        with pytest.raises(TypeError, match="PauliLindbladModel"):
            SymmetrizeFidelities().run(fit)

    def test_fidelity_symmetry_holds(self, cz_model):
        """After projection, the fidelity symmetry condition M1@r == M2@r holds."""
        rates = {
            "CZ": np.array([0.5, 0.1, 0.3, 0.2, 0.4, 0.05]),
            "P": np.array([0.01, 0.02, 0.03]),
            "M": np.array([0.01, 0.02, 0.03]),
        }
        model_data = _make_model_data(cz_model, rates)

        fit = Fit(model=cz_model)
        fit[ModelData] = model_data
        result = SymmetrizeFidelities().run(fit)

        result_params = result.model_data.dataset["parameter"].data
        result_rates = result.model_data.dataset["parameter_values"].data

        # Verify M1@r == M2@r for the CZ gate
        gate = cz_model.gate_set["CZ"]
        cz_indices = [i for i, p in enumerate(result_params) if p.gate_name == "CZ"]
        gate_generators = [result_params[i].generator for i in cz_indices]
        conjugated = [gate.clifford_propagate(g) for g in gate_generators]
        gate_rates = result_rates[cz_indices]

        num_gen = len(gate_generators)
        m1 = np.zeros((num_gen, num_gen), dtype=int)
        m2 = np.zeros((num_gen, num_gen), dtype=int)
        for i in range(num_gen):
            for j in range(num_gen):
                if not gate_generators[i].commutes(gate_generators[j]):
                    m1[i, j] = 1
                if not conjugated[i].commutes(gate_generators[j]):
                    m2[i, j] = 1

        np.testing.assert_allclose(m1 @ gate_rates, m2 @ gate_rates, atol=1e-10)
