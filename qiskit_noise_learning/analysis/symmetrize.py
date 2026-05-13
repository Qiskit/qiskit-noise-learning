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

"""Noise symmetry projection stages."""

import numpy as np
from scipy.linalg import null_space

from qiskit_noise_learning.analysis.analysis_pipeline import AnalysisStage
from qiskit_noise_learning.analysis.fit import Fit
from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.models import PauliLindbladModel


class SymmetrizeGenerators(AnalysisStage):
    """Project generator rates to satisfy conjugation symmetry, gate by gate.

    For each gate, finds the pairs of generators that map to each under its conjugate action. The
    covariance is propagated through averaging.
    """

    input_level = ModelData
    output_level = ModelData

    def _run(self, fit: Fit):
        model = fit.model
        if not isinstance(model, PauliLindbladModel):
            raise TypeError("SymmetrizeGenerators requires a PauliLindbladModel on the Fit.")

        model_data = fit[ModelData]
        params = model_data.dataset["parameter"].data
        rates = model_data.dataset["parameter_values"].data.copy()
        covariance = model_data.dataset["covariance"].data.copy()

        n = len(rates)
        transform = np.eye(n)

        for gate_name, gate in model.gate_set.items():
            if gate_name in model.prep_names or gate_name in model.meas_names:
                continue

            gate_indices = [i for i, p in enumerate(params) if p.gate_name == gate_name]
            if not gate_indices:
                continue

            gate_generators = [params[i].generator for i in gate_indices]
            conjugated = [gate.clifford_propagate(g) for g in gate_generators]

            paired = set()
            for i, conj_i in enumerate(conjugated):
                if i in paired:
                    continue
                for j in range(i + 1, len(gate_generators)):
                    if j in paired:
                        continue
                    if conj_i == gate_generators[j]:
                        paired.add(i)
                        paired.add(j)
                        gi = gate_indices[i]
                        gj = gate_indices[j]
                        transform[gi, gi] = 0.5
                        transform[gi, gj] = 0.5
                        transform[gj, gi] = 0.5
                        transform[gj, gj] = 0.5
                        break

        rates = transform @ rates
        covariance = transform @ covariance @ transform.T

        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=list(params),
            parameter_values=rates,
            covariance=covariance,
            time_lbs=model_data.dataset["time_lbs"].data,
            time_ubs=model_data.dataset["time_ubs"].data,
            metadata=model_data.metadata,
        )


class SymmetrizeFidelities(AnalysisStage):
    """Project generator rates into the fidelity-symmetry null space, gate by gate.

    For each gate, builds commutation matrices of generators against generators and conjugated
    generators against generators, then projects rates into the null space of their difference.
    Covariance is zeroed (the iterative projection with clipping is non-linear).
    """

    input_level = ModelData
    output_level = ModelData

    def _run(self, fit: Fit):
        model = fit.model
        if not isinstance(model, PauliLindbladModel):
            raise TypeError("SymmetrizeFidelities requires a PauliLindbladModel on the Fit.")

        model_data = fit[ModelData]
        params = model_data.dataset["parameter"].data
        rates = model_data.dataset["parameter_values"].data.copy()

        for gate_name, gate in model.gate_set.items():
            if gate_name in model.prep_names or gate_name in model.meas_names:
                continue

            gate_indices = [i for i, p in enumerate(params) if p.gate_name == gate_name]
            if not gate_indices:
                continue

            gate_generators = [params[i].generator for i in gate_indices]
            conjugated = [gate.clifford_propagate(g) for g in gate_generators]

            num_gen = len(gate_generators)
            m1 = np.zeros((num_gen, num_gen), dtype=int)
            m2 = np.zeros((num_gen, num_gen), dtype=int)

            for i in range(num_gen):
                for j in range(num_gen):
                    if not gate_generators[i].commutes(gate_generators[j]):
                        m1[i, j] = 1
                    if not conjugated[i].commutes(gate_generators[j]):
                        m2[i, j] = 1

            ns = null_space(m1 - m2)
            if ns.size == 0:
                continue

            gate_rates = rates[gate_indices]
            projected = self._project_to_nullspace(gate_rates, ns)

            eps = 1e-6
            for _ in range(100):
                if not np.any(projected < -eps):
                    break
                projected[projected < 0] = 0
                projected = self._project_to_nullspace(projected, ns)
            else:
                raise RuntimeError(
                    f"Failed to project rates for gate '{gate_name}' to meet symmetry condition."
                )

            projected[projected < 0] = 0
            rates[gate_indices] = projected

        n = len(rates)
        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=list(params),
            parameter_values=rates,
            covariance=np.zeros((n, n)),
            time_lbs=model_data.dataset["time_lbs"].data,
            time_ubs=model_data.dataset["time_ubs"].data,
            metadata=model_data.metadata,
        )

    @staticmethod
    def _project_to_nullspace(vec: np.ndarray, ns: np.ndarray) -> np.ndarray:
        return np.sum(ns * (ns.T @ vec), axis=1)


class SymmetrizeFidelities2(AnalysisStage):
    """Project generator rates into the fidelity-symmetry null space, gate by gate.

    For each gate, builds commutation matrices of generators against generators and conjugated
    generators against generators, then projects rates into the null space of their difference.
    Covariance is zeroed (the iterative projection with clipping is non-linear).
    """

    input_level = ModelData
    output_level = ModelData

    def _run(self, fit: Fit):
        model = fit.model
        if not isinstance(model, PauliLindbladModel):
            raise TypeError("SymmetrizeFidelities requires a PauliLindbladModel on the Fit.")

        model_data = fit[ModelData]
        params = model_data.dataset["parameter"].data
        rates = model_data.dataset["parameter_values"].data.copy()

        for gate_name, gate in model.gate_set.items():
            if gate_name in model.prep_names or gate_name in model.meas_names:
                continue

            gate_indices = [i for i, p in enumerate(params) if p.gate_name == gate_name]
            if not gate_indices:
                continue

            gate_generators = [params[i].generator for i in gate_indices]
            found = []
            symmetric_generators = []
            for g in gate_generators:
                if g in found:
                    continue
                found.append(g)
                conj_g = gate.clifford_propagate(g)
                if (conj_g == g) or conj_g not in gate_generators:
                    symmetric_generators.append([g])
                else:
                    symmetric_generators.append([conj_g, g])
                    found.append(conj_g)

            # construct map from reduced fidelities to full fidelities
            A = np.zeros((len(gate_generators), len(symmetric_generators)), dtype=float)
            for col_idx, sym_gens in enumerate(symmetric_generators):
                for g in sym_gens:
                    row_idx = gate_generators.index(g)
                    A[row_idx, col_idx] = 1.0

            # construct map from rates to full fidelities
            # ideally should use fidelity model for this, however I'm worried about ordering
            H = np.zeros((len(gate_generators), len(gate_generators)))
            for row_idx, row_gen in enumerate(gate_generators):
                for col_idx, col_gen in enumerate(gate_generators):
                    if not row_gen.commutes(col_gen):
                        H[row_idx, col_idx] = 2.0  # 1. or 2.???

            N = null_space(A.T @ H)
            sub_rates = rates[gate_indices]
            projected = sub_rates - (N @ N.T) @ sub_rates

            projected[projected < 0] = 0
            rates[gate_indices] = projected

        n = len(rates)
        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=list(params),
            parameter_values=rates,
            covariance=np.zeros((n, n)),
            time_lbs=model_data.dataset["time_lbs"].data,
            time_ubs=model_data.dataset["time_ubs"].data,
            metadata=model_data.metadata,
        )

    @staticmethod
    def _project_to_nullspace(vec: np.ndarray, ns: np.ndarray) -> np.ndarray:
        return np.sum(ns * (ns.T @ vec), axis=1)


class SymmetrizeFidelities3(AnalysisStage):
    """Project generator rates into the fidelity-symmetry null space, gate by gate.

    For each gate, builds commutation matrices of generators against generators and conjugated
    generators against generators, then projects rates into the null space of their difference.
    Covariance is zeroed (the iterative projection with clipping is non-linear).
    """

    input_level = ModelData
    output_level = ModelData

    def _run(self, fit: Fit):
        model = fit.model
        if not isinstance(model, PauliLindbladModel):
            raise TypeError("SymmetrizeFidelities requires a PauliLindbladModel on the Fit.")

        model_data = fit[ModelData]
        params = model_data.dataset["parameter"].data
        rates = model_data.dataset["parameter_values"].data.copy()

        pauli_lindblad_maps = model.to_pauli_lindblad_maps(model_data)

        for gate_name, gate in model.gate_set.items():
            if gate_name in model.prep_names or gate_name in model.meas_names:
                continue

            gate_indices = [i for i, p in enumerate(params) if p.gate_name == gate_name]
            if not gate_indices:
                continue
            gate_generators = [params[i].generator for i in gate_indices]

            # build list of fidelities for this gate
            pauli_lindblad_map = pauli_lindblad_maps[gate_name]
            fidelities = np.empty(len(gate_indices), dtype=float)
            found = []
            symmetric_indices = []
            for idx, g in enumerate(gate_generators):
                # compute fidelity
                fidelities[idx] = pauli_lindblad_map.pauli_fidelity(g)

                # logic for symmetric index construction
                if g in found:
                    continue
                found.append(g)
                conj_g = gate.clifford_propagate(g)
                if (conj_g != g) and conj_g in gate_generators:
                    found.append(conj_g)
                    symmetric_indices.append((idx, gate_generators.index(conj_g)))
            # modify fidelities so symmetric indices have the same value
            for i, j in symmetric_indices:
                # compute fidelity product
                fid_prod = pauli_lindblad_maps[gate_name].pauli_fidelity(
                    gate_generators[i]
                ) * pauli_lindblad_maps[gate_name].pauli_fidelity(gate_generators[j])

                # compute square root
                fid_prod_sqrt = np.sqrt(fid_prod)

                # set to the values in fidelities
                fidelities[i] = fid_prod_sqrt
                fidelities[j] = fid_prod_sqrt

            # map back to rates
            # construct map from rates to full fidelities
            # ideally should use fidelity model for this, however I'm worried about ordering
            H = np.zeros((len(gate_generators), len(gate_generators)))
            for row_idx, row_gen in enumerate(gate_generators):
                for col_idx, col_gen in enumerate(gate_generators):
                    if not row_gen.commutes(col_gen):
                        H[row_idx, col_idx] = 2.0  # 1. or 2.???

            mod_rates = np.linalg.inv(H) @ -np.log(fidelities)
            rates[gate_indices] = mod_rates

        n = len(rates)
        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=list(params),
            parameter_values=rates,
            covariance=np.zeros((n, n)),
            time_lbs=model_data.dataset["time_lbs"].data,
            time_ubs=model_data.dataset["time_ubs"].data,
            metadata=model_data.metadata,
        )

    @staticmethod
    def _project_to_nullspace(vec: np.ndarray, ns: np.ndarray) -> np.ndarray:
        return np.sum(ns * (ns.T @ vec), axis=1)
