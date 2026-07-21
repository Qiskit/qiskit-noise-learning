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

"""Legacy noise-model fitter retained as a cross-check reference. Accepts only a single layer
in a gate set.
"""

from collections import defaultdict
from typing import Any, Literal

import numpy as np
import scipy.optimize as opt
from numpy.typing import ArrayLike
from qiskit.quantum_info import PauliLindbladMap, PauliList, QubitSparsePauliList
from scipy.sparse import csr_array

from qiskit_noise_learning.analysis import AnalysisStage, Fit
from qiskit_noise_learning.data import AveragedData, ModelData
from qiskit_noise_learning.data.xarray_utils import time_bound
from qiskit_noise_learning.models import GeneratorIndex

from ..optionals import HAS_CVXPY

OptimizerLiteral = Literal["nnls", "lsq_linear_sparse", "cvxpy"]
NoiseAssumptionLiteral = Literal["symmetric_fidelities", "symmetric_generators"]


class LegacySolve(AnalysisStage):
    """Solves for the :class:`~.ModelData` using the legacy pair-fidelity method.

    This solver assumes that the gate set only has a single unitary gate, and that the paths
    are of a vanilla-learning type (i.e. even depth with no single-qubit Cliffords required).

    Delegates to :func:`fit_noise_model_legacy` with ``noise_assumption="symmetric_fidelities"``,
    ``optimizer_name="nnls"``, and ``constrained=True``.
    """

    input_level = AveragedData
    output_level = ModelData

    def _run(self, fit: Fit) -> None:
        averaged_data = fit[AveragedData]

        noise_map = fit_noise_model_legacy(
            fit,
            noise_assumption="symmetric_fidelities",
            decimals=None,
            optimizer_name="nnls",
            constrained=True,
        )

        layer_name = (
            fit.averaged_data.dataset.unbound_path[0].item().repeatable_fragment[0].gate_name
        )

        param_labels = [
            GeneratorIndex(gate_name=layer_name, generator=g) for g in noise_map.generators()
        ]
        x = np.array(noise_map.rates)
        cov_x = np.zeros(shape=(len(x), len(x)))
        metadata = {}
        # Filter to decay data (fragment_depth == -1)
        decay_mask = averaged_data.dataset["fragment_depth"].data == -1
        decay_dataset = averaged_data.dataset.sel({"observable": decay_mask})

        time_lb = time_bound(decay_dataset["time_lbs"].data, "min")
        time_ub = time_bound(decay_dataset["time_ubs"].data, "max")

        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=param_labels,
            parameter_values=x,
            covariance=cov_x,
            time_lbs=np.full(len(x), time_lb, dtype="datetime64[us]"),
            time_ubs=np.full(len(x), time_ub, dtype="datetime64[us]"),
            metadata=metadata,
        )


def get_fid_pairs(unbound_paths) -> tuple[QubitSparsePauliList, QubitSparsePauliList]:
    """Extract the first and second Paulis from the repeatable fragment of each unbound path.

    Args:
        unbound_paths.

    Returns:
        A pair ``(fid_ps_1, fid_ps_2)`` of ``QubitSparsePauliList`` objects holding,
        respectively, the first and second Pauli of each repeatable fragment.

    Raises:
        ValueError: If any unbound path's ``repeatable_fragment`` does not have exactly 2 entries.
    """
    fid_pairs = []

    for path in unbound_paths:
        fragment: Any = path.repeatable_fragment
        if len(fragment) != 2:
            raise ValueError(
                "Expected each unbound path's repeatable_fragment to have exactly 2 entries, "
                f"but got {len(fragment)}."
            )
        if (
            path.repeatable_fragment[0].transition[1] != path.repeatable_fragment[1].transition[0]
        ) or (
            path.repeatable_fragment[1].transition[0] != path.repeatable_fragment[0].transition[1]
        ):
            raise ValueError(
                "Encountered path whose repeatable fragment requires single qubit Cliffords to "
                "traverse."
            )
        fid_pairs.append([fragment[0].pauli, fragment[1].pauli])

    # Convert to QubitSparsePauliList
    fid_ps_1 = QubitSparsePauliList(np.array(fid_pairs)[:, 0].tolist())
    fid_ps_2 = QubitSparsePauliList(np.array(fid_pairs)[:, 1].tolist())
    return fid_ps_1, fid_ps_2


def make_canonical_fid_dict(
    fid_ps_1: list[str], fid_ps_2: list[str], fid_pairs_data: ArrayLike
) -> dict[str, float]:
    """Build a dict mapping each weight-<3 Pauli label to its mean pair fidelity.

    Paulis with weight >= 3 are excluded. When a Pauli appears in both *fid_ps_1* and
    *fid_ps_2*, its fidelity values from both positions are averaged.

    Args:
        fid_ps_1: Pauli labels for the first element of each repeatable fragment.
        fid_ps_2: Pauli labels for the second element of each repeatable fragment.
        fid_pairs_data: Fidelity values indexed parallel to *fid_ps_1* and *fid_ps_2*.

    Returns:
        A dict whose keys are weight-<3 Pauli labels and whose values are the
        corresponding mean fidelities.
    """
    # collect each weight-<3 Pauli's fidelity values (one per occurrence in either list)
    fid_pair_p_dict = defaultdict(list)
    for i, p in enumerate(fid_ps_1):
        if len(p.replace("I", "")) < 3:
            fid_pair_p_dict[p].append(fid_pairs_data[i])
    for i, p in enumerate(fid_ps_2):
        if len(p.replace("I", "")) < 3:
            fid_pair_p_dict[p].append(fid_pairs_data[i])

    fid_pair_p_dict_mean = {}
    for p, evlist in fid_pair_p_dict.items():
        if np.std(evlist) != 0:
            raise ValueError("fid_pairs_data does not meet legacy learner assumptions!")
        fid_pair_p_dict_mean[p] = np.mean(evlist)
    return fid_pair_p_dict_mean


def make_conj_pauli_list(
    pauli_list: list[str], fid_ps_1: list[str], fid_ps_2: list[str]
) -> list[str]:
    """Return the conjugate Pauli for each entry in ``pauli_list``.

    For each ``p`` in ``pauli_list``: if ``p`` appears in ``fid_ps_1`` at index ``i``, the
    conjugate is ``fid_ps_2[i]``; if ``p`` appears in *fid_ps_2* at index ``i``, the
    conjugate is ``fid_ps_1[i]``. Paulis present in neither list are silently omitted.

    Args:
        pauli_list: Pauli labels to look up.
        fid_ps_1: First-element Pauli labels from the repeatable fragments.
        fid_ps_2: Second-element Pauli labels from the repeatable fragments.

    Returns:
        A list of conjugate Pauli labels, one per entry of *pauli_list* that was found.
    """
    pconj_list = []
    for p in pauli_list:
        if p in fid_ps_1:
            p_conj = fid_ps_2[fid_ps_1.index(p)]
            pconj_list.append(p_conj)
        elif p in fid_ps_2:
            p_conj = fid_ps_1[fid_ps_2.index(p)]
            pconj_list.append(p_conj)
    return pconj_list


def fit_noise_model_legacy(
    fit: Fit,
    noise_assumption: NoiseAssumptionLiteral = "symmetric_fidelities",
    decimals: int | None = None,
    optimizer_name: OptimizerLiteral = "nnls",
    constrained: bool = True,
) -> PauliLindbladMap:
    """Fit a ``PauliLindbladMap`` from pair-fidelity data using the legacy method.

    This is a reference implementation ported from a prior codebase. The non-commuting
    (``M``) array is built directly from pair-fidelity data rather than from a
    :class:`~.FidelityModel`, so the algorithm shape intentionally differs from
    :class:`~.ModelSolve` and friends.

    Args:
        fit: A :class:`~.Fit` container holding :class:`~.AveragedData`.
        noise_assumption: How to treat Clifford conjugation of generators.
            ``"symmetric_fidelities"`` uses the square root of each pair fidelity as a
            single-layer fidelity. ``"symmetric_generators"`` assumes conjugate generator
            pairs share equal rates.
        decimals: If given, round fitted rates to this many decimal places.
        optimizer_name: Least-squares solver to use. ``"nnls"`` requires
            ``constrained=True``; ``"lsq_linear_sparse"`` and ``"cvxpy"`` support both
            constrained and unconstrained fitting.
        constrained: Whether to enforce non-negativity on the fitted rates.

    Returns:
        A ``PauliLindbladMap`` whose generators are the basis Paulis and whose rates are
        the fitted coefficients.

    Raises:
        ValueError: If *noise_assumption* or *optimizer_name* is not recognized, or if
            ``optimizer_name="nnls"`` and ``constrained=False``.
        MissingOptionalLibraryError: If ``optimizer_name="cvxpy"`` and ``cvxpy`` is not
            installed.
    """
    fid_ps_1, fid_ps_2 = get_fid_pairs(fit.averaged_data.dataset.observables.unbound_path.data)
    fid_pair_data = fit.averaged_data.dataset.observables
    fidelities_canonical = make_canonical_fid_dict(
        fid_ps_1.to_pauli_list().to_labels(), fid_ps_2.to_pauli_list().to_labels(), fid_pair_data
    )
    pauli_fidelities = np.array(list(fidelities_canonical.values()))
    basis_paulis = PauliList(list(fidelities_canonical.keys()))
    conjugated_basis_paulis = PauliList(
        make_conj_pauli_list(
            list(fidelities_canonical.keys()),
            fid_ps_1.to_pauli_list().to_labels(),
            fid_ps_2.to_pauli_list().to_labels(),
        )
    )

    sparse_model_paulis = basis_paulis.copy()
    # Form the non-commuting array (``M`` as in the PEC paper)
    nc_array_basis = np.logical_not([p.commutes(sparse_model_paulis) for p in basis_paulis]).astype(
        int
    )
    nc_array_conj_basis = np.logical_not(
        [p.commutes(sparse_model_paulis) for p in conjugated_basis_paulis]
    ).astype(int)
    nc_array = nc_array_basis + nc_array_conj_basis

    # Set up the least-squares problem according to the selected assumption:
    if noise_assumption == "symmetric_fidelities":
        # assumption lets us compute the layer fidelities:
        layer_fidelities = np.tile(np.sqrt(pauli_fidelities), 2)
        nc_array_shaped = np.concatenate([nc_array_basis, nc_array_conj_basis])
        fit_vector = -np.log(np.abs(layer_fidelities)) / 2
    elif noise_assumption == "symmetric_generators":
        # identify which generators are conjugate pairs:
        conjugated_generators = conjugated_basis_paulis.copy()

        conj_pairs = []
        for i, p in enumerate(sparse_model_paulis):
            j = np.where(conjugated_generators[i + 1 :].equiv(p))[0]
            if len(j):
                conj_pairs.append([i, j[0] + i + 1])
        conj_pairs = np.array(conj_pairs)
        # We will solve Ax = B. x is generator rates ("lambda").
        # For conj. pair (P1 P2) we fix x_1 = x_2.
        # Instead of solving for x_1 and x_2, we solve for (x_1 + x_2).
        # This makes the x vector shorter:
        #     replace x_1 with (x_1 + x_2), and delete x_2,
        # and removes columns from `nc_array_shaped`:
        #     replace col 1 with (col 1 + col 2), and delete col 2.
        # Solving Ax = b will now include the value of (x_1 + x_2).
        # Then later, we will use the symmetry assumption
        # x_1 = x_2 = (x_1 + x_2)/2
        # to construct the full list of generator rates.
        nc_array_shaped = nc_array.copy()
        nc_array_shaped[:, conj_pairs[:, 0]] += nc_array_shaped[:, conj_pairs[:, 1]]
        nc_array_shaped = np.delete(nc_array_shaped, conj_pairs[:, 1], axis=1)
        fit_vector = -np.log(np.abs(pauli_fidelities)) / 2
    else:
        raise ValueError(f"Noise assumption {noise_assumption} not recognized")

    # Perform the least squares fitting:
    if optimizer_name == "lsq_linear_sparse":
        nc_array_shaped = csr_array(nc_array_shaped)
        sparse_model_coeffs = opt.lsq_linear(
            nc_array_shaped, fit_vector, bounds=(0, np.inf) if constrained else (-np.inf, np.inf)
        ).x
    elif optimizer_name == "nnls":
        if not constrained:
            raise ValueError(
                "optimizer_name='nnls' does not support constrained=False; "
                "use 'lsq_linear_sparse' or 'cvxpy'."
            )
        sparse_model_coeffs, _ = opt.nnls(nc_array_shaped, fit_vector)
    elif optimizer_name == "cvxpy":
        HAS_CVXPY.require_now("legacy noise-model fitting with cvxpy")
        import cvxpy

        nc_array_shaped = csr_array(nc_array_shaped)
        cvxpy_fit_var = cvxpy.Variable(nc_array_shaped.shape[1])
        cost = cvxpy.sum_squares(nc_array_shaped @ cvxpy_fit_var - fit_vector)
        prob = cvxpy.Problem(
            cvxpy.Minimize(cost),
            constraints=[cvxpy_fit_var >= 0] if constrained else None,
        )
        prob.solve()
        sparse_model_coeffs = cvxpy_fit_var.value
    else:
        raise ValueError(f"Optimizer name {optimizer_name} not recognized.")

    if noise_assumption == "symmetric_generators":
        indices_to_restore = np.sort(conj_pairs[:, 1])
        sparse_model_coeffs = np.insert(
            sparse_model_coeffs,
            indices_to_restore - np.arange(len(indices_to_restore)),
            values=0,
            axis=0,
        )
        sparse_model_coeffs[conj_pairs[:, 1]] = sparse_model_coeffs[conj_pairs[:, 0]]

    # Discard small terms
    if decimals is not None:
        sparse_model_coeffs = np.round(sparse_model_coeffs, decimals)

    noise_map_pecr = PauliLindbladMap.from_list(
        list(zip(sparse_model_paulis.to_labels(), sparse_model_coeffs))
    )
    return noise_map_pecr
