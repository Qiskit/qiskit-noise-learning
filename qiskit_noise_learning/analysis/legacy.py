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

"""Legacy noise-model fitters retained as cross-check references."""

from collections import defaultdict
from typing import Any, Literal

import numpy as np
import scipy.optimize as opt
from numpy.typing import ArrayLike
from qiskit.quantum_info import (
    PauliLindbladMap,
    PauliList,
    QubitSparsePauli,
    QubitSparsePauliList,
)
from scipy.sparse import csr_array

from qiskit_noise_learning.analysis import AnalysisStage, Fit
from qiskit_noise_learning.data import AggregatedObservableData, ModelData
from qiskit_noise_learning.data.xarray_utils import time_bound
from qiskit_noise_learning.math import LinearMap
from qiskit_noise_learning.models import (
    GeneratorIndex,
    contains_pauli_lindblad_model,
    split_pauli_lindblad_model,
)
from qiskit_noise_learning.sequences import Path

from ..optionals import HAS_CVXPY

OptimizerLiteral = Literal["nnls", "lsq_linear_sparse", "cvxpy"]
NoiseAssumptionLiteral = Literal["symmetric_fidelities", "symmetric_generators"]


def get_fid_pairs(unbound_paths) -> tuple[QubitSparsePauliList, QubitSparsePauliList]:
    """Extract the first and second Paulis from the repeatable fragment of each unbound path.

    Args:
        unbound_paths.

    Returns:
        A pair ``(fid_ps_1, fid_ps_2)`` of ``QubitSparsePauliList`` objects holding,
        respectively, the first and second Pauli of each repeatable fragment.

    Raises:
        ValueError: If any unbound path's ``repeatable_fragment`` does not have exactly 2 entries,
            if its two entries are for different gates, or if traversing it would require
            single-qubit Cliffords.
    """
    fid_pairs = []

    for path in unbound_paths:
        fragment: Any = path.repeatable_fragment
        if len(fragment) != 2:
            raise ValueError(
                "Expected each unbound path's repeatable_fragment to have exactly 2 entries, "
                f"but got {len(fragment)}."
            )
        if fragment[0].gate_name != fragment[1].gate_name:
            raise ValueError(
                "Expected both entries of a repeatable fragment to be for the same gate, but got "
                f"'{fragment[0].gate_name}' and '{fragment[1].gate_name}'."
            )
        # Ensure single qubit Cliffords are not needed to repeat
        if (fragment[0].transition[1] != fragment[1].transition[0]) or (
            fragment[1].transition[1] != fragment[0].transition[0]
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
    aggregated_data: AggregatedObservableData,
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
        aggregated_data: The :class:`~.AggregatedObservableData` to fit.
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
        ValueError: If *noise_assumption* or *optimizer_name* is not recognized, if
            ``optimizer_name="nnls"`` and ``constrained=False``, if any observable is not a decay
            row, or if any pair fidelity is not positive.
        MissingOptionalLibraryError: If ``optimizer_name="cvxpy"`` and ``cvxpy`` is not
            installed.
    """
    fragment_depths = aggregated_data.dataset["fragment_depth"].data
    if np.any(fragment_depths != -1):
        depths = ", ".join(str(d) for d in sorted(set(fragment_depths[fragment_depths != -1])))
        raise ValueError(
            "Gate observables can only contain exponential decay data, but a fixed depth "
            f"observable was found, at fragment depth(s) {depths}."
        )

    fid_ps_1, fid_ps_2 = get_fid_pairs(aggregated_data.dataset.estimate_values.unbound_path.data)
    fid_pair_data = aggregated_data.dataset.estimate_values
    fidelities_canonical = make_canonical_fid_dict(
        fid_ps_1.to_pauli_list().to_labels(), fid_ps_2.to_pauli_list().to_labels(), fid_pair_data
    )
    pauli_fidelities = np.array(list(fidelities_canonical.values()))

    # Both noise assumptions take the logarithm of these. A fidelity above 1 is ordinary shot
    # noise and the non-negativity constraint absorbs it, but zero has no logarithm and a negative
    # fidelity has no square root, so neither can be fit.
    if np.any(pauli_fidelities <= 0):
        bad = [label for label, f in fidelities_canonical.items() if f <= 0]
        shown = ", ".join(bad[:10])
        if len(bad) > 10:
            shown += f", and {len(bad) - 10} more"
        raise ValueError(
            f"Pair fidelities must be positive, but {len(bad)} of {pauli_fidelities.size} are not, "
            f"for Pauli(s): {shown}."
        )
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
    # Against a one-element PauliList, Pauli.commutes returns a scalar rather than a length-1
    # array, so promote to 2D to keep a single-generator fit well-shaped.
    nc_array_basis = np.atleast_2d(
        np.logical_not([p.commutes(sparse_model_paulis) for p in basis_paulis])
    ).astype(int)
    nc_array_conj_basis = np.atleast_2d(
        np.logical_not([p.commutes(sparse_model_paulis) for p in conjugated_basis_paulis])
    ).astype(int)
    nc_array = nc_array_basis + nc_array_conj_basis

    # Set up the least-squares problem according to the selected assumption:
    if noise_assumption == "symmetric_fidelities":
        # assumption lets us compute the layer fidelities:
        layer_fidelities = np.tile(np.sqrt(pauli_fidelities), 2)
        nc_array_shaped = np.concatenate([nc_array_basis, nc_array_conj_basis])
        fit_vector = -np.log(layer_fidelities) / 2
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
        fit_vector = -np.log(pauli_fidelities) / 2
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


def _generator_key(generator: QubitSparsePauli) -> tuple:
    """A hashable stand-in for a generator, which is not itself hashable."""
    return tuple(generator.paulis), tuple(generator.indices), generator.num_qubits


def _validate_fitted_generators(model: LinearMap, indices: list[GeneratorIndex]) -> None:
    """Check that every generator this stage fit is one the model declares.

    A model that does not contain a :class:`~.PauliLindbladModel` has no generators to compare
    against and is left alone.

    Args:
        model: The fidelity model to check.
        indices: The generator indices this stage fit.

    Raises:
        ValueError: If any fitted generator is absent from the model.
    """
    if not contains_pauli_lindblad_model(model):
        return
    declared = {
        (name, _generator_key(generator))
        for name, generators in split_pauli_lindblad_model(model).model.generators.items()
        for generator in generators
    }

    undeclared = [
        index
        for index in indices
        if (index.gate_name, _generator_key(index.generator)) not in declared
    ]
    if undeclared:
        shown = ", ".join(f"{index.gate_name} {index.generator}" for index in undeclared[:5])
        if len(undeclared) > 5:
            shown += f", and {len(undeclared) - 5} more"
        raise ValueError(
            f"The observables determine {len(undeclared)} generator(s) that the model does not "
            f"have, so fitting them would report a model wider than the one given: {shown}."
        )


def _validate_spam_model(model: LinearMap, paths: list[Path]) -> None:
    r"""Check a model against the assumptions :func:`_spam_fit` makes about the noise it fits.

    Requires the preparation gate has no generators, and the measurement gate only has 1-local
    X generators.

    Args:
        model: The fidelity model to check.
        paths: The SPAM paths being fit.

    Raises:
        ValueError: If a preparation gate has any generators, or if a measurement gate has a
            generator on more than one qubit.
    """
    if not contains_pauli_lindblad_model(model):
        return
    generators = split_pauli_lindblad_model(model).model.generators

    for path in paths:
        prep_name = path.start_fragment[0].gate_name
        if len(generators.get(prep_name, ())) > 0:
            raise ValueError(
                f"SPAM observables attribute all of their noise to the measurement, so gate "
                f"'{prep_name}' must have no generators, but the model gives it "
                f"{len(generators[prep_name])}."
            )

        measurement = path.end_fragment[0]
        meas_generators = generators.get(measurement.gate_name)
        if meas_generators is None:
            continue

        wide = [g for g in meas_generators if len(g.indices) != 1]
        if wide:
            raise ValueError(
                f"SPAM observables are fit one qubit at a time, so gate "
                f"'{measurement.gate_name}' must only have single-qubit generators, but the model "
                f"gives it {len(wide)} on more than one qubit."
            )


def _spam_fit(
    path: Path, fidelity: float, fidelity_std: float
) -> tuple[GeneratorIndex, float, float]:
    r"""Fit the measurement generator rate a depth-0 SPAM path determines.

    A SPAM path measures :math:`F_P(Z_S) F_M(Z_S)`, one product of two unknowns, and this attributes
    all of it to the measurement gate: the rate is :math:`-\ln(F) / 2` for the single-qubit
    :math:`X` generator on the measured qubit, clipped at zero as the layer fit's non-negativity
    constraint would.

    The rate comes from a single row, so its variance is that row's uncertainty propagated through
    the same expression, :math:`(\sigma_F / 2 F)^2`. A rate clipped to zero sits on the
    non-negativity boundary, where :class:`~.ModelSolve` reports no variance, so it reports none
    either.

    Args:
        path: An unbound path with an empty repeatable fragment.
        fidelity: The path's measured fidelity.
        fidelity_std: The uncertainty on *fidelity*.

    Returns:
        The generator index, its rate, and the variance of that rate.

    Raises:
        ValueError: If the path does not have exactly one start and one end fragment entry, if its
            measurement does not act on exactly one qubit, or if *fidelity* is not positive.
    """
    if len(path.start_fragment) != 1 or len(path.end_fragment) != 1:
        raise ValueError(
            "SPAM observables must come from a path with one preparation and one measurement, but "
            f"got {len(path.start_fragment)} and {len(path.end_fragment)}."
        )

    measurement = path.end_fragment[0]
    if len(measurement.in_z_idxs) != 1:
        raise ValueError(
            "SPAM observables are fit one qubit at a time, but an observable on "
            f"{len(measurement.in_z_idxs)} qubits was found."
        )
    if fidelity <= 0:
        raise ValueError(f"SPAM fidelities must be positive, but got {fidelity}.")

    (qubit,) = measurement.in_z_idxs
    generator = QubitSparsePauli.from_sparse_label(
        ("X", [qubit]), num_qubits=measurement.pauli.num_qubits
    )
    index = GeneratorIndex(gate_name=measurement.gate_name, generator=generator)

    rate = -np.log(fidelity) / 2
    if rate < 0:
        return index, 0.0, 0.0
    return index, rate, (fidelity_std / (2 * fidelity)) ** 2


def _row_gate_name(path: Path) -> str:
    if len(path.repeatable_fragment) == 0:
        raise ValueError(
            "LegacySolve requires every observable to have a non-empty "
            "repeatable_fragment to determine its layer; encountered a path with an empty "
            "repeatable_fragment."
        )
    return path.repeatable_fragment[0].gate_name


class LegacySolve(AnalysisStage):
    """Solves for the :class:`~.ModelData` using the legacy pair-fidelity method, applied
    independently to each gate layer.

    For each distinct gate name found in the :class:`~.AggregatedObservableData`, this stage
    partitions the observable rows by that gate name, runs :func:`~.fit_noise_model_legacy`
    with ``noise_assumption="symmetric_fidelities"``, ``optimizer_name="nnls"``, and
    ``constrained=True``, then concatenates all per-layer results into a single
    :class:`~.ModelData`.

    Only generators estimated from the observable data are included in the output.
    Unestimated generators in the fit's model are omitted; their rates are not assumed to be zero.
    Model predictions requiring those missing parameters must be handled separately.

    If any layer violates the legacy-learner assumptions (an observable that is not a decay row,
    wrong repeatable-fragment length, a repeatable fragment spanning two gates, single-qubit
    Cliffords required, or inconsistent conjugate fidelities), the entire solve raises.  There is
    no per-layer skip or warning.

    Layer order in the output :class:`~.ModelData` follows first-seen order in the observable
    dataset, which is deterministic for a given :class:`~.AggregatedObservableData`.
    """

    input_level = AggregatedObservableData
    output_level = ModelData

    def _run(self, fit: Fit) -> None:
        aggregated_data = fit[AggregatedObservableData]
        dataset = aggregated_data.dataset

        paths = dataset["unbound_path"].data
        # A path with no repeatable fragment has no decay and no layer: it is a SPAM observable.
        spam_mask = np.array([len(path.repeatable_fragment) == 0 for path in paths], dtype=bool)
        layer_ds = dataset.sel({"observable": ~spam_mask})
        layer_paths = layer_ds["unbound_path"].data
        gate_names = np.array([_row_gate_name(path) for path in layer_paths], dtype=object)

        all_labels: list[GeneratorIndex] = []
        all_rates: list[float] = []
        all_time_lbs: list[np.datetime64] = []
        all_time_ubs: list[np.datetime64] = []
        # The gate fit is unweighted and reports no uncertainty; SPAM rates propagate theirs.
        all_variances: list[float] = []

        for name in dict.fromkeys(gate_names):
            mask = gate_names == name
            layer_data = AggregatedObservableData(layer_ds.sel({"observable": mask}))

            noise_map = fit_noise_model_legacy(
                layer_data,
                noise_assumption="symmetric_fidelities",
                decimals=None,
                optimizer_name="nnls",
                constrained=True,
            )

            layer_labels = [
                GeneratorIndex(gate_name=name, generator=g) for g in noise_map.generators()
            ]
            layer_rates = list(noise_map.rates)

            time_lb = time_bound(layer_data.dataset["time_lbs"].data, "min")
            time_ub = time_bound(layer_data.dataset["time_ubs"].data, "max")

            all_labels.extend(layer_labels)
            all_rates.extend(layer_rates)
            all_time_lbs.extend([time_lb] * len(layer_labels))
            all_time_ubs.extend([time_ub] * len(layer_labels))
            all_variances.extend([0.0] * len(layer_labels))

        spam_ds = dataset.sel({"observable": spam_mask})
        if spam_mask.any() and fit.model is not None:
            _validate_spam_model(fit.model, list(spam_ds["unbound_path"].data))

        for path, fidelity, fidelity_std, time_lb, time_ub in zip(
            spam_ds["unbound_path"].data,
            spam_ds["estimate_values"].data,
            spam_ds["estimate_std"].data,
            spam_ds["time_lbs"].data,
            spam_ds["time_ubs"].data,
        ):
            index, rate, variance = _spam_fit(path, float(fidelity), float(fidelity_std))
            all_labels.append(index)
            all_rates.append(rate)
            all_time_lbs.append(time_lb)
            all_time_ubs.append(time_ub)
            all_variances.append(variance)

        if fit.model is not None:
            _validate_fitted_generators(fit.model, all_labels)

        x = np.array(all_rates)
        cov_x = np.diag(all_variances)
        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=all_labels,
            parameter_values=x,
            covariance=cov_x,
            time_lbs=np.array(all_time_lbs, dtype="datetime64[us]"),
            time_ubs=np.array(all_time_ubs, dtype="datetime64[us]"),
            metadata={},
        )
