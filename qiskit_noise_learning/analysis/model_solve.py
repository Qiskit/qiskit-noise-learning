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

import numbers
import warnings
from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import numpy as np
import scipy.optimize as opt

from qiskit_noise_learning.analysis import AnalysisStage, Fit
from qiskit_noise_learning.data import AggregatedObservableData, ModelData
from qiskit_noise_learning.data.xarray_utils import time_bound
from qiskit_noise_learning.math import IndexedMatrix, IndexedVector
from qiskit_noise_learning.models import (
    contains_pauli_lindblad_model,
    split_pauli_lindblad_model,
)
from qiskit_noise_learning.optionals import HAS_CVXPY
from qiskit_noise_learning.sequences import LogPathMap, Path

RowIndex = TypeVar("RowIndex", bound=Hashable)
ColumnIndex = TypeVar("ColumnIndex", bound=Hashable)


@dataclass(frozen=True, eq=False)
class LinearSystemData(Generic[RowIndex, ColumnIndex]):
    """The linear system to solve and metadata in raw format.

    A linear system ``A @ x = b`` with axis labels and metadata.

    The row and column labels are of arbitrary hashable types: this class carries no assumptions
    about what a row or column denotes. In the systems built by :meth:`from_fit` the rows are
    :class:`~.Path` objects and the columns are the fidelity model's parameter labels.

    The index maps are the authoritative record of how labels correspond to positions in ``A``.
    Anything that needs to align a label-keyed quantity with the arrays should index through
    :attr:`row_index_map` or :attr:`column_index_map` rather than rebuilding the correspondence
    from :attr:`row_labels` or :attr:`column_labels`.

    Args:
        A: The matrix with shape ``(m, n)``.
        b: The target vector length ``m``.
        sigma_b: Statistical ``1``-sigma uncertainty on ``b`` per row, with length ``m``.
        row_diagnostics: Named per-row quantities recorded by the stages that produced ``b``, each
            an array of length ``m`` holding ``nan`` for rows the quantity is undefined for. Keys
            are the metadata names used upstream; see :meth:`from_fit`.
        row_index_map: A mapping from row labels to their integer row position in ``A``.
        column_index_map: A mapping from column labels to their integer column position in ``A``.
        time_lb: Earliest time bound across the rows.
        time_ub: Latest time bound across the rows.
    """

    A: np.ndarray
    b: np.ndarray
    sigma_b: np.ndarray
    row_diagnostics: Mapping[str, np.ndarray]
    row_index_map: Mapping[RowIndex, int]
    column_index_map: Mapping[ColumnIndex, int]
    time_lb: np.datetime64
    time_ub: np.datetime64

    @property
    def row_labels(self) -> list[RowIndex]:
        """Row labels, ordered by their row position in ``A``."""
        return sorted(self.row_index_map, key=self.row_index_map.__getitem__)

    @property
    def column_labels(self) -> list[ColumnIndex]:
        """Column labels, ordered by their column position in ``A``."""
        return sorted(self.column_index_map, key=self.column_index_map.__getitem__)

    @classmethod
    def from_fit(cls, fit: "Fit") -> "LinearSystemData[Path, Hashable]":
        """Build the linear system arrays from a :class:`~.Fit`.

        Rows are the :class:`~.Path` objects of the :class:`~.AggregatedObservableData`, columns are
        the fidelity model's parameter labels, and :attr:`row_diagnostics` holds every real-valued
        per-observable metadata entry under the name the producing stage used — for instance
        ``"reduced_chi_squared"`` from :class:`~.CurveFitObservables`.

        Warns:
            UserWarning: If any row's uncertainty is non-positive or non-finite, giving those rows'
                positions in :attr:`row_labels`. Such a row carries no usable statistical weight,
                and how it is treated is up to the solver.
        """
        dataset = fit[AggregatedObservableData].dataset
        fidelity_model = fit.model

        # Index the rows as "(unbound_path, fragment_depth) -> row position", -1 denotes unbound
        index_by_key: dict[tuple[Path, int], int] = {}
        for idx, key in enumerate(
            zip(dataset["unbound_path"].data, dataset["fragment_depth"].data)
        ):
            if key in index_by_key:
                raise ValueError(
                    f"ModelSolve assumes one entry per path, but a duplicate was found: {key}."
                )
            index_by_key[key] = idx

        # Resolve targets as (lookup_key, row_path) tuples.
        targets: Iterator[tuple[tuple[Path, int], Path]]
        if fit.paths:
            targets = (
                ((path, -1), path)
                if path.is_unbound
                else ((path.unbind(), path.fragment_depth), path)
                for path in fit.paths
            )
        else:
            targets = (
                (
                    (path, fragment_depth),
                    (path if fragment_depth == -1 else path.bind_at(fragment_depth)),
                )
                for path, fragment_depth in index_by_key
            )

        row_indices = []
        dataset_idxs = []

        for lookup_key, path in targets:
            if (idx := index_by_key.get(lookup_key)) is None:
                raise ValueError(
                    f"Required path-fragment-depth pair {lookup_key} missing from "
                    "AggregatedObservableData."
                )

            row_indices.append(path)
            dataset_idxs.append(idx)

        fidelities = dataset["estimate_values"].data[dataset_idxs]
        fidelity_stds = dataset["estimate_std"].data[dataset_idxs]
        metadatas = dataset["metadata"].data[dataset_idxs]
        time_lbs_list = dataset["time_lbs"].data[dataset_idxs]
        time_ubs_list = dataset["time_ubs"].data[dataset_idxs]

        # The design matrix maps the model's input parameters to path log-fidelities: compose the
        # fidelity model with the path linearization, then materialize the rows for the paths.
        path_model = LogPathMap(fidelity_model.output_space) @ fidelity_model
        design_matrix = path_model.rows(row_indices)

        # Build b, sigma_b and the diagnostics aligned with the design matrix's row order. add_rows
        # may drop all-zero rows, so iterate row_index_map, the surviving rows.
        row_data = dict(zip(row_indices, zip(fidelities, fidelity_stds, metadatas)))
        n_rows = len(design_matrix.row_index_map)
        b = np.empty(n_rows, dtype=float)
        sigma_b = np.empty(n_rows, dtype=float)
        row_diagnostics: dict[str, np.ndarray] = {}
        for row_index, array_idx in design_matrix.row_index_map.items():
            fidelity, fidelity_std, meta = row_data[row_index]
            b[array_idx] = -np.log(max(fidelity, 1e-300))
            sigma_b[array_idx] = fidelity_std / max(fidelity, 1e-300)

            # Carry every real-valued metadata entry through under its own name, rather than
            # hard-coding the ones a policy happens to want. Rows without an entry keep the nan the
            # array was created with, so a key's array covers all rows or says where it does not.
            if isinstance(meta, Mapping):
                for key, value in meta.items():
                    if isinstance(value, numbers.Real):
                        if key not in row_diagnostics:
                            row_diagnostics[key] = np.full(n_rows, np.nan, dtype=float)
                        row_diagnostics[key][array_idx] = float(value)

        # Warn once, here at the single point where data enters the solvers, so that every solver
        # and every constraint policy sees the same notion of an unusable row and none of them has
        # to report it. Report row positions rather than the labels themselves: a Path repr spans
        # many lines, and positions index row_labels so the caller can still identify the rows.
        unusable = np.flatnonzero(~np.isfinite(sigma_b) | (sigma_b <= 0))
        if unusable.size:
            shown = ", ".join(str(idx) for idx in unusable[:10])
            if unusable.size > 10:
                shown += f", and {unusable.size - 10} more"
            warnings.warn(
                f"{unusable.size} of {n_rows} row(s) of the linear system have a non-positive or "
                "non-finite uncertainty, so they carry no usable statistical weight. Positions in "
                f"LinearSystemData.row_labels: {shown}.",
                stacklevel=2,
            )

        all_time_lbs = np.array(time_lbs_list, dtype="datetime64[us]")
        all_time_ubs = np.array(time_ubs_list, dtype="datetime64[us]")
        time_lb = time_bound(all_time_lbs, "min")
        time_ub = time_bound(all_time_ubs, "max")

        return cls(
            A=design_matrix.data,
            b=b,
            sigma_b=sigma_b,
            row_diagnostics=row_diagnostics,
            row_index_map=dict(design_matrix.row_index_map),
            column_index_map=dict(design_matrix.column_index_map),
            time_lb=time_lb,
            time_ub=time_ub,
        )


class ModelSolve(AnalysisStage):
    """Base class for model fitting routines.

    Constructs the design matrix from the :class:`~.FidelityModel` stored on the :class:`~.Fit`
    container and the paths in the :class:`~.AggregatedObservableData`. Then solves ``A @ x = b``
    using a specified method, where ``A`` is the design matrix and ``b`` is the vector of negative
    log observables ``-log(o)``.

    If paths are specified on the :class:`~.Fit`, only data matching those paths is used. If no
    paths are specified, all data in the :class:`~.AggregatedObservableData` is used.

    This stage assumes a single observable value for each unique :class:`Path`.
    """

    @property
    def input_level(self):
        return AggregatedObservableData

    @property
    def output_level(self):
        return ModelData

    @abstractmethod
    def _solve(self, system: LinearSystemData) -> tuple[np.ndarray, np.ndarray, dict]:
        """Numerical method for solving the linear system.

        Args:
            system: The linear system to solve.

        Returns:
            A tuple of ``(x, cov_x, metadata)``.
        """

    def _run(self, fit: Fit):
        system = LinearSystemData.from_fit(fit)
        x, cov_x, metadata = self._solve(system)

        # add standard fields to metadata
        residual_vec = system.A @ x - system.b
        metadata["residual"] = np.linalg.norm(residual_vec)
        metadata["path_residual"] = IndexedVector(
            {path: val for path, val in zip(system.row_labels, np.abs(residual_vec))}
        )

        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=system.column_labels,
            parameter_values=x,
            covariance=cov_x,
            time_lbs=np.full(len(x), system.time_lb, dtype="datetime64[us]"),
            time_ubs=np.full(len(x), system.time_ub, dtype="datetime64[us]"),
            metadata=metadata,
        )

    @staticmethod
    def _covariance(
        A: np.ndarray, sigma_b: np.ndarray, x: np.ndarray, free_indices: np.ndarray
    ) -> np.ndarray:
        """Pinv-based error propagation covariance for a solution x.

        Args:
            A: The design matrix.
            sigma_b: Uncertainty on b.
            x: The solution vector.
            free_indices: Indices of parameters not on a constraint boundary.
        """
        n = len(x)
        cov_x = np.zeros((n, n))
        if free_indices.size > 0:
            cov_b = np.diag(sigma_b**2)
            A_S = A[:, free_indices]
            A_S_pinv = np.linalg.pinv(A_S)
            cov_x_S = A_S_pinv @ cov_b @ A_S_pinv.T
            cov_x[np.ix_(free_indices, free_indices)] = cov_x_S
        return cov_x


class NNLSSolve(ModelSolve):
    """Solves for the :class:`~.ModelData` using SciPy's non-negative least squares solver.

    See SciPy's
    `documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html>`_
    for details on the method. See :class:`~.ModelSolve` for more details about the general
    responsibility of a model solver in this library.

    Args:
        **nnls_opts: The options passed on to the SciPy solver.
    """

    def __init__(self, **nnls_opts):
        self.nnls_opts = nnls_opts

    def _solve(self, system: LinearSystemData) -> tuple[np.ndarray, np.ndarray, dict]:
        x, _ = opt.nnls(system.A, system.b, **self.nnls_opts)
        free_indices = np.where(x > 0)[0]
        cov_x = self._covariance(system.A, system.sigma_b, x, free_indices)
        return x, cov_x, dict()


class LSQLinearSolve(ModelSolve):
    """Solves for the :class:`~.ModelData` using SciPy's linear least squares solver.

    See SciPy's
    `documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html>`_
    for details on the method. See :class:`~.ModelSolve` for more details about the general
    responsibility of a model solver in this library.

    Args:
        **lsq_linear_opts: The options passed on to the SciPy solver.
    """

    def __init__(self, **lsq_linear_opts):
        self.lsq_linear_opts = lsq_linear_opts
        self.lsq_linear_opts.setdefault("bounds", (0, np.inf))
        self.lsq_linear_opts.setdefault("method", "bvls")

    def _solve(self, system: LinearSystemData) -> tuple[np.ndarray, np.ndarray, dict]:
        opt_res = opt.lsq_linear(system.A, system.b, **self.lsq_linear_opts)
        x = opt_res.x

        lb, ub = self.lsq_linear_opts["bounds"]
        at_lower = np.isfinite(lb) & np.isclose(x, lb)
        at_upper = np.isfinite(ub) if np.isscalar(ub) else np.isfinite(ub) & np.isclose(x, ub)
        free_indices = np.where(~at_lower & ~at_upper)[0]
        cov_x = self._covariance(system.A, system.sigma_b, x, free_indices)

        return x, cov_x, {"opt_res": opt_res}


# A constraint policy computes a constraint value from the solve-time linear system, so a bound can
# depend on how trustworthy each row's data is. The policy built by
# ``PositivityMinSolve.from_data_scaled_deltas`` is one such delta policy; users may supply their
# own. Fixed literals are wrapped into constant policies by ``PositivityMinSolve.from_constants``.
EpsilonPolicy = Callable[[LinearSystemData], float]
DeltaPolicy = Callable[[LinearSystemData], Mapping[Hashable, float]]
WeightsPolicy = Callable[[LinearSystemData], IndexedMatrix]


def _validate_policy(policy: object, name: str):
    """Check that a constraint bound was supplied as a policy rather than as a fixed value.

    Args:
        policy: The policy to check. ``None`` passes, meaning the constraint is not applied.
        name: Name of the constructor argument, used in the error message.

    Raises:
        TypeError: If ``policy`` is neither ``None`` nor callable.
    """
    if policy is not None and not callable(policy):
        raise TypeError(
            f"The '{name}' argument must be a callable policy accepting a LinearSystemData, but "
            f"got {type(policy).__name__}. To specify a fixed bound, use "
            "PositivityMinSolve.from_constants instead."
        )


def _validate_policy_labels(
    labels: Iterable[Hashable],
    row_index_map: Mapping[Hashable, int],
    name: str,
    require_complete: bool = True,
):
    """Check the labels a constraint policy produced against the rows of the linear system.

    Args:
        labels: The row labels the policy supplied a value for.
        row_index_map: The linear system's row index map.
        name: Name of the policy output, used in the error messages.
        require_complete: Whether every row of the linear system must appear in ``labels``.

    Raises:
        ValueError: If ``labels`` contains a label that is not a row of the linear system.
        ValueError: If ``require_complete`` and a row of the linear system is absent from
            ``labels``.
    """
    labels = set(labels)
    rows = set(row_index_map)

    if unknown := labels - rows:
        raise ValueError(
            f"The '{name}' policy produced {len(unknown)} label(s) that are not rows of the linear "
            f"system, for example '{next(iter(unknown))}'. Note that rows whose design matrix "
            "entries are all zero are dropped from the system."
        )

    if require_complete and (missing := rows - labels):
        raise ValueError(
            f"The '{name}' policy did not produce a value for {len(missing)} row(s) of the linear "
            f"system, for example '{next(iter(missing))}'."
        )


class PositivityMinSolve(ModelSolve):
    r"""Solves for the :class:`~.ModelData` while minimizing Pauli-Lindblad rate positivity.

    Requires that the :class:`~.Fit` uses a :class:`~.PauliLindbladModel`.

    For a gate set :math:`\mathcal{G}`, let :math:`\{r_{G, P}\}` denote the Pauli-Lindblad rates
    over gate-dependent generator sets :math:`\mathcal{K}(G)`, :math:`A` the design matrix and
    :math:`b` the observed data. For the user-specified algorithm parameters:

    - Gate coefficients :math:`\{c_G \in \mathbb{R} : G \in \mathcal{G}\}`,
    - Global fit bound :math:`\epsilon > 0`, and
    - Local fit bounds :math:`\delta_P` for each path :math:`P` measured in the design matrix,

    this class solves the convex optimization problem:

    .. math::

        \min \sum_{G \in \mathcal{G}} c_G \sum_{P \in \mathcal{K}(G)} \max(0, r_{P, G})

    subject to:

    - :math:`\|W (A r - b)\|_2 \leq \epsilon`
    - :math:`|(Ar - b)_i| \leq \delta_i` for each row :math:`i`
    - :math:`r \geq 0` (optional)

    See :class:`~.ModelSolve` for more details about the general responsibility of a model solver
    in this library.

    The constraint bounds are specified as **policies**: callables that receive the solve-time
    :class:`~.LinearSystemData` and return the corresponding bound.

    Args:
        coefficients: Per-gate coefficients for the objective function, as a mapping from gate
            name to float.
        epsilon: Policy returning the tolerance for the overall weighted L2 norm constraint. At
            least one of ``epsilon`` or ``deltas`` must be provided.
        deltas: Policy returning per-row tolerances as a mapping from :class:`~.Path` to float. It
            must cover every row of the linear system, since a row absent from the mapping would be
            left unconstrained. At least one of ``epsilon`` or ``deltas`` must be provided.
        weights: Policy returning the weight matrix ``W`` for the L2 constraint as an
            :class:`~.IndexedMatrix` whose row and column indices are :class:`~.Path` objects.
            Defaults to identity. A row or column of the linear system absent from ``W`` is
            weighted zero, which drops its residual from the norm. Only used when ``epsilon`` is
            provided.
        non_negative: Whether to enforce ``x >= 0``.

    Raises:
        TypeError: If ``epsilon``, ``deltas`` or ``weights`` is given as a fixed value rather than
            a policy. Use :meth:`from_constants` for fixed values.
        ValueError: At solve time, if the ``deltas`` or ``weights`` policy produces a label that is
            not a row of the linear system, or if ``deltas`` omits one.
        RuntimeError: At solve time, if the convex solver does not return a solution, for instance
            because the constraints are infeasible.
    """

    def __init__(
        self,
        coefficients: dict[str, float],
        epsilon: EpsilonPolicy | None = None,
        deltas: DeltaPolicy | None = None,
        weights: WeightsPolicy | None = None,
        non_negative: bool = False,
    ):
        _validate_policy(epsilon, "epsilon")
        _validate_policy(deltas, "deltas")
        _validate_policy(weights, "weights")

        if epsilon is None and deltas is None:
            raise ValueError("At least one of 'epsilon' or 'deltas' must be provided.")

        self.coefficients = coefficients
        self.epsilon = epsilon
        self.deltas = deltas
        self.weights = weights
        self.non_negative = non_negative

    @classmethod
    def from_constants(
        cls,
        coefficients: dict[str, float],
        epsilon: float | None = None,
        deltas: Mapping[Path, float] | None = None,
        weights: IndexedMatrix | None = None,
        non_negative: bool = False,
    ) -> Self:
        """Construct from fixed constant bounds instead of data-driven policies.

        Each supplied constant is wrapped in a policy that ignores the data and returns it. See
        the class docstring for the meaning of each argument; here they are fixed values rather
        than callables.
        """
        return cls(
            coefficients,
            epsilon=None if epsilon is None else (lambda system, value=epsilon: value),
            deltas=None if deltas is None else (lambda system, value=deltas: value),
            weights=None if weights is None else (lambda system, value=weights: value),
            non_negative=non_negative,
        )

    @classmethod
    def from_data_scaled_deltas(
        cls,
        coefficients: dict[str, float],
        scale: float = 1.0,
        non_negative: bool = False,
    ) -> Self:
        r"""Build with a delta-only policy based on statistical uncertainty of observables.

        Each row's tolerance is set from that row's own statistical uncertainty and its
        exponential-fit goodness-of-fit. For row :math:`i`, with statistical ``1``-sigma
        ``sigma_b`` and the reduced chi-squared ``chi2_red`` read from the
        ``"reduced_chi_squared"`` entry of :attr:`~.LinearSystemData.row_diagnostics`:

        .. math::

            \mathrm{inflation}_i &= \max(1, \sqrt{\mathrm{chi2\_red}_i}) \\
            \delta_i &= \mathrm{scale} \cdot \mathrm{inflation}_i \cdot \sigma_{b, i}

        Setting ``delta_i`` proportional to ``sigma`` follows the Morozov discrepancy principle
        (allow about ``scale`` standard deviations of slack). Because ``curve_fit`` reports
        ``sigma_b`` with ``absolute_sigma=True``, it is blind to model mismatch; the
        ``sqrt(chi2_red)`` factor loosens rows whose exponential fit is poor, and the ``max(1, .)``
        clamp means mismatch can only loosen a row, never tighten it below its statistical
        uncertainty. Rows with an undefined ``chi2_red`` (``nan``, e.g. averaged rows), and every
        row when the system carries no ``"reduced_chi_squared"`` diagnostic at all, get no
        inflation. If the model cannot be fit within the resulting tolerances the solve reports an
        infeasible problem, and ``scale`` is the knob that loosens every row at once.

        A row whose uncertainty is non-positive or non-finite carries no statistical information to
        scale by, so rather than being treated as an extremely precise row it is assigned the
        median tolerance of the rows that do have a usable uncertainty. This keeps such a row in
        the fit at a typical scale without letting it constrain the solution as a near-equality,
        and no row's tolerance ever depends on another row's uncertainty except through this
        substitution.

        Args:
            coefficients: Per-gate coefficients for the objective function, as a mapping from gate
                name to float.
            scale: Multiplier on the per-row tolerance, in units of (inflated) standard deviations.
            non_negative: Whether to enforce ``x >= 0``.

        Raises:
            ValueError: At solve time, if no row of the linear system has a positive, finite
                uncertainty, leaving nothing to derive tolerances from.
        """

        def deltas(system: LinearSystemData) -> dict[Hashable, float]:
            chi2_red = system.row_diagnostics.get("reduced_chi_squared")
            if chi2_red is None:
                inflation = 1.0
            else:
                inflation = np.where(np.isfinite(chi2_red), np.sqrt(np.maximum(chi2_red, 1.0)), 1.0)
            effective = inflation * system.sigma_b

            usable = np.isfinite(effective) & (effective > 0)
            if not usable.any():
                raise ValueError(
                    f"None of the {len(effective)} rows of the linear system has a positive, "
                    "finite uncertainty, so no residual tolerances can be derived from the data. "
                    "Supply fixed tolerances with PositivityMinSolve.from_constants instead."
                )
            values = scale * np.where(usable, effective, np.median(effective[usable]))
            return dict(zip(system.row_labels, values))

        return cls(coefficients, deltas=deltas, non_negative=non_negative)

    def _run(self, fit: Fit):
        if not contains_pauli_lindblad_model(fit.model):
            raise TypeError(
                "PositivityMinSolve requires a model containing a PauliLindbladModel, "
                f"but got {type(fit.model).__name__}."
            )
        if split_pauli_lindblad_model(fit.model).before is not None:
            raise NotImplementedError(
                "PositivityMinSolve does not yet support models with maps applied before the "
                "PauliLindbladModel (the fit parameters would not be the Pauli-Lindblad rates)."
            )
        super()._run(fit)

    def _solve(self, system: LinearSystemData) -> tuple[np.ndarray, np.ndarray, dict]:
        HAS_CVXPY.require_now("PositivityMinSolve")
        import cvxpy as cp

        A, b = system.A, system.b
        column_labels, row_labels = system.column_labels, system.row_labels
        row_index_map = system.row_index_map

        n = A.shape[1]
        m = A.shape[0]

        # Resolve each constraint policy against the solve-time data.
        epsilon = self.epsilon(system) if self.epsilon is not None else None
        deltas = self.deltas(system) if self.deltas is not None else None
        weights = self.weights(system) if self.weights is not None else None

        # Build coefficient vector from gate-name mapping
        coeff_vector = np.array([self.coefficients[label.gate_name] for label in column_labels])

        x = cp.Variable(n)
        objective = cp.Minimize(coeff_vector @ cp.pos(x))

        residual = A @ x - b

        constraints = []

        if epsilon is not None:
            if weights is not None:
                # A row or column absent from the weight matrix is weighted zero, dropping its
                # residual from the norm. Unknown labels are a misconfiguration.
                _validate_policy_labels(
                    weights.row_index_map, row_index_map, "weights rows", require_complete=False
                )
                _validate_policy_labels(
                    weights.column_index_map,
                    row_index_map,
                    "weights columns",
                    require_complete=False,
                )
                w = np.zeros((m, m))
                for row_label, row_idx in weights.row_index_map.items():
                    for col_label, col_idx in weights.column_index_map.items():
                        w[row_index_map[row_label], row_index_map[col_label]] = weights.data[
                            row_idx, col_idx
                        ]
                weighted_residual = w @ residual
            else:
                weighted_residual = residual
            constraints.append(cp.norm(weighted_residual, 2) <= epsilon)

        if deltas is not None:
            # Every row needs a bound: a row absent from the mapping would be left unconstrained.
            _validate_policy_labels(deltas, row_index_map, "deltas")
            deltas_array = np.array([deltas[row_label] for row_label in row_labels])
            constraints.append(cp.abs(residual) <= deltas_array)

        if self.non_negative:
            constraints.append(x >= 0)

        problem = cp.Problem(objective, constraints)
        problem.solve()

        # cvxpy reports failure through the status rather than by raising, and leaves the variable
        # unpopulated, so catch it here instead of letting a None propagate into the residuals.
        if problem.status not in cp.settings.SOLUTION_PRESENT or x.value is None:
            raise RuntimeError(
                f"The convex solve did not produce a solution (cvxpy status '{problem.status}'). "
                "An infeasible status usually means the residual bounds are too tight for the "
                "data; consider loosening 'deltas' or 'epsilon'."
            )
        if problem.status == cp.OPTIMAL_INACCURATE:
            warnings.warn(
                "The convex solve converged only to reduced accuracy (cvxpy status "
                f"'{problem.status}'); the returned rates may be unreliable.",
                stacklevel=2,
            )

        metadata = {"problem": problem}
        if epsilon is not None:
            metadata["weighted_residual"] = np.linalg.norm(weighted_residual.value)

        return x.value, np.zeros((n, n)), metadata
