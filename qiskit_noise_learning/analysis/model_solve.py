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

from abc import abstractmethod
from collections.abc import Iterator

import numpy as np
import scipy.optimize as opt

from qiskit_noise_learning.analysis import AnalysisStage, Fit
from qiskit_noise_learning.data import AveragedData, ModelData
from qiskit_noise_learning.data.xarray_utils import time_bound
from qiskit_noise_learning.math import IndexedMatrix, IndexedVector
from qiskit_noise_learning.models import (
    contains_pauli_lindblad_model,
    split_pauli_lindblad_model,
)
from qiskit_noise_learning.optionals import HAS_CVXPY
from qiskit_noise_learning.sequences import LogPathMap, Path


class ModelSolve(AnalysisStage):
    """Base class for model fitting routines.

    Constructs the design matrix from the :class:`~.FidelityModel` stored on the :class:`~.Fit`
    container and the paths in the :class:`~.AveragedData`. Then solves ``A @ x = b`` using a
    specified method, where ``A`` is the design matrix and ``b`` is the vector of negative log
    observables ``-log(o)``.

    If paths are specified on the :class:`~.Fit`, only data matching those paths is used. If no
    paths are specified, all data in the :class:`~.AveragedData` is used.

    This stage assumes a single observable value for each unique :class:`Path`.
    """

    @property
    def input_level(self):
        return AveragedData

    @property
    def output_level(self):
        return ModelData

    @abstractmethod
    def _solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        sigma_b: np.ndarray,
        param_labels: list,
        path_labels: list[Path],
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Numerical method for solving the linear system.

        Args:
            A: The design matrix with shape ``(m, n)``.
            b: The target vector with length ``m``.
            sigma_b: Uncertainty on ``b`` with length ``m``.
            param_labels: Ordered parameter labels corresponding to columns of ``A``.
            path_labels: Ordered path labels corresponding to rows of ``A``.

        Returns:
            A tuple of ``(x, cov_x, metadata)``.
        """

    def _run(self, fit: Fit):
        A, b, sigma_b, param_labels, path_labels, time_lb, time_ub = self._build_linear_system(fit)
        x, cov_x, metadata = self._solve(A, b, sigma_b, param_labels, path_labels)

        # add standard fields to metadata
        residual_vec = A @ x - b
        metadata["residual"] = np.linalg.norm(residual_vec)
        metadata["path_residual"] = IndexedVector(
            {path: val for path, val in zip(path_labels, np.abs(residual_vec))}
        )

        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=param_labels,
            parameter_values=x,
            covariance=cov_x,
            time_lbs=np.full(len(x), time_lb, dtype="datetime64[us]"),
            time_ubs=np.full(len(x), time_ub, dtype="datetime64[us]"),
            metadata=metadata,
        )

    def _build_linear_system(
        self, fit: Fit
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list[Path], np.datetime64, np.datetime64]:
        """Build the linear system arrays from a :class:`~.Fit`.

        Returns:
            A tuple of ``(A, b, sigma_b, param_labels, path_labels, time_lb, time_ub)``.
        """
        dataset = fit[AveragedData].dataset
        fidelity_model = fit.model

        # Index the as "(unbound_path, depth) -> row position", where -1 denotes unbound
        index_by_key: dict[tuple[Path, int], int] = {}
        for idx, key in enumerate(zip(dataset["unbound_path"].data, dataset["depth"].data)):
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
                else ((path.without_depth(), path.depth), path)
                for path in fit.paths
            )
        else:
            targets = (
                ((path, depth), (path if depth == -1 else path.bind_at(depth)))
                for path, depth in index_by_key
            )

        row_indices = []
        dataset_idxs = []

        for lookup_key, path in targets:
            if (idx := index_by_key.get(lookup_key)) is None:
                raise ValueError(
                    f"Required path-depth pair {lookup_key} missing from AveragedData."
                )

            row_indices.append(path)
            dataset_idxs.append(idx)

        fidelities = dataset["observables"].data[dataset_idxs]
        fidelity_stds = dataset["std"].data[dataset_idxs]
        time_lbs_list = dataset["time_lbs"].data[dataset_idxs]
        time_ubs_list = dataset["time_ubs"].data[dataset_idxs]

        # The design matrix maps the model's input parameters to path log-fidelities: compose the
        # fidelity model with the path linearization, then materialize the rows for the paths.
        path_model = LogPathMap(fidelity_model.output_space) @ fidelity_model
        design_matrix = path_model.rows(row_indices)

        # Build b and sigma_b aligned with the design matrix's row order. add_rows may drop all-zero
        # rows, so iterate row_index_map, the surviving rows.
        fidelity_by_row = dict(zip(row_indices, zip(fidelities, fidelity_stds)))
        n_rows = len(design_matrix.row_index_map)
        b = np.empty(n_rows, dtype=float)
        sigma_b = np.empty(n_rows, dtype=float)
        for row_index, array_idx in design_matrix.row_index_map.items():
            fidelity, fidelity_std = fidelity_by_row[row_index]
            b[array_idx] = -np.log(max(fidelity, 1e-300))
            sigma_b[array_idx] = fidelity_std / max(fidelity, 1e-300)

        all_time_lbs = np.array(time_lbs_list, dtype="datetime64[us]")
        all_time_ubs = np.array(time_ubs_list, dtype="datetime64[us]")
        time_lb = time_bound(all_time_lbs, "min")
        time_ub = time_bound(all_time_ubs, "max")

        param_labels = sorted(x := design_matrix.column_index_map, key=x.get)
        path_labels = sorted(x := design_matrix.row_index_map, key=x.get)

        return design_matrix.data, b, sigma_b, param_labels, path_labels, time_lb, time_ub

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
    [documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html)
    for details on the method. See :class:`~.ModelSolve` for more details about the general
    responsibility of a model solver in this library.

    Args:
        **nnls_opts: The options passed on to the SciPy solver.
    """

    def __init__(self, **nnls_opts):
        self.nnls_opts = nnls_opts

    def _solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        sigma_b: np.ndarray,
        param_labels: list,
        path_labels: list[Path],
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        x, _ = opt.nnls(A, b, **self.nnls_opts)
        free_indices = np.where(x > 0)[0]
        cov_x = self._covariance(A, sigma_b, x, free_indices)
        return x, cov_x, dict()


class LSQLinearSolve(ModelSolve):
    """Solves for the :class:`~.ModelData` using SciPy's linear least squares solver.

    See SciPy's
    [documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html)
    for details on the method. See :class:`~.ModelSolve` for more details about the general
    responsibility of a model solver in this library.

    Args:
        **lsq_linear_opts: The options passed on to the SciPy solver.
    """

    def __init__(self, **lsq_linear_opts):
        self.lsq_linear_opts = lsq_linear_opts
        self.lsq_linear_opts.setdefault("bounds", (0, np.inf))
        self.lsq_linear_opts.setdefault("method", "bvls")

    def _solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        sigma_b: np.ndarray,
        param_labels: list,
        path_labels: list[Path],
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        opt_res = opt.lsq_linear(A, b, **self.lsq_linear_opts)
        x = opt_res.x

        lb, ub = self.lsq_linear_opts["bounds"]
        at_lower = np.isfinite(lb) & np.isclose(x, lb)
        at_upper = np.isfinite(ub) if np.isscalar(ub) else np.isfinite(ub) & np.isclose(x, ub)
        free_indices = np.where(~at_lower & ~at_upper)[0]
        cov_x = self._covariance(A, sigma_b, x, free_indices)

        return x, cov_x, {"opt_res": opt_res}


class PositivityMinSolve(ModelSolve):
    r"""Solves for the :class:`~.ModelData` while minimizing Pauli-Lindblad rate positivity.

    Requires that the :class:`~.Fit` uses a :class:`~.PauliLindbladModel`.

    For a gate set :math:`\mathcal{G}`, let :math:`\{r_{G, P}}` denote the Pauli-Lindblad rates over
    gate-dependent generator sets \mathcal{K}(G)`, :math:`A` the design matrix and :math:`b` the
    observed data. For the user-specified algorithm parameters:
    - Gate coefficients :math:`\{c_G \in \mathbb{R} : G \in \mathcal{G}\}`,
    - Global fit bound :math:`\epsilon > 0`, and
    - Local fit bounds :math:`\delta_P` for each path :math:`P` measured in the design matrix,

    this class solves the convex optimization problem:

    .. math::

        \\min \\sum_{G \in \mathcal{G}} c_G \sum_{P \in \mathcal{K}(G)} \\max(0, r_{P, G})

    subject to:

    - :math:`\\|W (A r - b)\\|_2 \\leq \\epsilon`
    - :math:`|(Ar - b)_i| \\leq \\delta_i` for each row :math:`i`
    - :math:`r \\geq 0` (optional)

    See :class:`~.ModelSolve` for more details about the general responsibility of a model solver
    in this library.

    Args:
        coefficients: Per-gate coefficients for the objective function, as a mapping from gate
            name to float.
        epsilon: Tolerance for the overall weighted L2 norm constraint. At least one of
            ``epsilon`` or ``deltas`` must be provided.
        deltas: Per-row tolerances as a mapping from :class:`~.Path` to float. At least one of
            ``epsilon`` or ``deltas`` must be provided.
        weights: Weight matrix ``W`` for the L2 constraint as an :class:`~.IndexedMatrix` whose
            row and column indices are :class:`~.Path` objects. Defaults to identity. Only used
            when ``epsilon`` is provided.
        non_negative: Whether to enforce ``x >= 0``.
    """

    def __init__(
        self,
        coefficients: dict[str, float],
        epsilon: float | None = None,
        deltas: dict[Path, float] | None = None,
        weights: IndexedMatrix | None = None,
        non_negative: bool = False,
    ):
        if epsilon is None and deltas is None:
            raise ValueError("At least one of 'epsilon' or 'deltas' must be provided.")

        self.coefficients = coefficients
        self.epsilon = epsilon
        self.deltas = deltas
        self.weights = weights
        self.non_negative = non_negative

    def _run(self, fit: Fit):
        if not contains_pauli_lindblad_model(fit.model):
            raise TypeError(
                "PositivityMinSolve requires a model containing a PauliLindbladModel, "
                f"but got {type(fit.model).__name__}."
            )
        if split_pauli_lindblad_model(fit.model)[2] is not None:
            raise NotImplementedError(
                "PositivityMinSolve does not yet support models with maps applied before the "
                "PauliLindbladModel (the fit parameters would not be the Pauli-Lindblad rates)."
            )
        super()._run(fit)

    def _solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        sigma_b: np.ndarray,
        param_labels: list,
        path_labels: list[Path],
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        HAS_CVXPY.require_now("PositivityMinSolve")
        import cvxpy as cp

        n = A.shape[1]
        m = A.shape[0]

        # Build coefficient vector from gate-name mapping
        coeff_vector = np.array([self.coefficients[label.gate_name] for label in param_labels])

        x = cp.Variable(n)
        objective = cp.Minimize(coeff_vector @ cp.pos(x))

        residual = A @ x - b

        constraints = []

        if self.epsilon is not None:
            if self.weights is not None:
                path_to_row = {path: i for i, path in enumerate(path_labels)}
                w = np.zeros((m, m))
                for row_label, row_idx in self.weights.row_index_map.items():
                    for col_label, col_idx in self.weights.column_index_map.items():
                        w[path_to_row[row_label], path_to_row[col_label]] = self.weights.data[
                            row_idx, col_idx
                        ]
                weighted_residual = w @ residual
            else:
                weighted_residual = residual
            constraints.append(cp.norm(weighted_residual, 2) <= self.epsilon)

        if self.deltas is not None:
            deltas_array = np.array([self.deltas[path] for path in path_labels])
            constraints.append(cp.abs(residual) <= deltas_array)

        if self.non_negative:
            constraints.append(x >= 0)

        problem = cp.Problem(objective, constraints)
        problem.solve()

        return (
            x.value,
            np.zeros((n, n)),
            {
                "problem": problem,
                "weighted_residual": np.linalg.norm(weighted_residual.value),
            },
        )
