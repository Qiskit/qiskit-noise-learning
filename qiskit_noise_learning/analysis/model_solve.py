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

import numpy as np
import scipy.optimize as opt

from qiskit_noise_learning.analysis import AnalysisStage, Fit
from qiskit_noise_learning.data import AveragedData, ModelData
from qiskit_noise_learning.data.xarray_utils import time_bound
from qiskit_noise_learning.math import IndexedMatrix


class ModelSolve(AnalysisStage):
    """Base class for finding model parameters.

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
    def _linear_solve(self, a_mat: np.ndarray, b_vec: np.ndarray) -> tuple[np.ndarray, dict]:
        """Perform the linear inversion ``a_mat \\ b_vec``, somehow.

        Args:
            a_mat: The matrix to invert by, not necessarily invertible.
            b_vec: The vector to invert.

        Returns:
           The inverted vector and any dictionary of metadata.
        """

    def _run(self, fit: Fit):
        dataset = fit[AveragedData].dataset
        fidelity_model = fit.model

        if fit.paths:
            target_pairs = set()
            for path in fit.paths:
                if path.is_unbound:
                    target_pairs.add((path, -1))
                else:
                    target_pairs.add((path.without_depth(), path.depth))
        else:
            target_pairs = set(zip(dataset["unbound_path"].data, dataset["depth"].data))

        row_indices = []
        rows = []
        fidelities = []
        fidelity_stds = []
        time_lbs_list = []
        time_ubs_list = []

        for unbound_path, depth in target_pairs:
            mask = (dataset["unbound_path"].data == unbound_path) & (dataset["depth"].data == depth)
            if not mask.any():
                continue

            if mask.sum() > 1:
                raise ValueError(
                    "ModelSolve assumes only one entry per path, but a duplicate path was found."
                )

            fidelity = float(dataset["observables"].data[mask][0])
            fidelity_std = float(dataset["std"].data[mask][0])

            if depth == -1:
                row_index = unbound_path
                row = fidelity_model.row_from_path(unbound_path)
            else:
                bound_path = unbound_path.bind_at(depth)
                row_index = bound_path
                row = fidelity_model.row_from_path(bound_path)

            row_indices.append(row_index)
            rows.append(row)
            fidelities.append(fidelity)
            fidelity_stds.append(fidelity_std)
            time_lbs_list.append(dataset["time_lbs"].data[mask][0])
            time_ubs_list.append(dataset["time_ubs"].data[mask][0])

        design_matrix = IndexedMatrix()
        design_matrix.add_rows(row_indices=row_indices, rows=rows)

        row_index_map = design_matrix.row_index_map
        b = np.empty(len(row_index_map), dtype=float)
        sigma_b = np.empty(len(row_index_map), dtype=float)
        for idx, (row_index, array_idx) in enumerate(row_index_map.items()):
            list_idx = row_indices.index(row_index)
            fidelity = fidelities[list_idx]
            fidelity_std = fidelity_stds[list_idx]
            b[array_idx] = -np.log(max(fidelity, 1e-300))
            sigma_b[array_idx] = fidelity_std / max(fidelity, 1e-300)

        A = design_matrix.data

        # Solve the linear problem
        x, metadata = self._linear_solve(A, b)

        # Compute covariance
        cov_b = np.diag(sigma_b**2)
        free_indices = np.where(x > 0)[0]
        cov_x = np.zeros((len(x), len(x)))
        if free_indices.size > 0:
            A_S = A[:, free_indices]
            A_S_pinv = np.linalg.pinv(A_S)
            cov_x_S = A_S_pinv @ cov_b @ A_S_pinv.T
            cov_x[np.ix_(free_indices, free_indices)] = cov_x_S

        # Construct and store the ModelData instance
        col_index_map = design_matrix.column_index_map
        inv_col = {v: k for k, v in col_index_map.items()}
        param_labels = [inv_col[i] for i in range(len(x))]

        all_time_lbs = np.array(time_lbs_list, dtype="datetime64[us]")
        all_time_ubs = np.array(time_ubs_list, dtype="datetime64[us]")
        time_lb = time_bound(all_time_lbs, "min")
        time_ub = time_bound(all_time_ubs, "max")

        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=param_labels,
            parameter_values=x,
            covariance=cov_x,
            time_lbs=np.full(len(x), time_lb, dtype="datetime64[us]"),
            time_ubs=np.full(len(x), time_ub, dtype="datetime64[us]"),
            metadata=metadata,
        )


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

    def _linear_solve(self, a_mat: np.ndarray, b_vec: np.ndarray) -> tuple[np.ndarray, dict]:
        x, residual = opt.nnls(a_mat, b_vec, **self.nnls_opts)
        return x, {"residual": residual}


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

        self.lsq_linear_opts["bounds"] = self.lsq_linear_opts.get("bounds", (0, np.inf))
        self.lsq_linear_opts["method"] = self.lsq_linear_opts.get("method", "bvls")

    def _linear_solve(self, a_mat: np.ndarray, b_vec: np.ndarray) -> tuple[np.ndarray, dict]:
        opt_res = opt.lsq_linear(
            a_mat,
            b_vec,
            **self.lsq_linear_opts,
        )
        return opt_res.x, {"opt_res": opt_res}
