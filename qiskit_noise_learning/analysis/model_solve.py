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
from typing import Any

import numpy as np
import scipy.optimize as opt

from qiskit_noise_learning.analysis import AnalysisStage, Fit
from qiskit_noise_learning.data import AveragedData, ModelData
from qiskit_noise_learning.data.xarray_utils import time_bound
from qiskit_noise_learning.math import IndexedMatrix


class ModelSolve(AnalysisStage):
    """Base class for finding model parameters.

    Constructs the multiplicative design matrix from the :class:`~.FidelityModel` stored on the
    :class:`~.Fit` container and the path pattern keys in the :class:`~.AveragedData` (depth==-1
    entries). Then solves ``A @ x = b`` using an arbitrary method, where ``A`` is the
    design matrix and ``b`` is the vector of decay rates ``-log(f)``.
    """

    linear_solve_options: dict[str, Any] = {}

    @property
    def input_level(self):
        return AveragedData

    @property
    def output_level(self):
        return ModelData

    @abstractmethod
    def _linear_solve(self, a_mat: np.ndarray, b_vec: np.ndarray) -> tuple[np.ndarray, dict]:
        pass

    def _run(self, fit: Fit):
        averaged_data = fit[AveragedData]
        fidelity_model = fit.model

        # Filter to decay data (depth == -1)
        decay_mask = averaged_data.dataset["depth"].data == -1
        decay_dataset = averaged_data.dataset.sel({"observable": decay_mask})

        # Build design matrix from the fidelity model and the path patterns in the decay data
        path_patterns = list(decay_dataset["path_pattern"].data)
        rows = [fidelity_model.multiplicative_row_from_path_pattern(pp) for pp in path_patterns]
        design_matrix = IndexedMatrix()
        design_matrix.add_rows(row_indices=path_patterns, rows=rows)

        # Construct b from averaged_data taking the negative logarithm
        row_index_map = design_matrix.row_index_map
        b = np.empty(len(row_index_map), dtype=float)
        sigma_b = np.empty(len(row_index_map), dtype=float)
        for pp, row_idx in row_index_map.items():
            pp_mask = decay_dataset["path_pattern"].data == pp
            fidelity = float(decay_dataset["observables"].data[pp_mask][0])
            fidelity_std = float(decay_dataset["std"].data[pp_mask][0])
            b[row_idx] = -np.log(max(fidelity, 1e-300))
            sigma_b[row_idx] = fidelity_std / max(fidelity, 1e-300)

        A = design_matrix.data

        # Solve the nnls problem
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


class NNLSSolve(ModelSolve):
    """Solves for the :class:`~.ModelData` using the [scipy's NNLS implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html)

    See :class:`~.ModelSolve` for more details.
    """

    def _linear_solve(self, a_mat: np.ndarray, b_vec: np.ndarray) -> tuple[np.ndarray, dict]:
        x, residual = opt.nnls(a_mat, b_vec, **self.linear_solve_options)
        return x, {"residual": residual}


class LSQLinearSolve(ModelSolve):
    """Solves for the :class:`~.ModelData` using the [scipy's least squares linear implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html#scipy.optimize.lsq_linear)

    See :class:`~.ModelSolve` for more details.
    """

    linear_solve_options: dict[str, Any] = {
        "bounds": (0, np.inf),
        "method": "bvls",
    }

    def _linear_solve(self, a_mat: np.ndarray, b_vec: np.ndarray) -> tuple[np.ndarray, dict]:
        opt_res = opt.lsq_linear(
            a_mat,
            b_vec,
            **self.linear_solve_options,
        )
        return opt_res.x, {"opt_res": opt_res}
