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

"""Propagation of fitted model data through linear maps."""

from collections.abc import Hashable, Iterable

import numpy as np

from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.math import IndexedMatrix, IndexedVector, LinearMap


def propagate_model_data(
    linear_map: LinearMap,
    model_data: ModelData,
    output_indices: Iterable[Hashable],
) -> ModelData:
    r"""Propagate fitted model data forward through a linear map.

    Given fitted parameters :math:`x` (with covariance :math:`\Sigma`) whose indices are the input
    space of ``linear_map``, this returns the parameters at the map's output indices: the values
    :math:`M x` and covariance :math:`M \Sigma M^\top`, where :math:`M` is the restriction of
    ``linear_map`` to ``output_indices``.

    The output indices must be supplied explicitly (the full output space is in general too large,
    or not enumerable, to materialize). Each output parameter's acquisition-time bounds are set to
    the envelope (min lower bound, max upper bound) of the input parameters that contribute to it;
    an output with no contributors falls back to the global envelope.

    Args:
        linear_map: The map to propagate through. Its input indices must match the parameter
            indices of ``model_data``.
        model_data: The fitted parameters and covariance to propagate.
        output_indices: The output indices to evaluate ``linear_map`` on.

    Returns:
        A :class:`~.ModelData` whose parameters are ``output_indices``.
    """
    output_indices = list(output_indices)
    dataset = model_data.dataset

    param_labels = list(dataset.coords["parameter"].values)
    cov_row_labels = list(dataset.coords["parameter_row"].values)
    cov_col_labels = list(dataset.coords["parameter_col"].values)
    time_lbs = dataset["time_lbs"].values
    time_ubs = dataset["time_ubs"].values

    value_vector = IndexedVector(
        {
            label: float(value)
            for label, value in zip(param_labels, dataset["parameter_values"].values)
        }
    )
    covariance = IndexedMatrix.from_index_lists(
        row_indices=cov_row_labels,
        column_indices=cov_col_labels,
        data=dataset["covariance"].values,
    )

    matrix = linear_map.to_indexed_matrix(output_indices)
    propagated_values = matrix @ value_vector
    propagated_covariance = matrix @ covariance @ matrix.transpose()

    lb_by_param = dict(zip(param_labels, time_lbs))
    ub_by_param = dict(zip(param_labels, time_ubs))
    global_lb = time_lbs.min()
    global_ub = time_ubs.max()

    n = len(output_indices)
    out_values = np.array([propagated_values.get(idx, 0.0) for idx in output_indices], dtype=float)

    out_covariance = np.zeros((n, n), dtype=float)
    for i, row_index in enumerate(output_indices):
        if row_index not in propagated_covariance.row_index_map:
            continue
        row_vector = propagated_covariance[row_index]
        for j, col_index in enumerate(output_indices):
            out_covariance[i, j] = row_vector.get(col_index, 0.0)

    out_lbs = np.empty(n, dtype=time_lbs.dtype)
    out_ubs = np.empty(n, dtype=time_ubs.dtype)
    for i, row_index in enumerate(output_indices):
        contributors = [
            idx
            for idx, coeff in linear_map.row(row_index).items()
            if coeff != 0.0 and idx in lb_by_param
        ]
        if contributors:
            out_lbs[i] = min(lb_by_param[idx] for idx in contributors)
            out_ubs[i] = max(ub_by_param[idx] for idx in contributors)
        else:
            out_lbs[i] = global_lb
            out_ubs[i] = global_ub

    return ModelData.from_arrays(
        parameter_indices=output_indices,
        parameter_values=out_values,
        covariance=out_covariance,
        time_lbs=out_lbs,
        time_ubs=out_ubs,
        metadata=model_data.metadata,
    )
