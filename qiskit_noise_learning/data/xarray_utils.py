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

"""Utility functions for data classes."""

import numpy as np
import xarray as xr


def time_bound(times: np.ndarray, kind: str) -> np.datetime64:
    """Compute a min or max time bound from an array, ignoring NaT values.

    Args:
        times: A datetime64 array (any shape).
        kind: Either ``"min"`` or ``"max"``.

    Returns:
        The min or max non-NaT value, or NaT if the array is empty or all-NaT.
    """
    flat = times.flatten()
    valid = flat[~np.isnat(flat)]
    if len(valid) == 0:
        return np.datetime64("NaT")
    return np.min(valid) if kind == "min" else np.max(valid)


def filter_time(datatree: xr.DataTree, lb: np.datetime64, ub: np.datetime64) -> xr.DataTree:
    r"""Return a new data tree containing only data within the given time-window.

    This function only filters data on ``Dataset``\s within the datatree containing both a
    ``time_lbs`` and ``time_ubs`` data variables or coordinates.

    Args:
        lb: The time lower bound.
        ub: The time upper bound.

    Returns:
        Data only within the time bounds, inclusive.
    """

    def _dataset_func(dataset: xr.Dataset, lb: np.datetime64, ub: np.datetime64):
        if ("time_lbs" not in dataset.keys()) or ("time_ubs" not in dataset.keys()):
            return dataset

        return dataset.where((dataset["time_lbs"] >= lb) & (dataset["time_ubs"] <= ub))

    return datatree.map_over_datasets(
        _dataset_func,
        kwargs={
            "lb": np.datetime64(lb),
            "ub": np.datetime64(ub),
        },
    )


def ragged_concat(datasets: list[xr.Dataset], concat_dim: str, ragged_dim: str):
    """Concatenate two datasets along a given dimension with handling for a ragged dimension.

    The ``ragged_dim`` argument specifies a dimension in the datasets that may have different
    lengths, but that can be consolodated to the same length by adding either ``NaN`` (for numeric
    type) or ``NaT`` values (for time types). Note that for boolean variables, ``NaN`` is cast to
    ``True``, and as such ragged dimensions in boolean variables should have a corresponding
    mask variable (in which ``True`` is interpreted as masking the array).

    Note that if raggedness handling is not needed, the standard ``xr.concat`` function should be
    used.

    Args:
        datasets: The datasets to concatenate.
        concat_dim: The dimension to concatenate.
        ragged_dim: The dimensions to apply raggedness handling to.
    """
    # find max ragged dim
    max_ragged_size = 0
    for dataset in datasets:
        max_ragged_size = max(max_ragged_size, dataset.sizes[ragged_dim])

    extended_datasets = [
        extend_dimension(x, ragged_dim, extension_size=max_ragged_size - x.sizes[ragged_dim])
        for x in datasets
    ]
    return xr.concat(extended_datasets, dim=concat_dim)


def extend_dimension(dataset: xr.Dataset, dim: str, extension_size: int) -> xr.Dataset:
    r"""Extend a dimension to a larger size via filling dummy values.

    For time types, ``"NAT"`` is the dummy value, and for all others, ``np.nan``. Note that for
    boolean arrays, ``np.nan`` casts to ``True``, and as such the dummy value is ``True``.

    Note that this does nothing to any ``DataArray``\s in the ``Dataset`` that do not have the
    specified ``dim``.

    Args:
        dataset: The dataset to extend.
        dim: The dimension to extend.
        extension_size: The size of the padding.
    """
    if extension_size == 0:
        return dataset

    dims = []
    nan_shape = []
    for ds_dim, size in dataset.sizes.items():
        dims.append(ds_dim)
        if ds_dim == dim:
            nan_shape.append(extension_size)
        else:
            nan_shape.append(size)

    def extend_array(dataarray: xr.DataArray):
        if dim not in dataarray.dims:
            return dataarray

        if dataarray.dtype == "datetime64[us]":
            nat_array = np.empty(nan_shape, dtype="datetime64[us]")
            nat_array.fill("NAT")
            return xr.concat([dataarray, xr.DataArray(data=nat_array, dims=dims)], dim=dim)
        else:
            nan_array = np.empty(nan_shape, dtype=dataarray.dtype)
            nan_array.fill(np.nan)
            return xr.concat([dataarray, xr.DataArray(data=nan_array, dims=dims)], dim=dim)

    return dataset.map(extend_array)
