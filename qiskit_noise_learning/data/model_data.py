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

from collections.abc import Hashable
from typing import Any, Generic, Self, TypeVar

import numpy as np
import xarray as xr

from .leveled_data import LeveledData
from .xarray_utils import filter_time

ParameterIndex = TypeVar("ParameterIndex", bound=Hashable)


class ModelData(LeveledData, Generic[ParameterIndex]):
    """Results from fitting, backed by an xarray Dataset.

    The dataset has data variables:

    - ``parameters``: 1D array with dimension ``parameter``.
    - ``covariance``: 2D array with dimensions ``(parameter_row, parameter_col)``.

    The ``parameter``, ``parameter_row``, and ``parameter_col`` coordinates all share the same
    parameter labels. Additional fit metadata is stored in dataset attrs.
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    @property
    def dataset(self) -> xr.Dataset:
        return self._dataset

    @classmethod
    def from_arrays(
        cls,
        parameter_indices: list[ParameterIndex],
        parameter_values: np.ndarray[np.float64],
        covariance: np.ndarray[np.float64],
        time_lbs: np.ndarray[np.datetime64],
        time_ubs: np.ndarray[np.datetime64],
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        """Instantiate from data specified as arrays in standard containers.

        Args:
            parameter_indices: A list of ``ParameterIndex`` instances.
            parameter_values: A 1d array of floats indicating parameter values.
            covariance: A 2d array of floats indicating the covariances of the parameter values.
            time_lbs: A 1d array of data acquisition time lower bounds for each parameter estimate.
            time_ubs: A 1d array of data acquisition time upper bounds for each parameter estimate.
            metadata: Any metadata to attach to the dataset.
        """
        labels = np.array(parameter_indices, dtype=object)

        dataset = xr.Dataset(
            data_vars={
                "parameter_values": xr.DataArray(data=parameter_values, dims=["parameter"]),
                "covariance": xr.DataArray(
                    data=covariance, dims=["parameter_row", "parameter_col"]
                ),
                "time_lbs": xr.DataArray(data=time_lbs, dims=["parameter"]),
                "time_ubs": xr.DataArray(data=time_ubs, dims=["parameter"]),
            },
            coords={
                "parameter": labels,
                "parameter_row": labels.copy(),
                "parameter_col": labels.copy(),
            },
            attrs=metadata or {},
        )

        return cls(dataset=dataset)

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata describing the model parameter fit."""
        return dict(self._dataset.attrs)

    def filter_time(self, lb: np.datetime64, ub: np.datetime64) -> Self:
        """Filter to data gathered within the time bounds.

        Args:
            lb: The time lower bound (inclusive).
            ub: The time upper bound (inclusive).

        Returns:
            The time filtered version of self.
        """
        return ModelData(filter_time(xr.DataTree(self.dataset), lb=lb, ub=ub).dataset)
