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

from typing import Self

import numpy as np
import xarray as xr

from qiskit_noise_learning.sequences import Path

from .leveled_data import LeveledData
from .xarray_utils import filter_time


class AveragedData(LeveledData):
    """Observable data aggregated over randomizations, or through other curve fitting.

    This class represents aggregated information for specific observables, averaged over
    randomizations, including through potential curve fitting. The data is stored as an XArray
    ``Dataset`` with the following data:
    - Data variables:
        - ``observables``: A 1d float array with dimensions ``("observable",)``.
        - ``std``: A 1d array of standard deviations for the observable estimates, with dimensions
            ``("observable",)``.
        - time_lbs: A lower bound on the data collection for observable, with dimensions
            ``("observable",)``.
        - time_ubs: An upper bound on the data collection for observable, with dimensions
            ``("observable",)``.
    - Coordinates:
        - ``unbound_path``: A 1d array of unbound :class:`Path` instances labelling each observable,
            with dimensions ``("observable",)``.
        - ``fragment_depth``: A 1d array of type ``int`` specifying the fragment depth associated to
            the observable. A value of ``-1`` indicates an estimate of only the
            ``repeatable_fragment`` of the path.

    Args:
        dataset: A ``Dataset`` with the above formatting.
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    @property
    def dataset(self) -> xr.Dataset:
        """The averaged observable data set."""
        return self._dataset

    @classmethod
    def from_arrays(
        cls,
        unbound_paths: list[Path],
        fragment_depths: list[int],
        observables: np.ndarray[float],
        std: np.ndarray[float],
        time_lbs: np.ndarray[np.datetime64],
        time_ubs: np.ndarray[np.datetime64],
        metadata: np.ndarray[object] | None = None,
    ) -> Self:
        """Instantiate from data specified as arrays in standard containers.

        Args:
            unbound_paths: A list of unbound paths (with ``fragment_depth=None``).
            fragment_depths: A list of fragment depths, with ``-1`` indicating the corresponding
                observable is in reference to only the repeatable fragment of the corresponding
                path.
            observables: A 1d array of observable estimates.
            std: A 1d array of standard deviations.
            time_lbs: A 1d array of time lower bounds.
            time_ubs: A 1d array of time upper bounds.
            metadata: Any additional data associated with a given observable.
        """
        dataset = xr.Dataset(
            data_vars={
                "observables": xr.DataArray(data=observables, dims=["observable"]),
                "std": xr.DataArray(data=std, dims=["observable"]),
                "time_lbs": xr.DataArray(data=time_lbs, dims=["observable"]),
                "time_ubs": xr.DataArray(data=time_ubs, dims=["observable"]),
                "metadata": xr.DataArray(
                    data=(
                        np.array([None] * len(observables), dtype=object)
                        if metadata is None
                        else metadata
                    ),
                    dims=["observable"],
                ),
            },
            coords={
                "unbound_path": (("observable",), np.array(unbound_paths, dtype=object)),
                "fragment_depth": (("observable",), fragment_depths),
            },
        )

        return cls(dataset=dataset)

    def merge(self, other: Self) -> Self:
        """Merge the data from self and other into a single instance.

        Args:
            other: The other data.

        Returns:
            A new instance containing both data sets.
        """
        return AveragedData(xr.concat([self.dataset, other.dataset], dim="observable"))

    def filter_time(self, lb: np.datetime64, ub: np.datetime64) -> Self:
        """Filter to data gathered within the time bounds.

        Args:
            lb: The time lower bound (inclusive).
            ub: The time upper bound (inclusive).

        Returns:
            The time filtered version of self.
        """
        return AveragedData(filter_time(xr.DataTree(self.dataset), lb=lb, ub=ub).dataset)
