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
from .xarray_utils import filter_time, ragged_concat


class ObservableData(LeveledData):
    """A collection of calculated expectation values.

    This class is a wrapper around an XArray ``Dataset``, containing the following data:
    - Data variables:
        - ``observables``: Observables computed from single :class:`InstructionSequence`
        and :class:`Path` pairs at a given depth, separated by randomizations. Has dimensions
        ``("observable", "randomization")``. ``np.nan`` values are assumed to be due to raggedness
        of the ``"randomization"`` dimension for different observables.
        - ``time_lbs``: Lower bound on data acquisition times, with dimensions
          ``("observable", "randomization")``, and of type ``"datetime64[us]"``.
        - ``time_ubs``: Upper bound on data acquisition times, with dimensions
          ``("randomization", "randomization")``, and of type ``"datetime64[us]"``.
    - Coordinates:
        - ``unbound_path``: The unbound path (with ``depth=None``) for each observable, along
            dimension ``("observable",)``, of type :class:`Path`.
        - ``depth``: Integer array of depths along dimension ``("observable",)``.

    Args:
        dataset: A dataset with the above formatting.
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    @property
    def dataset(self) -> xr.Dataset:
        return self._dataset

    @classmethod
    def from_arrays(
        cls,
        unbound_paths: list[Path],
        depths: np.ndarray[int],
        observables: np.ndarray[np.float64],
        time_lbs: np.ndarray[np.datetime64],
        time_ubs: np.ndarray[np.datetime64],
    ):
        """Instantiate from data specified as arrays.

        Args:
            unbound_paths: The unbound path (with ``depth=None``) for each observable.
            depths: The depths for each observable.
            observables: A 2d numpy array of ``floats`` with axes
                ``("observable", "randomization")``.
            time_lbs: A lower bound on the data collection time for each observable and
                randomization. Has axes ``("observable", "randomization")``.
            time_ubs: Upper bounds on the data collection time for each observable and
                randomization. Has axes ``("observable", "randomization")``.
        """

        dataset = xr.Dataset(
            data_vars={
                "observables": xr.DataArray(data=observables, dims=["observable", "randomization"]),
                "time_lbs": xr.DataArray(data=time_lbs, dims=["observable", "randomization"]),
                "time_ubs": xr.DataArray(data=time_ubs, dims=["observable", "randomization"]),
            },
            coords={
                "unbound_path": (("observable",), np.array(unbound_paths, dtype=object)),
                "depth": (("observable",), depths),
            },
        )

        return cls(dataset=dataset)

    @property
    def observables(self) -> xr.DataArray:
        """Observables data array."""
        return self.dataset["observables"]

    @property
    def time_lbs(self) -> xr.DataArray:
        """Time lower bounds data array."""
        return self.dataset["time_lbs"]

    @property
    def time_ubs(self) -> xr.DataArray:
        """Time upper bounds data array."""
        return self.dataset["time_ubs"]

    def merge(self, other: Self) -> Self:
        """Merge observable data into a single instance.

        Args:
            other: The other observable data set.

        Returns:
            The merged data.
        """
        return ObservableData(
            ragged_concat(
                [self.dataset, other.dataset], concat_dim="observable", ragged_dim="randomization"
            )
        )

    def filter_time(self, lb: np.datetime64, ub: np.datetime64) -> Self:
        """Filter to data gathered within the time bounds.

        Args:
            lb: The time lower bound (inclusive).
            ub: The time upper bound (inclusive).

        Returns:
            The time filtered version of self.
        """
        return ObservableData(filter_time(xr.DataTree(self.dataset), lb=lb, ub=ub).dataset)
