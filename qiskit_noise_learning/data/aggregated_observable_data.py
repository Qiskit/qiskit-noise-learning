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


class AggregatedObservableData(LeveledData):
    """Per-path estimates obtained by aggregating :class:`~.ObservableData`.

    This class holds a single estimate per observable, obtained by collapsing the
    per-randomization structure of :class:`~.ObservableData`. Each observable estimate is labelled
    by an unbound path and a corresponding fragment depth. A non-negative fragment depth corresponds
    to labelling by a bound path with that depth, and a fragment depth of ``-1`` signals that the
    estimate corresponds to a genuinely unbound path. For a non-negative fragment depth, the
    estimate value corresponds to the product of all fidelities in the path, and the estimate for a
    fragment depth of ``-1`` corresponds to the product of the fidelities in the repeatable
    fragment.

    - Data variables:

        - ``estimate_values``: A 1d float array of per-observable estimates, with dimensions
          ``("observable",)``.
        - ``estimate_std``: A 1d array of standard deviations for the estimates, with dimensions
          ``("observable",)``.
        - ``time_lbs``: A lower bound on the data collection for each observable, with dimensions
          ``("observable",)``.
        - ``time_ubs``: An upper bound on the data collection for each observable, with dimensions
          ``("observable",)``.
        - ``metadata``: A 1d object array of any additional per-observable data, with dimensions
          ``("observable",)``.

    - Coordinates:

        - ``unbound_path``: A 1d array of unbound :class:`~.Path` instances labelling each
          observable, with dimensions ``("observable",)``.
        - ``fragment_depth``: A 1d array of type ``int`` specifying the fragment depth
          associated to the observable. A value of ``-1`` indicates an estimate of only the
          ``repeatable_fragment`` of the path.

    Args:
        dataset: A ``Dataset`` with the above formatting.
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    @property
    def dataset(self) -> xr.Dataset:
        """The aggregated observable data set."""
        return self._dataset

    @classmethod
    def from_arrays(
        cls,
        unbound_paths: list[Path],
        fragment_depths: list[int],
        estimate_values: np.ndarray[float],
        estimate_std: np.ndarray[float],
        time_lbs: np.ndarray[np.datetime64],
        time_ubs: np.ndarray[np.datetime64],
        metadata: np.ndarray[object] | None = None,
    ) -> Self:
        """Instantiate from data specified as arrays in standard containers.

        Args:
            unbound_paths: A list of unbound paths (with ``fragment_depth=None``).
            fragment_depths: A list of fragment depths, with ``-1`` indicating the corresponding
                estimate is in reference to only the repeatable fragment of the corresponding
                path.
            estimate_values: A 1d array of per-observable estimates.
            estimate_std: A 1d array of standard deviations.
            time_lbs: A 1d array of time lower bounds.
            time_ubs: A 1d array of time upper bounds.
            metadata: Any additional data associated with a given observable.
        """
        dataset = xr.Dataset(
            data_vars={
                "estimate_values": xr.DataArray(data=estimate_values, dims=["observable"]),
                "estimate_std": xr.DataArray(data=estimate_std, dims=["observable"]),
                "time_lbs": xr.DataArray(data=time_lbs, dims=["observable"]),
                "time_ubs": xr.DataArray(data=time_ubs, dims=["observable"]),
                "metadata": xr.DataArray(
                    data=metadata or np.array([None] * len(estimate_values), dtype=object),
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
        return AggregatedObservableData(xr.concat([self.dataset, other.dataset], dim="observable"))

    def filter_time(self, lb: np.datetime64, ub: np.datetime64) -> Self:
        """Filter to data gathered within the time bounds.

        Args:
            lb: The time lower bound (inclusive).
            ub: The time upper bound (inclusive).

        Returns:
            The time filtered version of self.
        """
        return AggregatedObservableData(
            filter_time(xr.DataTree(self.dataset), lb=lb, ub=ub).dataset
        )
