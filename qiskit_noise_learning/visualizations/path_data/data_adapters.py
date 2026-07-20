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

"""Adapters that extract plotter inputs (points / decay parameters) from the data classes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from .primitives import PointSeries

if TYPE_CHECKING:
    from ...data import AveragedData, ObservableData
    from ...sequences import Path


def exponential_fit_curves(
    averaged_data: AveragedData,
    paths: Iterable[Path] | None = None,
) -> tuple[dict[Path, float], dict[Path, float]]:
    """Extract exponential-fit decay parameters from averaged data.

    Extracts the ``base`` and ``intercept`` for entries with ``depth == -1``, for use in
    :func:`plot_path_decay_curves`. The ``intercept`` is drawn from ``metadata["spam_fidelity"]``,
    and defaults to ``1.0`` if absent.

    Args:
        averaged_data: The averaged data to extract from.
        paths: Optional paths to restrict extraction to. Defaults to all paths in the data.

    Returns:
        A ``(bases, intercepts)`` pair of mappings from path to float.
    """
    dataset = averaged_data.dataset
    wanted = set(paths) if paths is not None else None
    mask = dataset["depth"].data == -1
    entry_paths = dataset["unbound_path"].data[mask]
    observables = dataset["observables"].data[mask]
    metadata = dataset["metadata"].data[mask]

    bases: dict[Path, float] = {}
    intercepts: dict[Path, float] = {}
    for path, fidelity, meta in zip(entry_paths, observables, metadata):
        if wanted is not None and path not in wanted:
            continue
        bases[path] = float(fidelity)
        intercepts[path] = float(meta["spam_fidelity"]) if meta and "spam_fidelity" in meta else 1.0

    return bases, intercepts


def averaged_data_points(
    averaged_data: AveragedData,
    paths: Iterable[Path] | None = None,
) -> dict[Path, PointSeries]:
    """Extract point series data for the bound paths in an average data instance.

    Args:
        averaged_data: The averaged data to extract from.
        paths: Optional paths to restrict extraction to. Defaults to all paths in the data.

    Returns:
        A mapping from path to its :class:`PointSeries` (with ``stds`` populated).
    """
    dataset = averaged_data.dataset
    wanted = set(paths) if paths is not None else None
    mask = dataset["depth"].data >= 0
    entry_paths = dataset["unbound_path"].data[mask]
    depths = dataset["depth"].data[mask]
    observables = dataset["observables"].data[mask]
    stds = dataset["std"].data[mask]

    result: dict[Path, PointSeries] = {}
    for path in dict.fromkeys(entry_paths):
        if wanted is not None and path not in wanted:
            continue
        path_mask = np.array([other == path for other in entry_paths])
        result[path] = PointSeries(
            xs=depths[path_mask].astype(float),
            ys=observables[path_mask].astype(float),
            stds=stds[path_mask].astype(float),
        )

    return result


def observable_data_points(
    observable_data: ObservableData,
    paths: Iterable[Path] | None = None,
) -> dict[Path, PointSeries]:
    """Extract raw per-randomization decay points from observable data.

    Flattens the ``randomization`` axis (dropping ``nan`` raggedness padding) so each retained
    randomization becomes its own point.

    Args:
        observable_data: The observable data to extract from.
        paths: Optional paths to restrict extraction to. Defaults to all paths in the data.

    Returns:
        A mapping from path to its :class:`PointSeries` (with ``stds`` unset).
    """
    dataset = observable_data.dataset
    wanted = set(paths) if paths is not None else None
    entry_paths = dataset["unbound_path"].data
    depths = dataset["depth"].data
    observables = dataset["observables"].data

    result: dict[Path, PointSeries] = {}
    for path in dict.fromkeys(entry_paths):
        if wanted is not None and path not in wanted:
            continue
        row_mask = np.array([other == path for other in entry_paths])

        flat_depths: list[float] = []
        flat_values: list[float] = []
        for depth, row_values in zip(depths[row_mask], observables[row_mask]):
            valid = row_values[~np.isnan(row_values)]
            flat_depths.extend([float(depth)] * valid.size)
            flat_values.extend(float(value) for value in valid)

        result[path] = PointSeries(
            xs=np.array(flat_depths, dtype=float),
            ys=np.array(flat_values, dtype=float),
        )

    return result


def _dataset_paths(*datas: AveragedData | ObservableData | None) -> list[Path]:
    """The unique unbound paths across the given data objects, in first-seen order."""
    seen: dict[Path, None] = {}
    for data in datas:
        if data is not None:
            for path in data.dataset["unbound_path"].data:
                seen.setdefault(path, None)
    return list(seen)
