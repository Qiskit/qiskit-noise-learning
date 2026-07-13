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
    from ...data import AveragedData, ModelData, ObservableData
    from ...math import LinearMap
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


def model_curves(
    model: LinearMap,
    model_data: ModelData,
    paths: Iterable[Path],
) -> tuple[dict[Path, float], dict[Path, float]]:
    """Compute model-predicted decay curve parameters for a set of unbound paths.

    For each path, the base is the product of the fidelities in the repeatable fragment, and the
    intercept is the product of the fidelities in the start and end fragments. Every path must be
    unbound, i.e. unbound (:attr:`~.BaseSequence.is_unbound`) with a non-empty repeatable fragment.

    Args:
        model: A fidelity model, i.e. a :class:`~.LinearMap` whose output space is a
            :class:`~.LogFidelitySpace`.
        model_data: The fitted model parameters.
        paths: The unbound paths to predict decays for.

    Returns:
        A ``(bases, intercepts)`` pair of mappings from path to float.

    Raises:
        ValueError: If ``model`` is not a fidelity model, or if any path is bound or has an empty
            repeatable fragment.
    """
    from ...models import LogPathMap, is_fidelity_model

    if not is_fidelity_model(model):
        raise ValueError(
            "model must be a fidelity model (its output space must be a LogFidelitySpace)."
        )

    paths = list(paths)
    non_decay = [path for path in paths if not (path.is_unbound and path.repeatable_fragment)]
    if non_decay:
        raise ValueError(
            f"model_curves requires unbound decay paths, but received {len(non_decay)} path(s) "
            "that are bound or have an empty repeatable fragment (e.g. SPAM or depth-1 paths). "
            "Filter to decay paths before predicting model curves."
        )
    rates = dict(
        zip(model_data.dataset["parameter"].data, model_data.dataset["parameter_values"].data)
    )
    path_map = LogPathMap(model.output_space) @ model

    unbound = [path.without_depth() for path in paths]
    depth_zero = [path.bind_at(0) for path in paths]
    log_fidelities = path_map.projected_output(unbound, rates)
    log_intercepts = path_map.projected_output(depth_zero, rates)

    bases = {path: float(np.exp(-log_fidelities[u])) for path, u in zip(paths, unbound)}
    intercepts = {path: float(np.exp(-log_intercepts[z])) for path, z in zip(paths, depth_zero)}
    return bases, intercepts


def _dataset_paths(*datas: AveragedData | ObservableData | None) -> list[Path]:
    """The unique unbound paths across the given data objects, in first-seen order."""
    seen: dict[Path, None] = {}
    for data in datas:
        if data is not None:
            for path in data.dataset["unbound_path"].data:
                seen.setdefault(path, None)
    return list(seen)
