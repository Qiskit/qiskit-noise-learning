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

"""Analysis utilities."""

from collections.abc import Iterable

import numpy as np

from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.math import LinearMap
from qiskit_noise_learning.models import is_fidelity_model
from qiskit_noise_learning.sequences import LogPathMap, Path


def predicted_path_decays(
    model: LinearMap,
    model_data: ModelData,
    paths: Iterable[Path],
) -> tuple[dict[Path, float], dict[Path, float]]:
    """Compute the decay curve parameters a fitted model predicts for a set of unbound paths.

    For each path, the base is the product of the fidelities in the repeatable fragment, and the
    intercept is the product of the fidelities in the start and end fragments. Every path must be
    unbound (:attr:`~.BaseSequence.is_unbound`) with a non-empty repeatable fragment.

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
    if not is_fidelity_model(model):
        raise ValueError(
            "model must be a fidelity model (its output space must be a LogFidelitySpace)."
        )

    paths = list(paths)
    non_decay = [path for path in paths if not (path.is_unbound and path.repeatable_fragment)]
    if non_decay:
        raise ValueError(
            f"predicted_path_decays requires unbound decay paths, but received {len(non_decay)} "
            "path(s) that are bound or have an empty repeatable fragment (e.g. SPAM or depth-1 "
            "paths). Filter to decay paths before predicting model curves."
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
