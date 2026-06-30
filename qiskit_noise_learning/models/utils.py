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

"""Utility functions operating on models."""

from qiskit_noise_learning.math import ComposedLinearMap, LinearMap

from .log_fidelity_space import LogFidelitySpace
from .pauli_lindblad_model import PauliLindbladModel


def is_fidelity_model(model: LinearMap) -> bool:
    """Whether a map is LinearMap with LogFidelitySpace output space.

    Args:
        model: The map to check.

    Returns:
        Whether the map's output space is a :class:`~.LogFidelitySpace`.
    """
    return isinstance(model.output_space, LogFidelitySpace)


def contains_pauli_lindblad_model(model: LinearMap) -> bool:
    """Whether a map is, or contains, a :class:`~.PauliLindbladModel`.

    Args:
        model: The map to inspect.

    Returns:
        ``True`` if ``model`` is a :class:`~.PauliLindbladModel` or a :class:`~.ComposedLinearMap`
        whose chain contains one.
    """
    return any(isinstance(sub_map, PauliLindbladModel) for sub_map in _chain(model))


def split_pauli_lindblad_model(
    model: LinearMap,
) -> tuple[ComposedLinearMap | None, PauliLindbladModel, ComposedLinearMap | None]:
    """Split a map around its underlying :class:`~.PauliLindbladModel`.

    Returns a triple ``(outer, plm, inner)`` such that ``model`` is equivalent to
    ``outer @ plm @ inner``. If ``inner`` is ``None`` it indicates nothing is before ``plm``, and
    likewise ``outer`` being ``None`` implies indicates nothing is after.

    Args:
        model: The map to split.

    Returns:
        The ``(outer, plm, inner)`` triple.

    Raises:
        ValueError: If ``model`` does not contain exactly one :class:`~.PauliLindbladModel`.
    """
    maps = _chain(model)
    positions = [i for i, sub_map in enumerate(maps) if isinstance(sub_map, PauliLindbladModel)]

    if len(positions) != 1:
        raise ValueError(
            f"Expected exactly one PauliLindbladModel in the map, found {len(positions)}."
        )

    index = positions[0]
    before = maps[:index]
    after = maps[index + 1 :]

    outer = ComposedLinearMap(after) if after else None
    inner = ComposedLinearMap(before) if before else None
    return outer, maps[index], inner


def _chain(model: LinearMap) -> list[LinearMap]:
    """The maps of ``model`` in application order (its chain, or itself as a singleton)."""
    return model.maps if isinstance(model, ComposedLinearMap) else [model]
