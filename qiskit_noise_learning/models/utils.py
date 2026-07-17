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

from typing import NamedTuple

from qiskit_noise_learning.math import ComposedLinearMap, LinearMap

from .log_fidelity_space import LogFidelitySpace
from .pauli_lindblad_model import PauliLindbladModel


class PauliLindbladSplit(NamedTuple):
    """The decomposition of a map around its underlying :class:`~.PauliLindbladModel`.

    The original map is equivalent to ``after @ model @ before`` (``before`` applied first). A
    ``None`` value for ``before`` or ``after`` indicates there are no maps on that side of
    ``model``.

    Args:
        before: The maps applied before ``model`` (mapping into ``model``'s input space), or
            ``None``.
        model: The underlying :class:`~.PauliLindbladModel`.
        after: The maps applied after ``model`` (mapping out of ``model``'s output space), or
            ``None``.
    """

    before: ComposedLinearMap | None
    model: PauliLindbladModel
    after: ComposedLinearMap | None


def is_fidelity_model(model: object) -> bool:
    """Whether an object is a :class:`~.LinearMap` with a :class:`~.LogFidelitySpace` output space.

    Args:
        model: The object to check.

    Returns:
        ``True`` if ``model`` is a :class:`~.LinearMap` whose output space is a
        :class:`~.LogFidelitySpace`, otherwise ``False``.
    """
    return isinstance(model, LinearMap) and isinstance(model.output_space, LogFidelitySpace)


def contains_pauli_lindblad_model(model: LinearMap) -> bool:
    """Whether a map is, or contains, a :class:`~.PauliLindbladModel`.

    Args:
        model: The map to inspect.

    Returns:
        ``True`` if ``model`` is a :class:`~.PauliLindbladModel` or a :class:`~.ComposedLinearMap`
        whose chain contains one.
    """
    return any(isinstance(sub_map, PauliLindbladModel) for sub_map in _chain(model))


def split_pauli_lindblad_model(model: LinearMap) -> PauliLindbladSplit:
    """Split a map around its underlying :class:`~.PauliLindbladModel`.

    Args:
        model: The map to split.

    Returns:
        A :class:`PauliLindbladSplit` ``(before, model, after)`` such that the input map is
        equivalent to ``after @ model @ before``.

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
    before_maps = maps[:index]
    after_maps = maps[index + 1 :]

    before = ComposedLinearMap(before_maps) if before_maps else None
    after = ComposedLinearMap(after_maps) if after_maps else None
    return PauliLindbladSplit(before=before, model=maps[index], after=after)


def _chain(model: LinearMap) -> list[LinearMap]:
    """The maps of ``model`` in application order (its chain, or itself as a singleton)."""
    return model.maps if isinstance(model, ComposedLinearMap) else [model]
