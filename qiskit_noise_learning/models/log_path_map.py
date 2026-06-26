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

"""LogPathMap"""

import math
from collections.abc import Iterable
from itertools import chain

from qiskit_noise_learning.gate_sets import ModelGateSet
from qiskit_noise_learning.math import IndexedMatrix, IndexedSpace, IndexedVector, LinearMap
from qiskit_noise_learning.sequences import FidelityIndex, Path

from .log_fidelity_space import LogFidelitySpace


class LogPathSpace(IndexedSpace[Path]):
    r"""The (infinite-dimensional) space of log path-fidelities.

    For a :class:`~.Path`, the "path-fidelity" is:
    - If the path is unbound, the product of the fidelities in the repeatable fragment.
    - If the path is bound, the product of all fidelities in the path (counting multiplicities).
    This corresponds to the sign-corrected observable of an experiment traversing the path.

    The log path space represents the vector space of such log path-fidelities for a given gate set,
    indexed by the paths themselves.

    Args:
        fidelity_space: The log-fidelity space whose fidelity indices the paths are built from.
    """

    def __init__(self, fidelity_space: LogFidelitySpace):
        self._fidelity_space = fidelity_space

    @property
    def gate_set(self) -> ModelGateSet:
        """The gate set."""
        return self._fidelity_space.gate_set

    @property
    def fidelity_space(self) -> LogFidelitySpace:
        """The log-fidelity space whose fidelity indices the paths are built from."""
        return self._fidelity_space

    @property
    def dim(self) -> int | float:
        return math.inf

    def __contains__(self, index: object) -> bool:
        return isinstance(index, Path) and all(
            fidelity in self._fidelity_space
            for fidelity in chain(
                index.start_fragment, index.repeatable_fragment, index.end_fragment
            )
        )


class LogPathMap(LinearMap[FidelityIndex, Path]):
    r"""The linear map from a LogFidelitySpace to its associated LogPathSpace.

    Args:
        fidelity_space: The log-fidelity space.
    """

    def __init__(self, fidelity_space: LogFidelitySpace):
        super().__init__(input_space=fidelity_space, output_space=LogPathSpace(fidelity_space))

    def rows(self, output_indices: Iterable[Path]) -> IndexedMatrix[Path, FidelityIndex]:
        output_indices = list(output_indices)
        return IndexedMatrix.from_rows(
            output_indices, [self._path_vector(path) for path in output_indices]
        )

    @staticmethod
    def _path_vector(path: Path) -> IndexedVector[FidelityIndex]:
        """The depth-weighted fidelity-index multiplicity vector of a path."""
        vector = IndexedVector[FidelityIndex]()
        for fidelity in path.repeatable_fragment:
            vector[fidelity] = vector.get(fidelity, 0.0) + 1.0

        if path.is_unbound:
            return vector

        vector = path.depth * vector
        for fidelity in chain(path.start_fragment, path.end_fragment):
            vector[fidelity] = vector.get(fidelity, 0.0) + 1.0

        return vector
