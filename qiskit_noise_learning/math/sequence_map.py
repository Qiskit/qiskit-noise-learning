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

"""SequenceMap"""

import math
from collections.abc import Hashable, Iterable
from itertools import chain
from typing import TypeVar

from .base_sequence import BaseSequence
from .indexed_matrix import IndexedMatrix
from .indexed_vector import IndexedVector
from .linear_map import LinearMap
from .parameter_space import ParameterSpace

Index = TypeVar("Index", bound=Hashable)


class SequenceSpace(ParameterSpace[BaseSequence[Index]]):
    """The infinite-dimensional space of sequences over an element space.

    A member is any :class:`~.BaseSequence` all of whose elements belong to the underlying element
    space.

    Args:
        element_space: The space whose elements the sequences are built from.
    """

    def __init__(self, element_space: ParameterSpace[Index]):
        self._element_space = element_space

    @property
    def element_space(self) -> ParameterSpace[Index]:
        """The space whose elements the sequences are built from."""
        return self._element_space

    @property
    def dim(self) -> int | float:
        return math.inf

    def __contains__(self, index: object) -> bool:
        return isinstance(index, BaseSequence) and all(
            element in self._element_space
            for element in chain(
                index.start_fragment, index.repeatable_fragment, index.end_fragment
            )
        )


class SequenceMap(LinearMap[Index, BaseSequence[Index]]):
    r"""The linear map sending each sequence to its depth-weighted vector of element multiplicities.

    This realizes the :class:`~.BaseSequence`\ -to-vector linearization as a :class:`~.LinearMap`
    whose output space is the (infinite-dimensional) :class:`SequenceSpace` over ``element_space``
    and whose input space is ``element_space`` itself. Pre-composing it with another map evaluates
    that map on sequences: ``SequenceMap(model.output_space) @ model`` maps the model's input space
    to sequences over its output indices.

    Args:
        element_space: The space whose elements the sequences are built from (the input space).
    """

    def __init__(self, element_space: ParameterSpace[Index]):
        super().__init__(input_space=element_space, output_space=SequenceSpace(element_space))

    def rows(
        self, output_indices: Iterable[BaseSequence[Index]]
    ) -> IndexedMatrix[BaseSequence[Index], Index]:
        """Materialize the rows for the given sequences.

        The row of a sequence is its vector of element multiplicities: the count of each element in
        the repeatable fragment if the sequence is unbound, otherwise ``depth`` times those counts
        plus the counts in the start and end fragments.

        Args:
            output_indices: The sequences whose rows to materialize.

        Returns:
            An :class:`~.IndexedMatrix` indexed by the sequences (rows) and their elements
            (columns).
        """
        output_indices = list(output_indices)
        return IndexedMatrix.from_rows(
            output_indices, [self._sequence_vector(sequence) for sequence in output_indices]
        )

    @staticmethod
    def _sequence_vector(sequence: BaseSequence[Index]) -> IndexedVector[Index]:
        """The depth-weighted element-multiplicity vector of a sequence."""
        vector = IndexedVector[Index]()
        for element in sequence.repeatable_fragment:
            vector[element] = vector.get(element, 0.0) + 1.0

        if sequence.is_unbound:
            return vector

        vector = sequence.depth * vector
        for element in chain(sequence.start_fragment, sequence.end_fragment):
            vector[element] = vector.get(element, 0.0) + 1.0

        return vector
