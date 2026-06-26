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

"""LinearMap"""

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Mapping
from typing import Generic, TypeVar

from .indexed_matrix import IndexedMatrix
from .indexed_vector import IndexedVector
from .parameter_space import ParameterSpace

InputIndex = TypeVar("InputIndex", bound=Hashable)
OutputIndex = TypeVar("OutputIndex", bound=Hashable)
OtherInput = TypeVar("OtherInput", bound=Hashable)
OtherOutput = TypeVar("OtherOutput", bound=Hashable)
RowLabel = TypeVar("RowLabel", bound=Hashable)


class LinearMap(Generic[InputIndex, OutputIndex], ABC):
    """An implicit linear map between two parameter spaces.

    Args:
        input_space: The input parameter space.
        output_space: The output parameter space.
    """

    def __init__(
        self,
        input_space: ParameterSpace[InputIndex],
        output_space: ParameterSpace[OutputIndex],
    ):
        self._input_space = input_space
        self._output_space = output_space

    @property
    def input_space(self) -> ParameterSpace[InputIndex]:
        """The input parameter space."""
        return self._input_space

    @property
    def output_space(self) -> ParameterSpace[OutputIndex]:
        """The output parameter space."""
        return self._output_space

    @abstractmethod
    def rows(self, output_indices: Iterable[OutputIndex]) -> IndexedMatrix[OutputIndex, InputIndex]:
        """Construct the sub-matrix whose rows are the given output indices.

        Args:
            output_indices: The labels for the desired rows of the matrix.

        Returns:
            :class:`~.IndexedMatrix`
        """

    def left_multiply(
        self, matrix: IndexedMatrix[RowLabel, OutputIndex]
    ) -> IndexedMatrix[RowLabel, InputIndex]:
        """Multiply on the left by an explicit matrix.

        Args:
            matrix: A matrix whose column indices are output indices of this map.

        Returns:
            The resulting matrix.
        """
        return matrix @ self.rows(matrix.column_index_map.keys())

    def evaluate(
        self,
        output_indices: Iterable[OutputIndex],
        parameters: Mapping[InputIndex, float],
    ) -> IndexedVector[OutputIndex]:
        """Compute the projection of the map applied to a parameter vector for some output indices.

        Args:
            output_indices: The output indices to evaluate.
            parameters: A mapping from input indices to parameter values.

        Returns:
            The projected output vector.

        Raises:
            KeyError: If an input index appearing in the rows is not present in ``parameters``.
        """
        matrix = self.rows(output_indices)

        parameter_vector = IndexedVector[InputIndex]()
        for input_index in matrix.column_index_map:
            if input_index not in parameters:
                raise KeyError(input_index)
            parameter_vector[input_index] = parameters[input_index]

        return matrix @ parameter_vector

    def compose(
        self, outer: "LinearMap[OutputIndex, OtherOutput]"
    ) -> "ComposedLinearMap[InputIndex, OtherOutput]":
        """Post-compose: self maps I->O, outer maps O->C, result maps I->C."""
        self_maps = self._maps if isinstance(self, ComposedLinearMap) else [self]
        outer_maps = outer.maps if isinstance(outer, ComposedLinearMap) else [outer]
        return ComposedLinearMap(self_maps + outer_maps)

    def pre_compose(
        self, inner: "LinearMap[OtherInput, InputIndex]"
    ) -> "ComposedLinearMap[OtherInput, OutputIndex]":
        """Pre-compose: inner maps A->I, self maps I->O, result maps A->O."""
        inner_maps = inner.maps if isinstance(inner, ComposedLinearMap) else [inner]
        self_maps = self._maps if isinstance(self, ComposedLinearMap) else [self]
        return ComposedLinearMap(inner_maps + self_maps)

    def __matmul__(
        self, other: "LinearMap[OtherInput, InputIndex]"
    ) -> "ComposedLinearMap[OtherInput, OutputIndex]":
        """``self @ other`` means ``self`` applied after ``other``."""
        return self.pre_compose(other)

    def __rmatmul__(
        self, other: "LinearMap[OutputIndex, OtherOutput]"
    ) -> "ComposedLinearMap[InputIndex, OtherOutput]":
        """``other @ self`` when ``other`` is not a :class:`LinearMap`, but ``self`` is."""
        return self.compose(other)


class ComposedLinearMap(LinearMap[InputIndex, OutputIndex]):
    """A linear map formed by composing a chain of maps.

    Maps are stored in application order: ``maps[0]`` is applied first (innermost), ``maps[-1]`` is
    applied last (outermost).

    Args:
        maps: The ordered sequence of maps to compose.
    """

    def __init__(self, maps: list[LinearMap]):
        self._maps = list(maps)
        super().__init__(
            input_space=self._maps[0].input_space, output_space=self._maps[-1].output_space
        )

    @property
    def maps(self) -> list[LinearMap]:
        """The ordered list of maps in application order."""
        return list(self._maps)

    def rows(self, output_indices: Iterable[OutputIndex]) -> IndexedMatrix[OutputIndex, InputIndex]:
        result = IndexedMatrix.identity(list(output_indices))
        for current_map in reversed(self._maps):
            result = current_map.left_multiply(result)
        return result

    def compose(self, outer: "LinearMap[OutputIndex, OtherOutput]") -> "ComposedLinearMap":
        outer_maps = outer.maps if isinstance(outer, ComposedLinearMap) else [outer]
        return self.__class__(self._maps + outer_maps)

    def pre_compose(self, inner: "LinearMap[OtherInput, InputIndex]") -> "ComposedLinearMap":
        inner_maps = inner.maps if isinstance(inner, ComposedLinearMap) else [inner]
        return self.__class__(inner_maps + self._maps)
