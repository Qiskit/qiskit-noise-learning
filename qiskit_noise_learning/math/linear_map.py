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
from collections.abc import Hashable, Mapping
from typing import Generic, TypeVar

from .indexed_vector import IndexedVector
from .parameter_space import ParameterSpace

InputIndex = TypeVar("InputIndex", bound=Hashable)
OutputIndex = TypeVar("OutputIndex", bound=Hashable)
OtherInput = TypeVar("OtherInput", bound=Hashable)
OtherOutput = TypeVar("OtherOutput", bound=Hashable)


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
    def row(self, output_index: OutputIndex) -> IndexedVector[InputIndex]:
        """The sparse row in the linear map corresponding to a given output parameter index."""

    def left_multiply(self, vector: IndexedVector[OutputIndex]) -> IndexedVector[InputIndex]:
        """Left-multiply a sparse vector against this map.

        Given a vector in the output space, expands each entry through the corresponding row
        to produce a vector in the input space.

        Args:
            vector: A sparse vector indexed by the output space.

        Returns:
            A sparse vector in the input space.
        """
        result = IndexedVector[InputIndex]()
        for idx, coeff in vector.items():
            result = result + coeff * self.row(idx)
        return result

    def evaluate(self, output_index: OutputIndex, parameters: Mapping[InputIndex, float]) -> float:
        """Compute the dot product of a row with an input parameter vector.

        Args:
            output_index: The output index whose row to evaluate.
            parameters: A mapping from input indices to parameter values.

        Returns:
            The dot product of the row with the parameters.

        Raises:
            KeyError: If a required parameter index is not found in parameters.
        """
        row = self.row(output_index)
        return sum(coeff * parameters[idx] for idx, coeff in row.items())

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

    Maps are stored in application order: ``maps[0]`` is applied first (innermost),
    ``maps[-1]`` is applied last (outermost).

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

    def row(self, output_index: OutputIndex) -> IndexedVector[InputIndex]:
        """Compute the composed row by folding left_multiply through the chain."""
        result = IndexedVector[OutputIndex]({output_index: 1.0})
        for m in reversed(self._maps):
            result = m.left_multiply(result)
        return result

    def compose(self, outer: "LinearMap[OutputIndex, OtherOutput]") -> "ComposedLinearMap":
        """Post-compose, flattening into a single chain via ``self.__class__``."""
        outer_maps = outer.maps if isinstance(outer, ComposedLinearMap) else [outer]
        return self.__class__(self._maps + outer_maps)

    def pre_compose(self, inner: "LinearMap[OtherInput, InputIndex]") -> "ComposedLinearMap":
        """Pre-compose, flattening into a single chain via ``self.__class__``."""
        inner_maps = inner.maps if isinstance(inner, ComposedLinearMap) else [inner]
        return self.__class__(inner_maps + self._maps)
