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
MiddleIndex = TypeVar("MiddleIndex", bound=Hashable)
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
    ) -> "ComposedLinearMap[InputIndex, OutputIndex, OtherOutput]":
        """Post-compose: self maps I->O, outer maps O->C, result maps I->C."""
        return ComposedLinearMap(inner=self, outer=outer)

    def pre_compose(
        self, inner: "LinearMap[OtherInput, InputIndex]"
    ) -> "ComposedLinearMap[OtherInput, InputIndex, OutputIndex]":
        """Pre-compose: inner maps A->I, self maps I->O, result maps A->O."""
        return ComposedLinearMap(inner=inner, outer=self)

    def __matmul__(
        self, other: "LinearMap[OtherInput, InputIndex]"
    ) -> "ComposedLinearMap[OtherInput, InputIndex, OutputIndex]":
        """``self @ other`` means ``self`` applied after ``other``."""
        return self.pre_compose(other)

    def __rmatmul__(
        self, other: "LinearMap[OutputIndex, OtherOutput]"
    ) -> "ComposedLinearMap[InputIndex, OutputIndex, OtherOutput]":
        """``other @ self`` when ``other`` is not a :class:`LinearMap`, but ``self`` is."""
        return self.compose(other)


class ComposedLinearMap(LinearMap[InputIndex, OutputIndex]):
    """A linear map formed by composing two maps: inner (I->M) and outer (M->O).

    The resulting map goes from I to O by chaining through M.

    Args:
        inner: The first map applied (I -> M).
        outer: The second map applied (M -> O).
    """

    def __init__(
        self,
        inner: LinearMap[InputIndex, MiddleIndex],
        outer: LinearMap[MiddleIndex, OutputIndex],
    ):
        super().__init__(input_space=inner.input_space, output_space=outer.output_space)
        self._inner = inner
        self._outer = outer

    @property
    def inner(self) -> LinearMap[InputIndex, MiddleIndex]:
        """The first map in the composition (I -> M)."""
        return self._inner

    @property
    def outer(self) -> LinearMap[MiddleIndex, OutputIndex]:
        """The second map in the composition (M -> O)."""
        return self._outer

    def row(self, output_index: OutputIndex) -> IndexedVector[InputIndex]:
        """Compute the composed row by expanding through the intermediate space."""
        return self._inner.left_multiply(self._outer.row(output_index))
