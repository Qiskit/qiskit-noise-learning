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
    """A linear map between two indexed spaces, represented row-by-row.

    Given an output index, returns the corresponding sparse row as an
    :class:`~.IndexedVector` over the input space.
    """

    @property
    @abstractmethod
    def input_space(self) -> ParameterSpace[InputIndex]:
        """The parameter space of input indices."""

    @property
    @abstractmethod
    def output_space(self) -> ParameterSpace[OutputIndex]:
        """The parameter space of output indices."""

    @abstractmethod
    def row(self, output_index: OutputIndex) -> IndexedVector[InputIndex]:
        """Return the sparse row corresponding to a given output index."""

    def evaluate(self, output_index: OutputIndex, parameters: Mapping[InputIndex, float]) -> float:
        """Compute the dot product of a row with a parameter vector.

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
        self._inner = inner
        self._outer = outer

    @property
    def input_space(self) -> ParameterSpace[InputIndex]:
        return self._inner.input_space

    @property
    def output_space(self) -> ParameterSpace[OutputIndex]:
        return self._outer.output_space

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
        outer_row = self._outer.row(output_index)
        result = IndexedVector[InputIndex]()
        for middle_index, coeff in outer_row.items():
            inner_row = self._inner.row(middle_index)
            result = result + coeff * inner_row
        return result
