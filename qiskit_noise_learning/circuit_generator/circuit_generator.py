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

from __future__ import annotations

import abc
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

from ..gate_sets import GateSet
from ..sequences import InstructionSequence

if TYPE_CHECKING:
    from ..analysis import Fit
    from ..experiment_builder import Experiment

TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


class _StructureKey:
    __slots__ = ("sequence",)

    def __init__(self, sequence: InstructionSequence):
        self.sequence = sequence

    def __eq__(self, other):
        return self.sequence.has_same_structure_as(other.sequence)

    def __hash__(self):
        # this hash has a lot of collisions. if it becomes a bottleneck, implement one with fewer
        return hash(self.sequence.depth)


class CircuitGenerator(abc.ABC, Generic[TaskT, ResultT]):
    """Generate experimental tasks from an :class:`.Experiment`.

    In addition to generating experimental tasks, this class also provides an interface to collect
    results, that is, to convert the results from tasks to the standard :class:`.Fit` form.
    """

    @property
    @abc.abstractmethod
    def gate_set(self) -> GateSet:
        """The gate set this generator constructs against."""

    @staticmethod
    @abc.abstractmethod
    def collect(result: ResultT) -> Fit:
        """Coerce data from a specific execution framework into a canonical :class:`.Fit`."""

    @abc.abstractmethod
    def generate(self, experiment: Experiment) -> TaskT:
        """Generate a new experimental task from an experiment."""

    @classmethod
    def partition(cls, sequences: Sequence[InstructionSequence]) -> list[list[InstructionSequence]]:
        """Partition instruction sequences into groups that can share a common generation output."""
        groups: dict[_StructureKey, list[InstructionSequence]] = defaultdict(list)
        for idx, key in enumerate(map(_StructureKey, sequences)):
            groups[key].append((idx, key.sequence))
        return list(groups.values())
