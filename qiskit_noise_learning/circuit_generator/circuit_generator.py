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

import abc
from collections import defaultdict
from collections.abc import Sequence
from typing import Generic, TypeVar

from qiskit_noise_learning.analysis.fit import Fit
from qiskit_noise_learning.experiment_builder.experiment import Experiment

from ..gate_sets import GateSet
from ..sequences import InstructionSequence

TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")
DataMapperT = TypeVar("DataMapperT")


class _StructureKey:
    __slots__ = ("sequence",)

    def __init__(self, sequence: InstructionSequence):
        self.sequence = sequence

    def __eq__(self, other):
        return self.sequence.has_same_structure_as(other.sequence)

    def __hash__(self):
        # this hash has a lot of collisions. if it becomes a bottleneck, implement one with fewer
        return hash(self.sequence.fragment_depth)


class CircuitGenerator(abc.ABC, Generic[TaskT, DataMapperT, ResultT]):
    """Generate experimental tasks from given instruction sequences.

    In addition to generating experimental tasks, this class also provides an interface to designate
    data, that is, to convert the results from tasks to the standard results.
    """

    @property
    @abc.abstractmethod
    def gate_set(self) -> GateSet:
        """The gate set this generator constructs against."""

    @staticmethod
    @abc.abstractmethod
    def collect(result: ResultT, data_mapper: DataMapperT) -> Fit:
        """Coerce data from a specific execution framework into a canonical form."""

    @abc.abstractmethod
    def generate(self, experiment: Experiment) -> tuple[TaskT, DataMapperT]:
        """Generate a new experimental task from the provided experiment."""

    @classmethod
    def partition(cls, sequences: Sequence[InstructionSequence]) -> list[list[InstructionSequence]]:
        """Partition instruction sequences into groups that can share a common generation output."""
        groups: dict[_StructureKey, list[InstructionSequence]] = defaultdict(list)
        for idx, key in enumerate(map(_StructureKey, sequences)):
            groups[key].append((idx, key.sequence))
        return list(groups.values())
