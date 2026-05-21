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

"""Experiment Builder."""

from collections.abc import Iterable, Sequence
from typing import TypeVar

import numpy as np
from rustworkx import PyGraph, graph_greedy_color

from qiskit_noise_learning.gate_sets import ModelGateSet
from qiskit_noise_learning.math import IndexedMatrix
from qiskit_noise_learning.models import CompleteFidelityModel, FidelityModel
from qiskit_noise_learning.sequences import InstructionSequence, Path

from .experiment import Experiment

# the parameter in the fidelity model
ParameterIndex = TypeVar("ParameterIndex")

RANK_TOLERANCE = 1e-8
"""The tolerance to use when modifying the rank of design matrices."""


class ExperimentBuilder:
    r"""A description of a collection of experiments.

    This class describes, and can be used to build, sets of learning experiments for a
    :class:`ModelGateSet`. Experiments are represented as a collection of :class:`Path` instances, a
    collection of :class:`InstructionSequence` instances, and a many-to-many relation indicating
    which paths are traversed by which instruction sequences.

    Paths can be either bound (with an integer ``depth``) or unbound (with ``depth=None``). Unbound
    paths represent variable-depth experiments that are expanded at each depth when building.

    This class builds a design matrix, and methods for extending the collection can optionally
    choose to not add new paths that do not increase the rank of the matrix.

    Args:
        fidelity_model: A fidelity model for the gate set or a gate set. If it's a gate set, uses a
            :class:`CompleteFidelityModel`.
    """

    def __init__(self, fidelity_model: FidelityModel[ParameterIndex] | ModelGateSet):
        if isinstance(fidelity_model, ModelGateSet):
            fidelity_model = CompleteFidelityModel(fidelity_model)

        self._fidelity_model = fidelity_model
        self._design_matrix = IndexedMatrix[Path, ParameterIndex]()
        self._instruction_sequences: list[InstructionSequence] = []
        self._relations: set[tuple[int, int]] = set()

    @property
    def design_matrix(self) -> IndexedMatrix[Path, ParameterIndex]:
        """The design matrix.

        The row at index ``idx`` indicates how the sum of log fidelities traversed by
        ``self.paths[idx]`` is expressed as a linear combination of model parameters. For unbound
        paths, only the repeatable fragment is considered. For bound paths, the full path including
        start and end fragments scaled by depth is used.
        """
        return self._design_matrix

    @property
    def fidelity_model(self) -> FidelityModel[ParameterIndex]:
        """The fidelity model used for design matrix construction."""
        return self._fidelity_model

    @property
    def gate_set(self) -> ModelGateSet:
        """The model gate set the experiments are built for."""
        return self._fidelity_model.gate_set

    @property
    def instruction_sequences(self) -> list[InstructionSequence]:
        """The list of instruction sequences stored in this collection."""
        return self._instruction_sequences

    @property
    def is_complete(self) -> bool:
        """Whether all instruction sequences are complete."""
        return all(x.is_complete for x in self._instruction_sequences)

    @property
    def paths(self) -> list[Path]:
        """The list of paths stored in this collection."""
        return sorted(self.path_index_map, key=lambda k: self.path_index_map[k])

    @property
    def path_index_map(self) -> dict[Path, int]:
        """Dictionary mapping of paths to their index in ``self.paths``."""
        return self._design_matrix.row_index_map

    @property
    def relations(self) -> set[tuple[int, int]]:
        """The relations between the paths and instruction sequences in this collection.

        The relations are stored as a set of tuples of indices of the form
        ``(path_idx, inst_idx)``, which indicate the statement: ``self.paths[path_idx]`` is
        traversed by ``self.instruction_sequences[inst_idx]`` (assuming the right post-processing).
        """
        return self._relations

    def add_paths(
        self,
        sequence_iterator: Iterable[tuple[Path, InstructionSequence | None]],
        rank_reduce: bool = True,
        attempt_instruction_merge: bool = True,
    ):
        r"""Iteratively add paths to the builder.

        Handles both bound paths (with integer ``depth``) and unbound paths (with ``depth=None``).

        Args:
            sequence_iterator: An iterator over :class:`Path`\s and :class:`InstructionSequence`\s
                that traverse them. If the :class:`InstructionSequence` is ``None``, it is generated
                automatically from the path.
            rank_reduce: Whether to remove linearly dependent paths from ``self`` after adding the
                new paths.
            attempt_instruction_merge: Whether to attempt to merge the instruction sequence with a
                pre-existing instruction sequence.
        """

        new_paths = []
        rows = []
        new_instruction_sequences = []
        for path, instruction_sequence in sequence_iterator:
            if path not in self.path_index_map:
                new_paths.append(path)
                rows.append(self.fidelity_model.row_from_path(path))
                new_instruction_sequences.append(
                    instruction_sequence
                    if instruction_sequence is not None
                    else path.to_instruction_sequence()
                )

        self._design_matrix.add_rows(row_indices=new_paths, rows=rows, tol=RANK_TOLERANCE)
        if rank_reduce:
            self.rank_reduce()

        for path, instruction_sequence in zip(new_paths, new_instruction_sequences):
            if path in self.path_index_map:
                inst_idx = _add_instruction_to_list(
                    instruction_list=self.instruction_sequences,
                    instruction_sequence=instruction_sequence,
                    attempt_instruction_merge=attempt_instruction_merge,
                )
                self._relations.add((self.path_index_map[path], inst_idx))

    def complete(self):
        """Complete all internal instruction sequences."""
        self._instruction_sequences = [x.complete() for x in self._instruction_sequences]

    def generate_instruction_sequences(
        self, depths: list[int] | None = None
    ) -> list[InstructionSequence]:
        r"""Generate a list of instruction sequences from this experiment builder.

        Unbound instruction sequences are expanded at each depth. Bound instruction sequences are
        included as-is.

        Args:
            depths: A list of depths to generate sequences for. Required if there are any unbound
                instruction sequences.

        Returns:
            A list of instruction sequences.

        Raises:
            ValueError: If there are any unbound instruction sequences and no depths are specified.
        """
        has_unbound = any(seq.is_unbound for seq in self._instruction_sequences)
        if has_unbound and depths is None:
            raise ValueError("At least one depth is required for unbound instruction sequences.")

        instruction_sequences = []
        for seq in self._instruction_sequences:
            if seq.is_unbound:
                instruction_sequences.extend(seq.bind_at(d) for d in depths)
            else:
                instruction_sequences.append(seq)

        return instruction_sequences

    def build(self, depths: list[int], shots: int) -> Experiment:
        """Build an :class:`Experiment` from this builder.

        Expands all unbound instruction sequences at the given depths and constructs analysis paths
        as the Cartesian product of unbound paths and depths, plus any bound paths.

        Args:
            depths: The depths at which to expand unbound sequences and paths.
            shots: The number of shots per randomization.

        Returns:
            An :class:`Experiment` ready to pass to :meth:`.CircuitGenerator.generate`.
        """
        sequences = [x.complete() for x in self.generate_instruction_sequences(depths=depths)]

        paths = []
        for path in self.paths:
            if path.is_unbound:
                paths.extend(path.bind_at(d) for d in depths)
            else:
                paths.append(path)

        return Experiment(
            sequences=sequences,
            paths=paths,
            fidelity_model=self._fidelity_model,
            shots=shots,
        )

    def identify_relations(
        self, attempt_instruction_extension: bool = True
    ) -> set[tuple[int, int]]:
        """Identify new relations amongst the existing paths and instruction sequences.

        Args:
            attempt_instruction_extension: Whether or not to extend the definition of compatible
                instruction sequences to traverse new paths.

        Returns:
            A set containing the newly added relations.
        """
        return _identify_relations(
            path_index_map=self.path_index_map,
            instruction_list=self.instruction_sequences,
            relations=self.relations,
            attempt_instruction_extension=attempt_instruction_extension,
        )

    def merge_instruction_sequences(self):
        """Merge instruction sequences."""
        new_sequences, colors = minimize_instruction_sequences(self._instruction_sequences)
        self._instruction_sequences = new_sequences
        self._relations = {(path_idx, colors[inst_idx]) for path_idx, inst_idx in self._relations}

    def rank_reduce(self):
        """Reduce the collection of paths to a maximal linearly independent set."""
        new_matrix = self._design_matrix.linearly_independent_rows(tol=RANK_TOLERANCE)

        old_to_new_map = {}
        for path, data_idx in new_matrix.row_index_map.items():
            old_to_new_map[self._design_matrix.row_index_map[path]] = data_idx

        new_relations = set()
        for path_idx, inst_idx in self._relations:
            if path_idx in old_to_new_map:
                new_relations.add((old_to_new_map[path_idx], inst_idx))

        self._relations = new_relations
        self._design_matrix = new_matrix
        self.remove_unused_instruction_sequences()

    def remove_unused_instruction_sequences(self):
        """Remove any instruction sequences not referenced in the relations.

        Note that this method may result in re-indexing of instruction sequences.
        """

        idxs_to_keep = set(idx for _, idx in self._relations)

        new_instruction_sequences = []
        index_map = {}
        for idx, instruction_sequence in enumerate(self.instruction_sequences):
            if idx in idxs_to_keep:
                index_map[idx] = len(new_instruction_sequences)
                new_instruction_sequences.append(instruction_sequence)

        new_relations = set(
            (path_idx, index_map[inst_idx]) for path_idx, inst_idx in self.relations
        )

        self._relations = new_relations
        self._instruction_sequences = new_instruction_sequences


def _identify_relations(
    path_index_map: dict[Path, int],
    instruction_list: list[InstructionSequence],
    relations: set[tuple[int, int]],
    attempt_instruction_extension: bool = True,
) -> set[tuple[int, int]]:
    """A generalized relation identification function.

    Note that this function modifies ``instruction_list`` and ``relations`` in place.

    Args:
        path_index_map: A dict of :class:`Path` instances mapped to an index.
        instruction_list: A list of :class:`InstructionSequence` instances.
        relations: A set of pairs of indices indicating which paths are traversed by which
            instruction sequences.
        attempt_instruction_extension: Whether or not to extend the definition of compatible
            instruction sequences to traverse new paths.

    Returns:
        A set of newly identified relations.
    """

    new_relations = set()
    for path, path_idx in path_index_map.items():
        for inst_idx, instruction_sequence in enumerate(instruction_list):
            if (new_relation := (path_idx, inst_idx)) in relations:
                continue

            if attempt_instruction_extension:
                new_instruction = path.extend_permutations(instruction_sequence)
                if new_instruction is None:
                    continue

                new_relations.add(new_relation)
                relations.add(new_relation)
                instruction_list[inst_idx] = new_instruction
            else:
                if path.is_traversed_by(instruction_sequence):
                    new_relations.add(new_relation)
                    relations.add(new_relation)

    return new_relations


def _add_instruction_to_list(
    instruction_list: list[InstructionSequence],
    instruction_sequence: InstructionSequence,
    attempt_instruction_merge: bool = True,
) -> int:
    """Add an instruction sequence to a list, optionally merging with an existing entry.

    Note that this method modifies ``instruction_list`` in place.

    Args:
        instruction_list: A list of :class:`InstructionSequence` instances.
        instruction_sequence: The :class:`InstructionSequence` instance to add.
        attempt_instruction_merge: Whether to attempt to merge.

    Returns:
        The index ``instruction_sequence`` is inserted into ``instruction_list``, either through
        merging or appending.
    """
    if attempt_instruction_merge:
        for idx, existing_instruction in enumerate(instruction_list):
            if existing_instruction.is_mergeable_with(instruction_sequence):
                instruction_list[idx] = existing_instruction.merge(instruction_sequence)
                return idx

    instruction_list.append(instruction_sequence)
    return len(instruction_list) - 1


def minimize_instruction_sequences(
    sequences: Sequence[InstructionSequence],
) -> tuple[list[InstructionSequence], dict[int, int]]:
    """Return a minimal list of instruction sequences by coloring mergeable instruction sequences.

    Given ``sequences``, this function constructs a matrix such that the entry at ``(i, j)``
    is ``False`` if ``sequences[i]`` is mergeable with ``sequences[j]``. By constructing and
    coloring a graph from this matrix, the ``sequences`` are partitioned into mutually mergeable
    groups. The groups are then merged into a single sequence ordered by color.

    This method also returns a dictionary from original index to color.

    Args:
        sequences: The sequences to merge.

    Returns:
        A minimal list of instruction sequences sorted by color and a dictionary from original index
        in ``sequences`` to the color it was assigned.
    """
    adjacency_mat = np.ones((len(sequences), len(sequences)), dtype=np.bool_)
    np.fill_diagonal(adjacency_mat, np.False_)
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            is_not_mergeable = not sequences[i].is_mergeable_with(sequences[j])
            adjacency_mat[(i, j)] = is_not_mergeable
            adjacency_mat[(j, i)] = is_not_mergeable

    colors = graph_greedy_color(PyGraph.from_adjacency_matrix(adjacency_mat.astype(np.float64)))

    minimized_sequences = {}
    for idx, color in colors.items():
        if (this_sequence := minimized_sequences.get(color)) is None:
            minimized_sequences[color] = sequences[idx]
            continue
        minimized_sequences[color] = this_sequence.merge(sequences[idx])

    return [v for _, v in sorted(minimized_sequences.items())], colors
