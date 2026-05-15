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
from itertools import chain
from typing import TypeVar

import numpy as np
from rustworkx import PyGraph, graph_greedy_color

from qiskit_noise_learning.gate_sets import ModelGateSet
from qiskit_noise_learning.math import IndexedMatrix
from qiskit_noise_learning.models import CompleteFidelityModel, FidelityModel
from qiskit_noise_learning.sequences import (
    InstructionPattern,
    InstructionSequence,
    Path,
    PathPattern,
)

from .experiment import Experiment

# the parameter in the fidelity model
ParameterIndex = TypeVar("ParameterIndex")

RANK_TOLERANCE = 1e-8
"""The tolerance to use when modifying the rank of design matrices."""


class ExperimentBuilder:
    r"""A description of a collection of experiments.

    This class describes, and can be used to build, sets of fixed-depth and variable-depth learning
    experiments for a :class:`ModelGateSet`. Fixed-depth experiments are represented as a collection
    of :class:`Path` instances, a collection of :class:`InstructionSequence` instances, and a
    many-to-many relation indicating which paths are traversed by which instruction sequences.

    Similarly, variable-depth experiments are represented by a collection of :class:`PathPattern`
    instances, a collection of :class:`InstructionPattern` sequences, and a many-to-many relation
    indicating which path patterns are traversed by which instruction patterns.

    This class builds design matrices for both fixed depth and variable depth experiments, and
    methods for extending the collection can optionally choose to not add new paths or path patterns
    that do not increase the rank of these matrices.

    Args:
        fidelity_model: A fidelity model for the gate set or a gate set. If it's a gate set, uses a
            :class:`CompleteFidelityModel`.
    """

    def __init__(self, fidelity_model: FidelityModel[ParameterIndex] | ModelGateSet):
        if isinstance(fidelity_model, ModelGateSet):
            fidelity_model = CompleteFidelityModel(fidelity_model)

        self._fidelity_model = fidelity_model

        # initialize pattern fields
        self._multiplicative_design_matrix = IndexedMatrix[PathPattern, ParameterIndex]()
        self._instruction_patterns = []
        self._pattern_relations = set()

        # initialize sequence fields
        self._additive_design_matrix = IndexedMatrix[Path, ParameterIndex]()
        self._instruction_sequences = []
        self._sequence_relations = set()

    @property
    def additive_design_matrix(self) -> IndexedMatrix[Path, ParameterIndex]:
        """The additive precision design matrix.

        The row at index ``idx`` indicates how the sum of log fidelities traversed by
        ``self.paths[idx]`` is expressed as a linear combination of model parameters.
        """
        return self._additive_design_matrix

    @property
    def fidelity_model(self) -> FidelityModel[ParameterIndex]:
        """The fidelity model used for design matrix construction."""
        return self._fidelity_model

    @property
    def gate_set(self) -> ModelGateSet:
        """The model gate set the experiments are built for."""
        return self._fidelity_model.gate_set

    @property
    def instruction_patterns(self) -> list[InstructionPattern]:
        """The list of instruction patterns stored in this collection."""
        return self._instruction_patterns

    @property
    def instruction_sequences(self) -> list[InstructionSequence]:
        """The list of instruction sequences stored in this collection."""
        return self._instruction_sequences

    @property
    def is_complete(self) -> bool:
        """Whether all instruction patterns and sequences are complete."""
        return all(
            x.is_complete for x in chain(self._instruction_patterns, self._instruction_sequences)
        )

    @property
    def multiplicative_design_matrix(self) -> IndexedMatrix[PathPattern, ParameterIndex]:
        """The multiplicative precision design matrix.

        The row at index ``idx`` indicates how the sum of log fidelities traversed by the repeatable
        fragment of ``self.path_patterns[idx]`` is expressed as a linear combination of model
        parameters.
        """
        return self._multiplicative_design_matrix

    @property
    def paths(self) -> list[Path]:
        """The list of path patterns stored in this collection."""
        return sorted(self.path_index_map, key=lambda k: self.path_index_map[k])

    @property
    def path_index_map(self) -> dict[Path, int]:
        """Dictionary mapping of paths to their index in ``self.paths``."""
        return self._additive_design_matrix.row_index_map

    @property
    def path_patterns(self) -> list[PathPattern]:
        """The list of path patterns stored in this collection."""
        index_map = self.path_pattern_index_map
        return sorted(index_map, key=lambda k: index_map[k])

    @property
    def path_pattern_index_map(self) -> dict[PathPattern, int]:
        """Dictionary mapping of path patterns to their index in ``self.path_patterns``."""
        return self._multiplicative_design_matrix.row_index_map

    @property
    def pattern_relations(self) -> set[tuple[int, int]]:
        """The relations between the path patterns and instruction patterns in this collection.

        The relations are stored as a set of tuples of indices of the form
        ``(path_idx, inst_idx)``, which indicate the statement: ``self.path_patterns[path_idx]`` is
        traversed by ``self.instruction_patterns[inst_idx]`` (assuming the right post-processing).
        """
        return self._pattern_relations

    @property
    def sequence_relations(self) -> set[tuple[int, int]]:
        """The relations between the paths and instruction sequences in this collection.

        The relations are stored as a set of tuples of indices of the form
        ``(path_idx, inst_idx)``, which indicate the statement: ``self.path[path_idx]`` is
        traversed by ``self.instruction_sequences[inst_idx]`` (assuming the right post-processing).
        """
        return self._sequence_relations

    def add_paths(
        self,
        sequence_iterator: Iterable[tuple[Path, InstructionSequence | None]],
        rank_reduce: bool = True,
        attempt_instruction_merge: bool = True,
    ):
        r"""Iteratively add paths to the builder.

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
            # skip already present patterns
            if path not in self.path_index_map:
                new_paths.append(path)
                rows.append(self.fidelity_model.row_from_path(path))
                new_instruction_sequences.append(
                    instruction_sequence if instruction_sequence else path.to_instruction_sequence()
                )

        # add rows and rank reduce
        self._additive_design_matrix.add_rows(row_indices=new_paths, rows=rows, tol=RANK_TOLERANCE)
        if rank_reduce:
            self.rank_reduce_paths()

        # add new instruction sequences and new relations
        for path, instruction_sequence in zip(new_paths, new_instruction_sequences):
            if path in self.path_index_map:
                inst_idx = _add_instruction_to_list(
                    instruction_like_list=self.instruction_sequences,
                    instruction_like=instruction_sequence,
                    attempt_instruction_merge=attempt_instruction_merge,
                )
                self._sequence_relations.add((self.path_index_map[path], inst_idx))

    def add_path_patterns(
        self,
        pattern_iterator: Iterable[tuple[PathPattern, InstructionPattern | None]],
        rank_reduce: bool = True,
        attempt_instruction_merge: bool = False,
    ):
        r"""Iteratively add patterns to the builder.

        Args:
            pattern_iterator: An iterator over :class:`PathPattern`\s and
                :class:`InstructionPattern`\s that traverse them. If the :class:`InstructionPattern`
                is ``None``, it is generated automatically from the path.
            rank_reduce: Whether to remove linearly dependent path patterns from ``self`` after
                adding the new patterns.
            attempt_instruction_merge: Whether to attempt to merge the instruction pattern with a
                pre-existing instruction pattern.
        """

        new_path_patterns = []
        rows = []
        new_instruction_patterns = []
        for path_pattern, instruction_pattern in pattern_iterator:
            # skip already present patterns
            if path_pattern not in self.path_pattern_index_map:
                new_path_patterns.append(path_pattern)
                rows.append(self.fidelity_model.multiplicative_row_from_path_pattern(path_pattern))
                new_instruction_patterns.append(
                    instruction_pattern
                    if instruction_pattern
                    else path_pattern.to_instruction_pattern()
                )

        # add rows and rank reduce
        self._multiplicative_design_matrix.add_rows(
            row_indices=new_path_patterns, rows=rows, tol=RANK_TOLERANCE
        )
        if rank_reduce:
            self.rank_reduce_path_patterns()

        # add new instruction patterns and new relations
        for path_pattern, instruction_pattern in zip(new_path_patterns, new_instruction_patterns):
            if path_pattern in self.path_pattern_index_map:
                inst_idx = _add_instruction_to_list(
                    instruction_like_list=self.instruction_patterns,
                    instruction_like=instruction_pattern,
                    attempt_instruction_merge=attempt_instruction_merge,
                )
                self._pattern_relations.add((self.path_pattern_index_map[path_pattern], inst_idx))

    def complete(self):
        """Complete all internal instruction patterns and sequences."""
        self._instruction_patterns = [x.complete() for x in self._instruction_patterns]
        self._instruction_sequences = [x.complete() for x in self._instruction_sequences]

    def generate_instruction_sequences(
        self, depths: list[int] | None = None
    ) -> list[InstructionSequence]:
        r"""Generate a list of instruction sequences from this experiment builder.

        Args:
            depths: A list of depths to generate sequences for.

        Returns:
            A list of instruction sequences.

        Raises:
            ValueError: If there are any instruction patterns in this builder and no depths are
                specified.
        """
        if self.instruction_patterns and depths is None:
            raise ValueError("At least one depth is required for path patterns.")

        instruction_sequences = [
            InstructionSequence(inst_pattern, d)
            for d in depths
            for inst_pattern in self.instruction_patterns
        ]
        instruction_sequences.extend(sequence for sequence in self.instruction_sequences)

        return instruction_sequences

    def build(self, depths: list[int], shots: int) -> Experiment:
        """Build a fully-expanded :class:`Experiment` from this builder.

        Expands all instruction patterns at the given depths and constructs analysis paths as the
        Cartesian product of path patterns and depths, plus any fixed-depth paths.

        Args:
            depths: The depths at which to expand instruction patterns.
            shots: The number of shots per randomization.

        Returns:
            An :class:`Experiment` ready to pass to :meth:`.CircuitGenerator.generate`.
        """
        sequences = self.generate_instruction_sequences(depths=depths)
        paths = [Path(p, d) for p in self.path_patterns for d in depths]
        paths.extend(self.paths)
        return Experiment(
            sequences=sequences,
            paths=paths,
            fidelity_model=self._fidelity_model,
            shots=shots,
        )

    def identify_pattern_relations(
        self, attempt_instruction_extension: bool = True
    ) -> set[tuple[int, int]]:
        """Identify new relations amongst the existing path patterns and instruction patterns.

        Args:
            attempt_instruction_extension: Whether or not to extend the definition of compatible
                instruction patterns to traverse new paths.

        Returns:
            A set containing the newly added pattern relations.
        """
        return _identify_relations(
            path_like_index_map=self.path_pattern_index_map,
            instruction_like_list=self.instruction_patterns,
            relations=self.pattern_relations,
            attempt_instruction_extension=attempt_instruction_extension,
        )

    def identify_sequence_relations(
        self, attempt_instruction_extension: bool = True
    ) -> set[tuple[int, int]]:
        """Identify new relations amongst the existing paths and instruction sequences.

        Args:
            attempt_instruction_extension: Whether or not to extend the definition of compatible
                instruction sequences to traverse new paths.

        Returns:
            A set containing the newly added sequence relations.
        """
        return _identify_relations(
            path_like_index_map=self.path_index_map,
            instruction_like_list=self.instruction_sequences,
            relations=self.sequence_relations,
            attempt_instruction_extension=attempt_instruction_extension,
        )

    def identify_relations(
        self, attempt_instruction_extension: bool = True
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Identify new pattern and sequence relations.

        Args:
            attempt_instruction_extension: Whether or not to extend the definitions of instruction
                patterns and sequences to traverse additional path patterns and paths, respectively.

        Returns:
            A tuple of sets giving, respectively, the newly generated pattern and sequence
            relations.
        """

        return (
            self.identify_pattern_relations(attempt_instruction_extension),
            self.identify_sequence_relations(attempt_instruction_extension),
        )

    def merge_instruction_patterns(self):
        """Merge instruction patterns."""
        new_patterns, colors = minimize_instruction_patterns(self._instruction_patterns)
        self._instruction_patterns = new_patterns
        self._pattern_relations = {
            (pp_idx, colors[ip_idx]) for pp_idx, ip_idx in self._pattern_relations
        }

    def rank_reduce_paths(self):
        """Reduce the collection of paths to a maximal linearly independent set."""
        new_matrix = self._additive_design_matrix.linearly_independent_rows(tol=RANK_TOLERANCE)

        # re-index existing relations - previously existing experiments may have been dropped
        old_to_new_map = {}
        for path, data_idx in new_matrix.row_index_map.items():
            old_to_new_map[self._additive_design_matrix.row_index_map[path]] = data_idx

        new_relations = set()
        for path_idx, inst_idx in self._sequence_relations:
            if path_idx in old_to_new_map:
                new_relations.add((old_to_new_map[path_idx], inst_idx))

        self._sequence_relations = new_relations
        self._additive_design_matrix = new_matrix
        self.remove_unused_instruction_sequences()

    def rank_reduce_path_patterns(self):
        """Reduce the collection of path patterns to a maximal linearly independent set."""
        new_matrix = self._multiplicative_design_matrix.linearly_independent_rows(
            tol=RANK_TOLERANCE
        )

        # re-index existing relations - previously existing experiments may have been dropped
        old_to_new_map = {}
        for path_pattern, data_idx in new_matrix.row_index_map.items():
            old_to_new_map[self._multiplicative_design_matrix.row_index_map[path_pattern]] = (
                data_idx
            )

        new_relations = set()
        for path_idx, inst_idx in self._pattern_relations:
            if path_idx in old_to_new_map:
                new_relations.add((old_to_new_map[path_idx], inst_idx))

        self._pattern_relations = new_relations
        self._multiplicative_design_matrix = new_matrix
        self.remove_unused_instruction_patterns()

    def remove_unused_instruction_patterns(self):
        """Remove any instruction patterns not referenced in the pattern relations.

        Note that this method may result in re-indexing of instruction patterns.
        """

        idxs_to_keep = set(idx for _, idx in self._pattern_relations)

        new_instruction_patterns = []
        index_map = {}
        for idx, instruction_pattern in enumerate(self.instruction_patterns):
            if idx in idxs_to_keep:
                index_map[idx] = len(new_instruction_patterns)
                new_instruction_patterns.append(instruction_pattern)

        new_relations = set(
            (path_idx, index_map[inst_idx]) for path_idx, inst_idx in self.pattern_relations
        )

        self._pattern_relations = new_relations
        self._instruction_patterns = new_instruction_patterns

    def remove_unused_instruction_sequences(self):
        """Remove any instruction sequences not referenced in the sequence relations.

        Note that this method may result in re-indexing of instruction sequences.
        """

        idxs_to_keep = set(idx for _, idx in self._sequence_relations)

        new_instruction_sequences = []
        index_map = {}
        for idx, instruction_sequence in enumerate(self.instruction_sequences):
            if idx in idxs_to_keep:
                index_map[idx] = len(new_instruction_sequences)
                new_instruction_sequences.append(instruction_sequence)

        new_relations = set(
            (path_idx, index_map[inst_idx]) for path_idx, inst_idx in self.sequence_relations
        )

        self._sequence_relations = new_relations
        self._instruction_sequences = new_instruction_sequences


def _identify_relations(
    path_like_index_map: dict[PathPattern | Path, int],
    instruction_like_list: list[InstructionPattern] | list[InstructionSequence],
    relations: set[tuple[int, int]],
    attempt_instruction_extension: bool = True,
) -> set[tuple[int, int]]:
    """A generalized relation identification function for either patterns or sequences.

    Note that this function modifies ``instruction_like_list`` and ``relations`` in place.

    Args:
        path_like_index_map: A dict of :class:`PathPattern` or :class:`Path` instances mapped to an
            index.
        instruction_like_list: A list of :class:`InstructionPattern` or :class:`InstructionSequence`
            instances. The pattern or sequence type must match that of ``path_like_list``.
        relations: A set of pairs of indices indicating which "path"s are traversed by which
            "instruction"s.
        attempt_instruction_extension: Whether or not to extend the definition of compatible
            "instruction"s to traverse new "path"s.

    Returns:
        A set of newly identified relations.
    """

    new_relations = set()
    for path_like, path_idx in path_like_index_map.items():
        for inst_idx, instruction_like in enumerate(instruction_like_list):
            if (new_relation := (path_idx, inst_idx)) in relations:
                continue

            if attempt_instruction_extension:
                new_instruction = path_like.extend_permutations(instruction_like)
                if new_instruction is None:
                    continue

                new_relations.add(new_relation)
                relations.add(new_relation)
                instruction_like_list[inst_idx] = new_instruction
            else:
                if path_like.is_traversed_by(instruction_like):
                    new_relations.add(new_relation)
                    relations.add(new_relation)

    return new_relations


def _add_instruction_to_list(
    instruction_like_list: list[InstructionPattern] | list[InstructionSequence],
    instruction_like: InstructionPattern | InstructionSequence,
    attempt_instruction_merge: bool = True,
) -> int:
    """A generalized method for adding an instruction pattern or sequence to a list.

    This method works for instruction patterns or instruction sequences. If
    ``attempt_instruction_merge``, it will greedily attempt to merge ``instruction_like`` into a
    pre-existing entry in ``instruction_like_list``.

    Note that this method modifies ``instruction_like_list`` in place.

    Args:
        instruction_like_list: A list of :class:`InstructionPattern` or :class:`InstructionSequence`
            instances.
        instruction_like: A :class:`InstructionPattern` or :class:`InstructionSequence` instance.
            The type should match the elements of ``instruction_like_list``.
        attempt_instruction_merge: Whether to attempt to merge.

    Returns:
        The index ``instruction_like`` is inserted into ``instruction_like_list``, either through
        merging or appending.
    """
    if attempt_instruction_merge:
        for idx, existing_instruction in enumerate(instruction_like_list):
            if existing_instruction.is_mergeable_with(instruction_like):
                instruction_like_list[idx] = existing_instruction.merge(instruction_like)
                return idx

    # if no merge, add to the end
    instruction_like_list.append(instruction_like)
    return len(instruction_like_list) - 1


def minimize_instruction_patterns(
    patterns: Sequence[InstructionPattern],
) -> tuple[list[InstructionPattern], dict[int, int]]:
    """Return a minimal list of instruction patterns by coloring mergeable instruction patterns.

    Given ``patterns``, this function constructs a matrix such that the entry at ``(i, j)``
    is ``False`` if ``patterns[i]`` is mergeable with ``patterns[j]``. By constructing and
    coloring a graph from this matrix, the ``patterns`` are partitioned into mutually mergeable
    groups. The groups are then merged into a single pattern ordered by color.

    This method also returns a dictionary from original index to color.

    Args:
        patterns: The patterns to merge.

    Returns:
        A minimal list of instruction patterns sorted by color and a dictionary from original index
        in ``patterns`` to the color it was assigned.
    """
    adjacency_mat = np.ones((len(patterns), len(patterns)), dtype=np.bool_)
    np.fill_diagonal(adjacency_mat, np.False_)
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            is_not_mergeable = not patterns[i].is_mergeable_with(patterns[j])
            adjacency_mat[(i, j)] = is_not_mergeable
            adjacency_mat[(j, i)] = is_not_mergeable

    colors = graph_greedy_color(PyGraph.from_adjacency_matrix(adjacency_mat.astype(np.float64)))

    minimized_patterns = {}
    for idx, color in colors.items():
        if (this_pattern := minimized_patterns.get(color)) is None:
            minimized_patterns[color] = patterns[idx]
            continue
        minimized_patterns[color] = this_pattern.merge(patterns[idx])

    return [v for _, v in sorted(minimized_patterns.items())], colors
