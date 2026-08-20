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

"""MergeInstructionSequences stage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import chain

import numpy as np
from rustworkx import PyGraph, graph_greedy_color

from qiskit_noise_learning.sequences import InstructionSequence
from qiskit_noise_learning.sequences.apply_gate import ApplyGate
from qiskit_noise_learning.sequences.partial_pauli_permutation import (
    PartialPauliPermutation,
    consistency_matrix,
)

from ..experiment import Experiment
from ..experiment_builder_stage import ExperimentBuilderStage


class MergeInstructionSequences(ExperimentBuilderStage):
    """Merge instruction sequences into a minimal set via graph coloring.

    Constructs a conflict graph where sequences that cannot be merged share an edge,
    then colors the graph to find groups of mutually mergeable sequences. Each group
    is merged into a single sequence.
    """

    required_fields = ("instruction_sequences", "randomization_multipliers", "relations")

    def _run(self, experiment: Experiment) -> Experiment:
        new_sequences, colors = _minimize_instruction_sequences(experiment.instruction_sequences)
        new_relations = {
            (path_idx, colors[inst_idx]) for path_idx, inst_idx in experiment.relations
        }

        old_multipliers = experiment.randomization_multipliers
        new_multipliers = [1] * len(new_sequences)
        for old_idx, color in colors.items():
            new_multipliers[color] = max(new_multipliers[color], old_multipliers[old_idx])

        return experiment.replace(
            validate=False,
            instruction_sequences=new_sequences,
            randomization_multipliers=new_multipliers,
            relations=new_relations,
        )


def _minimize_instruction_sequences(
    sequences: Sequence[InstructionSequence],
) -> tuple[list[InstructionSequence], dict[int, int]]:
    """Return a minimal list of instruction sequences by coloring mergeable sequences.

    Args:
        sequences: The sequences to merge.

    Returns:
        A minimal list of instruction sequences and a dictionary from original index to color.
    """
    adjacency_mat = _conflict_matrix(sequences)

    colors = graph_greedy_color(PyGraph.from_adjacency_matrix(adjacency_mat.astype(np.float64)))

    minimized_sequences = {}
    for idx, color in colors.items():
        if (this_sequence := minimized_sequences.get(color)) is None:
            minimized_sequences[color] = sequences[idx]
            continue
        minimized_sequences[color] = this_sequence.merge(sequences[idx])

    return [v for _, v in sorted(minimized_sequences.items())], colors


def _conflict_matrix(sequences: Sequence[InstructionSequence]) -> np.ndarray:
    """Return the boolean conflicts matrix where ``True`` marks a non-mergeable pair.

    Equivalent to filling the matrix with pairwise
    :meth:`~.InstructionSequence.is_mergeable_with`, but computed in bulk: only sequences that share
    the same structure (fragment depth and per-position instruction layout) can merge, and within
    such a group merging reduces to per-qubit consistency of the ``PartialPauliPermutation``\\s
    (gate applications always merge).
    """
    num_sequences = len(sequences)
    conflicts = ~np.eye(num_sequences, dtype=np.bool_)

    groups: dict[tuple, list[int]] = defaultdict(list)
    for idx, sequence in enumerate(sequences):
        groups[_structure_key(sequence)].append(idx)

    consistency = consistency_matrix()
    for indices in groups.values():
        if len(indices) < 2:
            continue

        permutations = np.stack([_permutation_indices(sequences[i]) for i in indices])
        mergeable = np.ones((len(indices), len(indices)), dtype=np.bool_)
        for column in permutations.T:
            # a column where every sequence agrees imposes no constraint
            if (column == column[0]).all():
                continue
            mergeable &= consistency[np.ix_(column, column)]

        conflicts[np.ix_(indices, indices)] = ~mergeable

    return conflicts


def _structure_key(sequence: InstructionSequence) -> tuple:
    """A hashable key identifying the mergeable structure of a sequence."""

    def tokens(fragment: Sequence) -> tuple:
        return tuple(
            instr.gate_name if isinstance(instr, ApplyGate) else None for instr in fragment
        )

    return (
        sequence.fragment_depth,
        tokens(sequence.start_fragment),
        tokens(sequence.repeatable_fragment),
        tokens(sequence.end_fragment),
    )


def _permutation_indices(sequence: InstructionSequence) -> np.ndarray:
    """The concatenated per-qubit permutation indices of a sequence's partial permutations."""
    columns = [
        instr.partial_permutation_indices
        for instr in chain(
            sequence.start_fragment, sequence.repeatable_fragment, sequence.end_fragment
        )
        if isinstance(instr, PartialPauliPermutation)
    ]
    return np.concatenate(columns) if columns else np.zeros(0, dtype=np.int8)
