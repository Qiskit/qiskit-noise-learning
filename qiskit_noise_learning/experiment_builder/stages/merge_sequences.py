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

from collections.abc import Sequence

import numpy as np
from rustworkx import PyGraph, graph_greedy_color

from qiskit_noise_learning.sequences import InstructionSequence

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
