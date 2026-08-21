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

import numpy as np
from rustworkx import PyGraph, graph_greedy_color

from qiskit_noise_learning.sequences import InstructionSequence, partition_instruction_sequences

from ..experiment import Experiment
from ..experiment_builder_stage import ExperimentBuilderStage


class MergeInstructionSequences(ExperimentBuilderStage):
    """Merge instruction sequences into a smaller set.

    This stage uses a combination of strategies: partitioning the instruction sequences such that
    elements of disjoint sets are never mergeable, and then using graph colouring to determine a
    merging strategy within each partition.
    """

    required_fields = ("instruction_sequences", "randomization_multipliers", "relations")

    def _run(self, experiment: Experiment) -> Experiment:
        new_sequences, merged_indices = _minimize_instruction_sequences(
            experiment.instruction_sequences
        )
        new_relations = {
            (path_idx, merged_indices[inst_idx]) for path_idx, inst_idx in experiment.relations
        }

        old_multipliers = experiment.randomization_multipliers
        new_multipliers = [1] * len(new_sequences)
        for old_idx, new_idx in merged_indices.items():
            new_multipliers[new_idx] = max(new_multipliers[new_idx], old_multipliers[old_idx])

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

    The sequences are first partitioned into groups whose members can only ever merge within their
    own group, and each group is colored on its own. Since every pair of sequences drawn from two
    different groups conflicts, this yields the same number of merged sequences as coloring the
    conflict graph of all sequences at once, but colors several small graphs instead of one large
    one.

    Args:
        sequences: The sequences to merge.

    Returns:
        A minimal list of instruction sequences, and a dictionary from the index of each input
        sequence to the index of the merged sequence it contributed to.
    """
    minimized_sequences = []
    merged_indices = {}
    for group_indices, mergeable in partition_instruction_sequences(sequences):
        colors = graph_greedy_color(PyGraph.from_adjacency_matrix((~mergeable).astype(np.float64)))

        indices_by_color: dict[int, list[int]] = defaultdict(list)
        for node, color in colors.items():
            indices_by_color[color].append(group_indices[node])

        for indices in indices_by_color.values():
            merged = sequences[indices[0]]
            for idx in indices[1:]:
                merged = merged.merge(sequences[idx])

            merged_indices.update(dict.fromkeys(indices, len(minimized_sequences)))
            minimized_sequences.append(merged)

    return minimized_sequences, merged_indices
