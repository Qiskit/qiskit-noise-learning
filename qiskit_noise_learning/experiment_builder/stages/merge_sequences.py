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

from qiskit_noise_learning.sequences import InstructionSequence, merge_groups

from ..experiment import Experiment
from ..experiment_builder_stage import ExperimentBuilderStage


class MergeInstructionSequences(ExperimentBuilderStage):
    """Merge instruction sequences into a smaller set.

    The sequences that can be merged with each other are grouped together by
    :func:`~.merge_groups`, and every group is merged into a single sequence.
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
    """Return a smaller list of instruction sequences by merging the mergeable ones together.

    Args:
        sequences: The sequences to merge.

    Returns:
        A reduced list of instruction sequences, and a dictionary from the index of each input
        sequence to the index of the merged sequence it contributed to.
    """
    minimized_sequences = []
    merged_indices = {}
    for group in merge_groups(sequences):
        merged = sequences[group[0]]
        for idx in group[1:]:
            merged = merged.merge(sequences[idx])

        merged_indices.update(dict.fromkeys(group, len(minimized_sequences)))
        minimized_sequences.append(merged)

    return minimized_sequences, merged_indices
