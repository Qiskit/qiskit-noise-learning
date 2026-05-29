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

"""IdentifyRelations stage."""

from __future__ import annotations

from qiskit_noise_learning.sequences import InstructionSequence, Path

from ..experiment import Experiment
from ..stage import ExperimentBuilderStage


class IdentifyRelations(ExperimentBuilderStage):
    """Identify new relations amongst existing paths and instruction sequences.

    Iterates over all path/sequence pairs and identifies which sequences traverse which paths.
    Optionally extends instruction sequence permutations to enable new traversals.

    Args:
        attempt_instruction_extension: Whether to extend instruction sequence permutations
            to traverse new paths.
    """

    required_fields = ("paths", "instruction_sequences")

    def __init__(self, attempt_instruction_extension: bool = True):
        self._attempt_instruction_extension = attempt_instruction_extension

    def _run(self, experiment: Experiment) -> Experiment:
        sequences = list(experiment.instruction_sequences)
        relations = set(experiment.relations) if experiment.relations else set()

        _identify_relations(
            paths=experiment.paths,
            instruction_list=sequences,
            relations=relations,
            attempt_instruction_extension=self._attempt_instruction_extension,
        )

        return experiment.replace(
            validate=False, instruction_sequences=sequences, relations=relations
        )


def _identify_relations(
    paths: list[Path],
    instruction_list: list[InstructionSequence],
    relations: set[tuple[int, int]],
    attempt_instruction_extension: bool = True,
) -> None:
    """Identify and add new relations in place.

    Args:
        paths: The list of paths.
        instruction_list: The list of instruction sequences (may be modified in place).
        relations: The set of relations (modified in place).
        attempt_instruction_extension: Whether to extend instruction sequences.
    """
    for path_idx, path in enumerate(paths):
        for inst_idx, instruction_sequence in enumerate(instruction_list):
            if (path_idx, inst_idx) in relations:
                continue

            if attempt_instruction_extension:
                new_instruction = path.extend_permutations(instruction_sequence)
                if new_instruction is None:
                    continue
                relations.add((path_idx, inst_idx))
                instruction_list[inst_idx] = new_instruction
            else:
                if path.is_traversed_by(instruction_sequence):
                    relations.add((path_idx, inst_idx))
