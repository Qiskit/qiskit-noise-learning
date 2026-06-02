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

"""GenerateInstructionSequences stage."""

from ...experiment import Experiment
from .add_instruction_sequences import AddInstructionSequences


class GenerateInstructionSequences(AddInstructionSequences):
    """Generate one instruction sequence per path via :meth:`~.Path.to_instruction_sequence`.

    For each path (optionally skipping paths already referenced by a relation), generates
    a corresponding instruction sequence and adds it along with a new relation entry.

    Args:
        bound_multiplier: Randomization multiplier for bound sequences.
        unbound_multiplier: Randomization multiplier for unbound sequences.
        skip_related_paths: If ``True`` (default), paths that already appear in a relation
            are not used to generate new sequences.
    """

    required_fields = ("paths",)
    populates_fields = ("instruction_sequences", "randomization_multipliers", "relations")

    def __init__(
        self,
        *,
        bound_multiplier: int = 1,
        unbound_multiplier: int = 1,
        skip_related_paths: bool = True,
    ):
        self._bound_multiplier = bound_multiplier
        self._unbound_multiplier = unbound_multiplier
        self._skip_related_paths = skip_related_paths

    def _run(self, experiment: Experiment) -> Experiment:
        existing_seqs = list(experiment.instruction_sequences or [])
        existing_mults = list(experiment.randomization_multipliers or [])
        existing_relations = (
            set(experiment.relations) if experiment.relations is not None else set()
        )

        related_path_indices = set()
        if self._skip_related_paths:
            related_path_indices = {path_idx for path_idx, _ in existing_relations}

        seq_offset = len(existing_seqs)
        new_seqs = []
        new_mults = []
        new_relations = set()

        for path_idx, path in enumerate(experiment.paths):
            if path_idx in related_path_indices:
                continue

            seq = path.to_instruction_sequence()
            mult = self._bound_multiplier if not seq.is_unbound else self._unbound_multiplier
            new_seqs.append(seq)
            new_mults.append(mult)
            new_relations.add((path_idx, seq_offset + len(new_seqs) - 1))

        existing_seqs.extend(new_seqs)
        existing_mults.extend(new_mults)

        return experiment.replace(
            validate=False,
            instruction_sequences=existing_seqs,
            randomization_multipliers=existing_mults,
            relations=existing_relations | new_relations,
        )
