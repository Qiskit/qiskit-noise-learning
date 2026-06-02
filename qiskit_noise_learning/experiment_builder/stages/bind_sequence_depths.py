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

"""BindSequenceDepths stage."""

from __future__ import annotations

from ..experiment import Experiment
from ..stage import ExperimentBuilderStage


class BindSequenceDepths(ExperimentBuilderStage):
    """Expand unbound instruction sequences at the given depths.

    Each unbound instruction sequence is replaced by a bound copy at each depth. Bound
    sequences are kept as-is. The ``randomization_multipliers`` and ``relations`` are updated
    accordingly: the multiplier of an unbound sequence is propagated to all its bound
    expansions, and any relation pointing to an unbound sequence is fanned out to all the
    resulting bound sequences.

    Args:
        depths: The depths at which to bind unbound sequences.
    """

    required_fields = ("instruction_sequences", "randomization_multipliers")

    def __init__(self, depths: list[int]):
        self._depths = depths

    def _run(self, experiment: Experiment) -> Experiment:
        old_sequences = experiment.instruction_sequences
        old_multipliers = experiment.randomization_multipliers
        old_relations = experiment.relations

        new_sequences = []
        new_multipliers = []
        # Maps old sequence index to list of new indices
        index_map: dict[int, list[int]] = {}

        for old_idx, seq in enumerate(old_sequences):
            if seq.is_unbound:
                start = len(new_sequences)
                for d in self._depths:
                    new_sequences.append(seq.bind_at(d))
                    new_multipliers.append(old_multipliers[old_idx])
                index_map[old_idx] = list(range(start, len(new_sequences)))
            else:
                index_map[old_idx] = [len(new_sequences)]
                new_sequences.append(seq)
                new_multipliers.append(old_multipliers[old_idx])

        # Expand relations
        new_relations = None
        if old_relations is not None:
            new_relations = set()
            for path_idx, seq_idx in old_relations:
                for new_idx in index_map[seq_idx]:
                    new_relations.add((path_idx, new_idx))

        return experiment.replace(
            validate=False,
            instruction_sequences=new_sequences,
            randomization_multipliers=new_multipliers,
            relations=new_relations,
        )
