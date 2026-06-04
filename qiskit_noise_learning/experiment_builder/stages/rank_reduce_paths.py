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

"""RankReducePaths stage."""

from __future__ import annotations

from ..experiment import Experiment
from ..stage import ExperimentBuilderStage

RANK_TOLERANCE = 1e-8


class RankReducePaths(ExperimentBuilderStage):
    """Reduce the paths to a maximal linearly independent set.

    Removes linearly dependent paths from the experiment's design matrix, updates relations
    accordingly, and removes any instruction sequences no longer referenced.

    Args:
        tol: Tolerance for determining linear independence.
    """

    required_fields = ("paths",)

    def __init__(self, tol: float = RANK_TOLERANCE):
        self._tol = tol

    def _run(self, experiment: Experiment) -> Experiment:
        design_matrix = experiment.design_matrix
        new_matrix = design_matrix.linearly_independent_rows(tol=self._tol)

        old_path_index_map = design_matrix.row_index_map
        new_path_index_map = new_matrix.row_index_map

        old_to_new_map = {}
        for path, new_idx in new_path_index_map.items():
            old_to_new_map[old_path_index_map[path]] = new_idx

        new_paths = sorted(new_path_index_map, key=lambda p: new_path_index_map[p])

        new_relations = None
        new_sequences = experiment.instruction_sequences
        new_multipliers = experiment.randomization_multipliers

        if experiment.relations is not None:
            new_relations = set()
            for path_idx, inst_idx in experiment.relations:
                if path_idx in old_to_new_map:
                    new_relations.add((old_to_new_map[path_idx], inst_idx))

            if new_sequences is not None:
                idxs_to_keep = {idx for _, idx in new_relations}
                index_map = {}
                kept_sequences = []
                kept_multipliers = []
                for idx, seq in enumerate(new_sequences):
                    if idx in idxs_to_keep:
                        index_map[idx] = len(kept_sequences)
                        kept_sequences.append(seq)
                        if new_multipliers is not None:
                            kept_multipliers.append(new_multipliers[idx])
                new_sequences = kept_sequences
                new_multipliers = kept_multipliers if new_multipliers is not None else None
                new_relations = {
                    (path_idx, index_map[inst_idx]) for path_idx, inst_idx in new_relations
                }

        return experiment.replace(
            validate=False,
            paths=new_paths,
            instruction_sequences=new_sequences,
            randomization_multipliers=new_multipliers,
            relations=new_relations,
        )
