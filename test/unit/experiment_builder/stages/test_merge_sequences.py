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

import pytest
from qiskit.quantum_info import QubitSparsePauli

from qiskit_noise_learning.experiment_builder.experiment import Experiment
from qiskit_noise_learning.experiment_builder.stages import MergeInstructionSequences
from qiskit_noise_learning.sequences import FidelityIndex, Path


class TestMergeInstructionSequences:
    def test_requires_fields(self):
        stage = MergeInstructionSequences()
        with pytest.raises(ValueError, match="requires 'instruction_sequences'"):
            stage.run(Experiment())

    def test_two_mergeable_sequences(self, gate_set_cz, make_cz_path):
        """Two compatible sequences are merged into one."""
        unbound_path_ix = make_cz_path("IX")
        unbound_path_xi = make_cz_path("XI")
        seqs = [
            unbound_path_ix.to_instruction_sequence(),
            unbound_path_xi.to_instruction_sequence(),
        ]
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix, unbound_path_xi],
            instruction_sequences=seqs,
            relations={(0, 0), (1, 1)},
            randomization_multipliers=[1, 1],
        )
        result = MergeInstructionSequences().run(exp)

        assert len(result.instruction_sequences) == 1
        assert result.relations == {(0, 0), (1, 0)}
        assert unbound_path_ix.is_traversed_by(result.instruction_sequences[0])
        assert unbound_path_xi.is_traversed_by(result.instruction_sequences[0])

    def test_three_way_merge(self, gate_set_cz, make_cz_path):
        """Three mutually compatible sequences are merged into one."""
        paths = [make_cz_path("IX"), make_cz_path("XI"), make_cz_path("XX")]
        seqs = [p.to_instruction_sequence() for p in paths]
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=paths,
            instruction_sequences=seqs,
            relations={(0, 0), (1, 1), (2, 2)},
            randomization_multipliers=[1, 1, 1],
        )
        result = MergeInstructionSequences().run(exp)

        assert len(result.instruction_sequences) == 1
        assert result.relations == {(0, 0), (1, 0), (2, 0)}
        for path in paths:
            assert path.is_traversed_by(result.instruction_sequences[0])

    def test_incompatible_sequences_not_merged(self, gate_set_1q):
        """Sequences with different gate structures cannot be merged."""
        path1 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
                ),
                FidelityIndex.from_transition(
                    gate_set_1q["L0"], QubitSparsePauli("Y"), QubitSparsePauli("Z")
                ),
                FidelityIndex.from_transition(
                    gate_set_1q["L0"], QubitSparsePauli("Z"), QubitSparsePauli("X")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
                )
            ],
        )
        path2 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("X"), QubitSparsePauli("X")
                ),
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("Y"), QubitSparsePauli("Y")
                ),
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("Z"), QubitSparsePauli("Z")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
                )
            ],
        )

        seqs = [path1.to_instruction_sequence(), path2.to_instruction_sequence()]
        exp = Experiment(
            fidelity_model=gate_set_1q,
            paths=[path1, path2],
            instruction_sequences=seqs,
            relations={(0, 0), (1, 1)},
            randomization_multipliers=[1, 1],
        )
        result = MergeInstructionSequences().run(exp)

        assert len(result.instruction_sequences) == 2
        assert result.relations == {(0, 0), (1, 1)}

    def test_multipliers_take_max(self, gate_set_cz, make_cz_path):
        """When merging, the resulting multiplier is the max of the merged group."""
        unbound_path_ix = make_cz_path("IX")
        unbound_path_xi = make_cz_path("XI")
        seqs = [
            unbound_path_ix.to_instruction_sequence(),
            unbound_path_xi.to_instruction_sequence(),
        ]
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix, unbound_path_xi],
            instruction_sequences=seqs,
            relations={(0, 0), (1, 1)},
            randomization_multipliers=[3, 7],
        )
        result = MergeInstructionSequences().run(exp)

        assert result.randomization_multipliers == [7]

    def test_single_sequence_unchanged(self, gate_set_cz, make_cz_path):
        unbound_path_ix = make_cz_path("IX")
        seq = unbound_path_ix.to_instruction_sequence()
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix],
            instruction_sequences=[seq],
            relations={(0, 0)},
            randomization_multipliers=[1],
        )
        result = MergeInstructionSequences().run(exp)

        assert len(result.instruction_sequences) == 1
        assert result.relations == {(0, 0)}
