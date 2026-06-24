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

from qiskit_noise_learning.experiment_builder.experiment import Experiment
from qiskit_noise_learning.experiment_builder.stages import BindSequenceDepths


class TestBindSequenceDepths:
    def test_requires_instruction_sequences(self):
        stage = BindSequenceDepths(depths=[2, 4])
        with pytest.raises(ValueError, match="requires 'instruction_sequences'"):
            stage.run(Experiment())

    def test_expands_unbound_sequences(self, make_cz_path):
        unbound_path_ix = make_cz_path("IX")
        unbound_path_xi = make_cz_path("XI")
        seq_ix = unbound_path_ix.to_instruction_sequence()
        seq_xi = unbound_path_xi.to_instruction_sequence()
        assert seq_ix.is_unbound
        assert seq_xi.is_unbound

        exp = Experiment(
            paths=[unbound_path_ix, unbound_path_xi],
            instruction_sequences=[seq_ix, seq_xi],
            relations={(0, 0), (1, 1)},
            randomization_multipliers=[1, 2],
        )
        result = BindSequenceDepths(depths=[2, 4]).run(exp)

        assert len(result.instruction_sequences) == 4
        assert all(not s.is_unbound for s in result.instruction_sequences)
        assert result.instruction_sequences[0].depth == 2
        assert result.instruction_sequences[1].depth == 4
        assert result.instruction_sequences[2].depth == 2
        assert result.instruction_sequences[3].depth == 4

    def test_keeps_bound_sequences(self, make_cz_path):
        unbound_path_ix = make_cz_path("IX")
        seq = unbound_path_ix.to_instruction_sequence().bind_at(3)
        exp = Experiment(
            paths=[unbound_path_ix],
            instruction_sequences=[seq],
            relations={(0, 0)},
            randomization_multipliers=[1],
        )
        result = BindSequenceDepths(depths=[2, 4]).run(exp)

        assert len(result.instruction_sequences) == 1
        assert result.instruction_sequences[0].depth == 3

    def test_relations_fanned_out(self, make_cz_path):
        unbound_path_ix = make_cz_path("IX")
        unbound_path_xi = make_cz_path("XI")
        seq_ix = unbound_path_ix.to_instruction_sequence()
        exp = Experiment(
            paths=[unbound_path_ix, unbound_path_xi],
            instruction_sequences=[seq_ix],
            relations={(0, 0), (1, 0)},
            randomization_multipliers=[1],
        )
        result = BindSequenceDepths(depths=[2, 4, 6]).run(exp)

        assert len(result.instruction_sequences) == 3
        assert result.relations == {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}

    def test_multipliers_propagated(self, make_cz_path):
        unbound_path_ix = make_cz_path("IX")
        unbound_path_xi = make_cz_path("XI")
        seq_ix = unbound_path_ix.to_instruction_sequence()
        seq_xi = unbound_path_xi.to_instruction_sequence()
        exp = Experiment(
            paths=[unbound_path_ix, unbound_path_xi],
            instruction_sequences=[seq_ix, seq_xi],
            relations={(0, 0), (1, 1)},
            randomization_multipliers=[3, 5],
        )
        result = BindSequenceDepths(depths=[2, 4]).run(exp)

        assert result.randomization_multipliers == [3, 3, 5, 5]
