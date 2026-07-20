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
from qiskit_noise_learning.experiment_builder.stages import GenerateInstructionSequences


class TestGenerateInstructionSequences:
    def test_requires_paths(self):
        stage = GenerateInstructionSequences()
        with pytest.raises(ValueError, match="requires 'paths'"):
            stage.run(Experiment())

    def test_generates_sequences_from_paths(self, gate_set_cz, make_cz_path):
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[make_cz_path("IX"), make_cz_path("XI")],
        )
        result = GenerateInstructionSequences().run(exp)

        assert len(result.instruction_sequences) == 2
        assert result.relations == {(0, 0), (1, 1)}
        assert result.randomization_multipliers == [1, 1]

    def test_skips_already_related_paths(self, gate_set_cz, make_cz_path):
        unbound_path_ix = make_cz_path("IX")
        unbound_path_xi = make_cz_path("XI")
        seq_ix = unbound_path_ix.to_instruction_sequence()
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix, unbound_path_xi],
            instruction_sequences=[seq_ix],
            relations={(0, 0)},
            randomization_multipliers=[1],
        )
        result = GenerateInstructionSequences().run(exp)

        assert len(result.instruction_sequences) == 2
        assert result.relations == {(0, 0), (1, 1)}

    def test_custom_multipliers(self, gate_set_cz, make_cz_path):
        unbound_path_ix = make_cz_path("IX")
        bound_path = unbound_path_ix.bind_at(2)
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix, bound_path],
        )
        result = GenerateInstructionSequences(bound_multiplier=5, unbound_multiplier=3).run(exp)

        assert result.randomization_multipliers == [3, 5]
