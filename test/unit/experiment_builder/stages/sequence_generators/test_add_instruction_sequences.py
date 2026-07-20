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

from qiskit_noise_learning.experiment_builder.experiment import Experiment
from qiskit_noise_learning.experiment_builder.stages import AddInstructionSequences


class TestAddInstructionSequences:
    def test_adds_sequences(self, make_cz_path):
        seq = make_cz_path("IX").to_instruction_sequence()
        exp = Experiment()
        result = AddInstructionSequences([seq]).run(exp)

        assert result.instruction_sequences == [seq]
        assert result.randomization_multipliers == [1]

    def test_appends_to_existing(self, make_cz_path):
        seq_ix = make_cz_path("IX").to_instruction_sequence()
        seq_xi = make_cz_path("XI").to_instruction_sequence()
        exp = Experiment(instruction_sequences=[seq_ix], randomization_multipliers=[1])
        result = AddInstructionSequences([seq_xi], randomization_multipliers=[3]).run(exp)

        assert result.instruction_sequences == [seq_ix, seq_xi]
        assert result.randomization_multipliers == [1, 3]
