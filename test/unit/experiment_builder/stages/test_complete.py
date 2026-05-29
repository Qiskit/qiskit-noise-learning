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
from qiskit_noise_learning.experiment_builder.stages import Complete


class TestComplete:
    def test_requires_instruction_sequences(self):
        stage = Complete()
        with pytest.raises(ValueError, match="requires 'instruction_sequences'"):
            stage.run(Experiment())

    def test_completes_sequences(self, gate_set_1q, unbound_path_ix):
        seq = unbound_path_ix.to_instruction_sequence().bind_at(3)
        assert not seq.is_complete

        exp = Experiment(instruction_sequences=[seq], randomization_multipliers=[1])
        result = Complete().run(exp)

        assert all(s.is_complete for s in result.instruction_sequences)

    def test_already_complete_sequences_unchanged(self, unbound_path_ix):
        seq = unbound_path_ix.to_instruction_sequence().bind_at(3).complete()
        exp = Experiment(instruction_sequences=[seq], randomization_multipliers=[1])
        result = Complete().run(exp)

        assert result.instruction_sequences == [seq]
