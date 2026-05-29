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

from itertools import product

import pytest

from qiskit_noise_learning.experiment_builder.experiment import Experiment
from qiskit_noise_learning.experiment_builder.stages import IdentifyRelations
from qiskit_noise_learning.sequences import (
    ApplyGate,
    InstructionSequence,
    PartialPauliPermutation,
)


class TestIdentifyRelations:
    def test_requires_paths_and_sequences(self):
        stage = IdentifyRelations()
        with pytest.raises(ValueError, match="requires 'paths'"):
            stage.run(Experiment())

        with pytest.raises(ValueError, match="requires 'instruction_sequences'"):
            stage.run(Experiment(paths=[]))

    def test_unbound_paths_full_relations(
        self, gate_set_cz, unbound_path_ix, unbound_path_xi, unbound_path_xx
    ):
        """All instruction sequences are mutually mergeable, so extension gives full relations."""
        paths = [unbound_path_ix, unbound_path_xi, unbound_path_xx]
        seqs = [p.to_instruction_sequence() for p in paths]
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=paths,
            instruction_sequences=seqs,
            relations={(0, 0), (1, 1), (2, 2)},
            randomization_multipliers=[1, 1, 1],
        )
        result = IdentifyRelations().run(exp)

        assert result.relations == {(a, b) for a, b in product(range(3), range(3))}

    def test_bound_paths_full_relations(
        self, gate_set_cz, unbound_path_ix, unbound_path_xi, unbound_path_xx
    ):
        """All instruction sequences are mutually mergeable, so extension gives full relations."""
        paths = [p.bind_at(2) for p in [unbound_path_ix, unbound_path_xi, unbound_path_xx]]
        seqs = [p.to_instruction_sequence() for p in paths]

        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=paths,
            instruction_sequences=seqs,
            relations={(0, 0), (1, 1), (2, 2)},
            randomization_multipliers=[1, 1, 1],
        )
        result = IdentifyRelations().run(exp)

        assert result.relations == {(a, b) for a, b in product(range(3), range(3))}

    def test_no_extension_limits_relations(
        self, gate_set_cz, unbound_path_ix, unbound_path_xi, unbound_path_xx
    ):
        paths = [unbound_path_ix, unbound_path_xi, unbound_path_xx]
        seqs = [p.to_instruction_sequence() for p in paths]
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=paths,
            instruction_sequences=seqs,
            relations={(0, 0), (1, 1), (2, 2)},
            randomization_multipliers=[1, 1, 1],
        )
        result = IdentifyRelations(attempt_instruction_extension=False).run(exp)

        assert result.relations == {(0, 0), (1, 1), (2, 2)}

    def test_manual_sequences_without_extension(
        self, gate_set_cz, unbound_path_ix, unbound_path_xi, unbound_path_xx
    ):
        paths = [unbound_path_ix, unbound_path_xi, unbound_path_xx]

        instruction_sequence_ix = InstructionSequence(
            start_fragment=[
                ApplyGate(gate_set_cz["P"]),
                PartialPauliPermutation.from_sets([{("Z", "X")}, set()]),
            ],
            repeatable_fragment=[ApplyGate(gate_set_cz["CZ"])] * 2,
            end_fragment=[
                PartialPauliPermutation.from_sets([{("X", "Z")}, set()]),
                ApplyGate(gate_set_cz["M"]),
            ],
        )
        instruction_sequence_xi = InstructionSequence(
            start_fragment=[
                ApplyGate(gate_set_cz["P"]),
                PartialPauliPermutation.from_sets([set(), {("Z", "X")}]),
            ],
            repeatable_fragment=[ApplyGate(gate_set_cz["CZ"])] * 2,
            end_fragment=[
                PartialPauliPermutation.from_sets([set(), {("X", "Z")}]),
                ApplyGate(gate_set_cz["M"]),
            ],
        )
        instruction_sequence_xx = InstructionSequence(
            start_fragment=[
                ApplyGate(gate_set_cz["P"]),
                PartialPauliPermutation.from_sets([{("Z", "X")}] * 2),
            ],
            repeatable_fragment=[ApplyGate(gate_set_cz["CZ"])] * 2,
            end_fragment=[
                PartialPauliPermutation.from_sets([{("X", "Z")}] * 2),
                ApplyGate(gate_set_cz["M"]),
            ],
        )

        seqs = [instruction_sequence_ix, instruction_sequence_xi, instruction_sequence_xx]
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=paths,
            instruction_sequences=seqs,
            relations={(0, 0), (1, 1), (2, 2)},
            randomization_multipliers=[1, 1, 1],
        )
        result = IdentifyRelations(attempt_instruction_extension=False).run(exp)

        assert result.relations == {(0, 0), (1, 1), (2, 2), (0, 2), (1, 2)}
