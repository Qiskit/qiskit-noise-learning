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
from qiskit_noise_learning.experiment_builder.stages import RankReducePaths
from qiskit_noise_learning.sequences import FidelityIndex, Path


class TestRankReducePaths:
    def test_requires_paths(self):
        stage = RankReducePaths()
        with pytest.raises(ValueError, match="requires 'paths'"):
            stage.run(Experiment())

    def test_removes_linearly_dependent_paths(self, gate_set_cz):
        path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IX"), QubitSparsePauli("ZX")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZX"), QubitSparsePauli("IX")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
        )
        dependent_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("ZZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZX"), QubitSparsePauli("IX")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IX"), QubitSparsePauli("ZX")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("ZZ"), QubitSparsePauli("II")
                )
            ],
        )

        seq_0 = path.to_instruction_sequence()
        seq_1 = dependent_path.to_instruction_sequence()
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[path, dependent_path],
            instruction_sequences=[seq_0, seq_1],
            relations={(0, 0), (1, 1)},
            randomization_multipliers=[1, 1],
        )
        result = RankReducePaths().run(exp)

        assert len(result.paths) == 1
        assert result.paths[0] == path
        assert len(result.instruction_sequences) == 1
        assert result.relations == {(0, 0)}

    def test_independent_paths_kept(self, gate_set_cz, make_cz_path):
        unbound_path_ix = make_cz_path("IX")
        unbound_path_xi = make_cz_path("XI")
        seq_ix = unbound_path_ix.to_instruction_sequence()
        seq_xi = unbound_path_xi.to_instruction_sequence()
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix, unbound_path_xi],
            instruction_sequences=[seq_ix, seq_xi],
            relations={(0, 0), (1, 1)},
            randomization_multipliers=[1, 1],
        )
        result = RankReducePaths().run(exp)

        assert len(result.paths) == 2
        assert len(result.instruction_sequences) == 2
