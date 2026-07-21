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
from qiskit_noise_learning.experiment_builder.stages import SPAMPaths
from qiskit_noise_learning.sequences import FidelityIndex, Path


class TestSPAMPaths:
    def test_generates_single_qubit_paths(self, gate_set_cz):
        exp = Experiment(fidelity_model=gate_set_cz)
        stage = SPAMPaths(
            prep_gate=gate_set_cz["P"],
            meas_gate=gate_set_cz["M"],
            indices_list=[[0], [1]],
        )
        result = stage.run(exp)

        expected = [
            Path(
                start_fragment=[
                    FidelityIndex.from_gate(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0]),
                    )
                ],
                repeatable_fragment=[],
                end_fragment=[
                    FidelityIndex.from_gate(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0]),
                    )
                ],
                fragment_depth=0,
            ),
            Path(
                start_fragment=[
                    FidelityIndex.from_gate(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([1]),
                    )
                ],
                repeatable_fragment=[],
                end_fragment=[
                    FidelityIndex.from_gate(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([1]),
                    )
                ],
                fragment_depth=0,
            ),
        ]
        assert result.paths == expected

    def test_requires_fidelity_model_when_gates_not_provided(self):
        stage = SPAMPaths()
        with pytest.raises(ValueError, match="requires 'fidelity_model'"):
            stage.run(Experiment())

    def test_appends_to_existing_paths(self, gate_set_cz, make_cz_path):
        unbound_path_ix = make_cz_path("IX")
        exp = Experiment(fidelity_model=gate_set_cz, paths=[unbound_path_ix])
        stage = SPAMPaths(
            prep_gate=gate_set_cz["P"],
            meas_gate=gate_set_cz["M"],
            indices_list=[[0]],
        )
        result = stage.run(exp)

        assert len(result.paths) == 2
        assert result.paths[0] == unbound_path_ix
