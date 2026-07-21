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
from qiskit.quantum_info import QubitSparsePauliList

from qiskit_noise_learning.experiment_builder.experiment import Experiment
from qiskit_noise_learning.experiment_builder.stages import Depth1Paths


class TestDepth1Paths:
    def test_generates_depth1_paths(self, gate_set_cz):
        exp = Experiment(fidelity_model=gate_set_cz)
        stage = Depth1Paths(
            prep_gate=gate_set_cz["P"],
            meas_gate=gate_set_cz["M"],
            gates=[gate_set_cz["CZ"]],
            input_paulis={"CZ": QubitSparsePauliList(["IZ", "IX", "XX"])},
        )
        result = stage.run(exp)

        assert len(result.paths) == 3
        assert all(p.fragment_depth == 0 for p in result.paths)

    def test_requires_fidelity_model_when_gates_not_provided(self):
        stage = Depth1Paths()
        with pytest.raises(ValueError, match="requires 'fidelity_model'"):
            stage.run(Experiment())
