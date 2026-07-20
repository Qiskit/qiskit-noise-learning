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
from qiskit_noise_learning.experiment_builder.stages import EvenDepthPaths


class TestEvenDepthPaths:
    def test_generates_even_depth_paths(self, gate_set_cz):
        exp = Experiment(fidelity_model=gate_set_cz)
        stage = EvenDepthPaths(
            prep_gate=gate_set_cz["P"],
            meas_gate=gate_set_cz["M"],
            gates=[gate_set_cz["CZ"]],
            input_paulis={"CZ": QubitSparsePauliList(["IZ", "IX", "IY", "XX"])},
        )
        result = stage.run(exp)

        assert len(result.paths) == 6
        assert all(p.is_unbound for p in result.paths)

    def test_requires_fidelity_model_when_input_paulis_not_provided(self, gate_set_cz):
        stage = EvenDepthPaths(
            prep_gate=gate_set_cz["P"],
            meas_gate=gate_set_cz["M"],
            gates=[gate_set_cz["CZ"]],
        )
        with pytest.raises(ValueError, match="requires 'fidelity_model'"):
            stage.run(Experiment())
