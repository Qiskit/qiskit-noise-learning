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
from qiskit_noise_learning.experiment_builder.stages import SetShots


class TestSetShots:
    def test_sets_shots(self):
        exp = Experiment()
        result = SetShots(100).run(exp)
        assert result.shots == 100

    def test_overrides_existing_shots(self):
        exp = Experiment(shots=50)
        result = SetShots(200).run(exp)
        assert result.shots == 200
