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

"""SetShots stage."""

from __future__ import annotations

from ..experiment import Experiment
from ..stage import ExperimentBuilderStage


class SetShots(ExperimentBuilderStage):
    """Set the global number of shots.

    Args:
        shots: The number of shots.
    """

    def __init__(self, shots: int):
        self._shots = shots

    def _run(self, experiment: Experiment) -> Experiment:
        return experiment.replace(shots=self._shots)
