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

"""SetRandomizations stage."""

from __future__ import annotations

from ..experiment import Experiment
from ..stage import ExperimentBuilderStage


class SetRandomizations(ExperimentBuilderStage):
    """Set the global number of randomizations.

    Args:
        randomizations: The number of randomizations.
    """

    def __init__(self, randomizations: int):
        self._randomizations = randomizations

    def _run(self, experiment: Experiment) -> Experiment:
        return experiment.replace(randomizations=self._randomizations)
