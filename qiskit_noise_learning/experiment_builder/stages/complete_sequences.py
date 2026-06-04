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

"""CompleteSequences stage."""

from __future__ import annotations

from ..experiment import Experiment
from ..stage import ExperimentBuilderStage


class CompleteSequences(ExperimentBuilderStage):
    """Complete all instruction sequences.

    Calls :meth:`~.InstructionSequence.complete` on each instruction sequence to finalize
    any :class:`~.PartialPauliPermutation` instances.
    """

    required_fields = ("instruction_sequences",)

    def _run(self, experiment: Experiment) -> Experiment:
        sequences = [seq.complete() for seq in experiment.instruction_sequences]
        return experiment.replace(validate=False, instruction_sequences=sequences)
