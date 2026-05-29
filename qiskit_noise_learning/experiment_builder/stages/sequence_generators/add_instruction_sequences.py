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

"""AddInstructionSequences stage."""

from qiskit_noise_learning.sequences import InstructionSequence

from ...experiment import Experiment
from ...stage import ExperimentBuilderStage


class AddInstructionSequences(ExperimentBuilderStage):
    """Add instruction sequences to an experiment.

    This is both a concrete stage (pass sequences directly) and the base class for
    sequence-generator stages that compute sequences from the experiment at runtime.

    Subclasses should override :meth:`_generate_sequences` to return sequences and
    multipliers computed from the experiment's data.

    Args:
        instruction_sequences: The instruction sequences to add.
        randomization_multipliers: Per-sequence randomization multipliers. If ``None``,
            defaults to all ones.
    """

    def __init__(
        self,
        instruction_sequences: list[InstructionSequence],
        randomization_multipliers: list[int] | None = None,
    ):
        self._instruction_sequences = instruction_sequences
        self._randomization_multipliers = randomization_multipliers

    def _run(self, experiment: Experiment) -> Experiment:
        existing_seqs = list(experiment.instruction_sequences or [])
        existing_mults = list(experiment.randomization_multipliers or [])

        new_seqs, new_mults = self._generate_sequences(experiment)
        existing_seqs.extend(new_seqs)
        existing_mults.extend(new_mults)

        return experiment.replace(
            validate=False,
            instruction_sequences=existing_seqs,
            randomization_multipliers=existing_mults,
        )

    def _generate_sequences(
        self, experiment: Experiment
    ) -> tuple[list[InstructionSequence], list[int]]:
        """Return sequences and multipliers to add.

        The default implementation returns the sequences and multipliers passed at
        construction. Subclasses override this to generate sequences from the experiment.
        """
        seqs = list(self._instruction_sequences)
        mults = self._randomization_multipliers or [1] * len(seqs)
        return seqs, mults
