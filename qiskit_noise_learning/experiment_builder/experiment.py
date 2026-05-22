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

"""Experiment dataclass."""

from dataclasses import dataclass

from qiskit_noise_learning.models import FidelityModel
from qiskit_noise_learning.sequences import InstructionSequence, Path


@dataclass
class Experiment:
    """An experiment specification ready for circuit generation.

    An :class:`Experiment` is the result of committing to a specific set of instruction sequences,
    analysis paths, fidelity model, and shot count. It is consumed by
    :meth:`.CircuitGenerator.generate` to produce an executable task.

    Args:
        sequences: The complete instruction sequences to run.
        paths: The analysis paths.
        fidelity_model: The fidelity model describing the noise parameterization.
        shots: The number of shots per randomization.
    """

    sequences: list[InstructionSequence]
    paths: list[Path]
    fidelity_model: FidelityModel
    shots: int
