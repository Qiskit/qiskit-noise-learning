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

"""FidelityMixer"""

from abc import ABC, abstractmethod

from qiskit_noise_learning.math import IndexedVector
from qiskit_noise_learning.sequences import FidelityIndex


class FidelityMixer(ABC):
    """An interface for a linear mapping from fidelities to fidelities.

    Each row is assumed to be a probability vector.
    """

    @abstractmethod
    def fidelity_mixture(self, fidelity_index: FidelityIndex) -> IndexedVector[FidelityIndex]:
        """Return the row in the linear mapping corresponding to the given fidelity index.

        Args:
            fidelity_index: The row index.

        Returns:
            The probability vector, labelled by standard fidelity indices, describing the mixture.
        """
