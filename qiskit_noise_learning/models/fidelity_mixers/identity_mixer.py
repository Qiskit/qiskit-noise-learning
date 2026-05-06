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

"""IdentityMixer"""

from qiskit_noise_learning.math import IndexedVector
from qiskit_noise_learning.sequences import FidelityIndex

from .fidelity_mixer import FidelityMixer


class IdentityMixer(FidelityMixer):
    """The fidelity mixer corresponding to the identity."""

    def fidelity_mixture(self, fidelity_index: FidelityIndex) -> IndexedVector[FidelityIndex]:
        return IndexedVector[FidelityIndex]({fidelity_index: 1.0})
