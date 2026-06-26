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

"""CompleteFidelityModel"""

from qiskit_noise_learning.math import IndexedVector
from qiskit_noise_learning.sequences import FidelityIndex

from .mixed_fidelity_model import MixedFidelityModel


class CompleteFidelityModel(MixedFidelityModel[FidelityIndex]):
    r"""A fidelity model where the parameters are the log fidelities themselves.

    In this case, the parameterization matrix :math:`A` is simply the identity.

    Args:
        gate_set: The gate set for which the fidelities are being modelled.
        fidelity_mixer: Averaging procedure for the fidelities.
    """

    def row_from_unmixed_fidelity(self, fidelity_index):
        return IndexedVector[FidelityIndex]({fidelity_index: 1.0})
