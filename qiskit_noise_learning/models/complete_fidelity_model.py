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

from qiskit_noise_learning.gate_sets import GateSet
from qiskit_noise_learning.math import IndexedVector
from qiskit_noise_learning.sequences import FidelityIndex

from .fidelity_index_space import FidelityIndexSpace
from .fidelity_model import FidelityModel


class CompleteFidelityModel(FidelityModel[FidelityIndex]):
    r"""A fidelity model where the parameters are the log fidelities themselves.

    In this case, the parameterization matrix :math:`A` is simply the identity.

    Args:
        gate_set: The gate set for which the fidelities are being modelled.
    """

    def __init__(self, gate_set: GateSet):
        model_gate_set = gate_set.model_gate_set
        super().__init__(gate_set=gate_set, input_space=FidelityIndexSpace(model_gate_set))

    def _core_row(self, fidelity_index):
        return IndexedVector[FidelityIndex]({fidelity_index: 1.0})
