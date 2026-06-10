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

from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate
from qiskit_noise_learning.math import IndexedVector
from qiskit_noise_learning.models.fidelity_mixers import IdentityMixer
from qiskit_noise_learning.sequences import FidelityIndex


def test_fidelity_mixture():
    """Test fidelity_mixture returns a delta distribution."""
    gate = ModelGate("L0", [((0, 1), Clifford(CZGate()))])
    fidelity_index = FidelityIndex.from_gate(
        gate=gate,
        pauli=QubitSparsePauli("ZX"),
        in_bit_indices=frozenset(),
        out_bit_indices=frozenset(),
    )

    mixer = IdentityMixer()
    result = mixer.fidelity_mixture(fidelity_index)
    assert result == IndexedVector({fidelity_index: 1.0})
