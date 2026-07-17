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

"""Modelling module."""

from collections.abc import Hashable

from qiskit_noise_learning.math import LinearMap
from qiskit_noise_learning.sequences import FidelityIndex

from .identity_fidelity_model import IdentityFidelityModel
from .log_fidelity_space import LogFidelitySpace
from .pauli_lindblad_model import GeneratorIndex, PauliLindbladModel, RateSpace
from .utils import (
    contains_pauli_lindblad_model,
    is_fidelity_model,
    split_pauli_lindblad_model,
)

# Type hint for a fidelity model. The authoritative criterion is structural -- a LinearMap whose
# output space is a LogFidelitySpace (see is_fidelity_model); this alias is the closest index-level
# approximation, since LinearMap is generic over index types, not space types.
FidelityModel = LinearMap[Hashable, FidelityIndex]
