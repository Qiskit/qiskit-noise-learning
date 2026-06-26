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

"""Legacy modelling module (pending removal).

These are the original :class:`FidelityModel`-based models, quarantined here while the
:class:`~.LinearMap`-based replacements are built up. Consumers are repointed here temporarily and
migrated off incrementally; this subpackage is deleted once nothing references it.
"""

from .complete_fidelity_model import CompleteFidelityModel
from .fidelity_mixers import FidelityMixer, IdentityMixer
from .fidelity_model import FidelityModel
from .mixed_fidelity_model import MixedFidelityModel
from .pauli_lindblad_model import GeneratorIndex, PauliLindbladModel
