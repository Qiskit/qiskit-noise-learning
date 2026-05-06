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

"""MixedFidelityModel"""

from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import TypeVar

from qiskit_noise_learning.gate_sets import GateSet
from qiskit_noise_learning.math import IndexedVector
from qiskit_noise_learning.sequences import FidelityIndex

from .fidelity_mixers import FidelityMixer, IdentityMixer
from .fidelity_model import FidelityModel

ParameterIndex = TypeVar("ParameterIndex", bound=Hashable)


class MixedFidelityModel(FidelityModel[ParameterIndex], ABC):
    r"""A linear parameterization of the log fidelities of a gate set.

    This subclass defines an interface in which the parameterization matrix :math:`A` of
    :class:`FidelityModel` is augmented by a :class:`FidelityMixer`. Effectively, this modifies the
    parameterization by composing it with the linear map implied by :class:`FidelityMixer`, which
    'mixes' the fidelities by averaging them together. This is relevant in the context of additional
    twirling of the noise model beyond the minimal twirling required to generate a Pauli channel
    or Pauli instrument noise model.

    Args:
        gate_set: The gate set for which the fidelities are being modelled.
        fidelity_mixer: Averaging procedure for the fidelities.
    """

    def __init__(self, gate_set: GateSet, fidelity_mixer: FidelityMixer | None = None):
        self._fidelity_mixer = fidelity_mixer or IdentityMixer()
        super().__init__(gate_set=gate_set)

    @property
    def fidelity_mixer(self) -> FidelityMixer:
        return self._fidelity_mixer

    @abstractmethod
    def row_from_unmixed_fidelity(
        self, fidelity_index: FidelityIndex
    ) -> IndexedVector[ParameterIndex]:
        """The row in the base parameterization matrix, unmodified by the fidelity mixer."""

    def row_from_fidelity(self, fidelity_index: FidelityIndex) -> IndexedVector[ParameterIndex]:
        """Get a row of the fidelity parameterization matrix."""

        fidelity_mix = self.fidelity_mixer.fidelity_mixture(fidelity_index)
        vector = IndexedVector[ParameterIndex]()

        for fid_idx, val in fidelity_mix.items():
            vec = self.row_from_unmixed_fidelity(fid_idx)
            vector += float(val) * vec

        return vector
