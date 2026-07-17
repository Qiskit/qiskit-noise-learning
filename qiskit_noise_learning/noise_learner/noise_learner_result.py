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

"""Noise learner result."""

from qiskit.quantum_info import PauliLindbladMap

from qiskit_noise_learning.analysis import Fit
from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.models import split_pauli_lindblad_model


class NoiseLearnerResult:
    """Result of noise learning.

    Wraps a :class:`~.Fit` container and provides conversion to
    :class:`~qiskit.quantum_info.PauliLindbladMap` per gate.

    Args:
        fit: The completed fit containing model data.
    """

    def __init__(self, fit: Fit):
        self._fit = fit

    @property
    def fit(self) -> Fit:
        """The underlying fit container."""
        return self._fit

    def to_dict(self) -> dict[str, PauliLindbladMap]:
        """Convert the result to a dictionary mapping gate names to Pauli Lindblad maps."""
        model_data = self._fit.model_data
        if not isinstance(model_data, ModelData):
            raise RuntimeError("Fit does not contain ModelData; analysis may not have completed.")
        pauli_lindblad_model = split_pauli_lindblad_model(self._fit.model).model
        return pauli_lindblad_model.to_pauli_lindblad_maps(model_data)
