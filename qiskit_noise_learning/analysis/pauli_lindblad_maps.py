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

"""Conversion of fitted model data to Pauli-Lindblad noise maps."""

from qiskit.quantum_info import PauliLindbladMap

from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.math import ComposedLinearMap
from qiskit_noise_learning.models import FidelityModel, find_pauli_lindblad_model

from .model_propagation import propagate_model_data


def to_pauli_lindblad_maps(
    model: FidelityModel,
    model_data: ModelData,
    include_spam: bool = False,
) -> dict[str, PauliLindbladMap]:
    """Return the Pauli-Lindblad noise maps for a fitted model.

    The model must be, or contain (in the case of a :class:`~.ComposedFidelityModel`), a
    :class:`~.PauliLindbladModel`. When the Pauli-Lindblad model is composed behind other maps, the
    fitted ``model_data`` is first propagated forward through those maps into the Pauli-Lindblad
    model's rate space, and the noise maps are then built from those rates.

    Args:
        model: The fitted model, either a :class:`~.PauliLindbladModel` or a
            :class:`~.ComposedFidelityModel` containing one.
        model_data: The fitted parameters, in the input space of ``model``.
        include_spam: Whether to include SPAM gates in the output.

    Returns:
        A dictionary from gate names to corresponding noise maps.

    Raises:
        ValueError: If ``model`` does not contain a :class:`~.PauliLindbladModel`.
    """
    pauli_lindblad_model = find_pauli_lindblad_model(model)
    if pauli_lindblad_model is None:
        raise ValueError("model does not contain a PauliLindbladModel.")

    if model is pauli_lindblad_model:
        rate_data = model_data
    else:
        maps = model.maps
        index = maps.index(pauli_lindblad_model)
        if index == 0:
            # The Pauli-Lindblad model is applied first, so model_data is already in its rate space.
            rate_data = model_data
        else:
            pre_chain = maps[0] if index == 1 else ComposedLinearMap(maps[:index])
            rate_data = propagate_model_data(
                pre_chain, model_data, pauli_lindblad_model.input_space
            )

    return pauli_lindblad_model.to_pauli_lindblad_maps(rate_data, include_spam=include_spam)
