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

"""What the vanilla protocol learned."""

from dataclasses import dataclass

from qiskit.quantum_info import PauliLindbladMap

from ...analysis import Fit


@dataclass(frozen=True)
class VanillaFit:
    """The fit results of the vanilla program.

    Contains the general :class:`~.Fit` object, along with two fields ``noise_maps`` and
    ``gate_names``, respectively containing the learned
    :class:`~qiskit.quantum_info.PauliLindbladMap` and string name assigned to each of the
    ``instructions`` supplied to :func:`~.prepare_vanilla_program`. I.e. the noise map and name
    for ``instructions[idx]`` is ``noise_maps[idx]`` and ``gate_names[idx]``.

    Args:
        noise_maps: The noise map learned for each submitted instruction.
        gate_names: The name assigned to each instruction used in :attr:`fit`.
        fit: The :class:`~.Fit` object.
    """

    noise_maps: list[PauliLindbladMap]
    gate_names: list[str]
    fit: Fit
