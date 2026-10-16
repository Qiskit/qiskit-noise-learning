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
class VanillaResults:
    """The noise the vanilla protocol learned, against the instructions whose noise it is.

    The three fields are parallel: entry ``i`` of :attr:`noise_maps` and of :attr:`gate_names`
    belong to entry ``i`` of the ``instructions`` given to :func:`~.prepare_vanilla_program`.

    A noise map acts on the qubits of its instruction, its qubit ``j`` being the ``j``-th of those
    in ascending physical-qubit order. This is the indexing a noise map is read back with, so a
    learned map can be applied to the instruction it came from, and the same qubits as
    ``fit.model.gate_set[name].qubit_idxs`` for the corresponding gate name.

    The map of a submitted measurement box is a readout channel, built from :math:`X` generators on
    the measured qubits. All of the state preparation and measurement noise is attributed to it,
    because the protocol models preparation as noiseless.

    When no measurement box is submitted, the protocol measures with a layer of its own, and that
    layer is absent from these lists: it is not one of the submitted instructions. Its learned
    noise is on the :attr:`fit` like any other gate's.

    Args:
        noise_maps: The noise learned for each submitted instruction.
        gate_names: The name of the gate each submitted instruction became, which is how the
            :attr:`fit` keys it.
        fit: The fit the noise was read out of, holding the data at every analysis level.
    """

    noise_maps: list[PauliLindbladMap]
    gate_names: list[str]
    fit: Fit
