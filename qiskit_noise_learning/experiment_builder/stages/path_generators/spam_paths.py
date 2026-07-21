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

"""SPAMPaths stage."""

from collections.abc import Iterator

from qiskit.quantum_info import QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate
from qiskit_noise_learning.sequences import FidelityIndex, Path

from ...experiment import Experiment
from ..utils import default_meas_gate, default_prep_gate
from .add_paths import AddPaths


class SPAMPaths(AddPaths):
    """Generate depth-0 paths.

    Args:
        prep_gate: The preparation gate. If ``None``, defaults to the gate named ``"P"``.
        meas_gate: The measurement gate. If ``None``, defaults to the gate named ``"M"``.
        indices_list: An iterable of qubit index lists. If ``None``, generates single-qubit
            paths for every qubit.
    """

    required_fields = ("fidelity_model",)

    def __init__(
        self,
        *,
        prep_gate: ModelGate | None = None,
        meas_gate: ModelGate | None = None,
        indices_list: list[list[int]] | None = None,
    ):
        self._prep_gate = prep_gate
        self._meas_gate = meas_gate
        self._indices_list = indices_list

    def _generate_paths(self, experiment: Experiment) -> Iterator[Path]:
        gate_set = experiment.gate_set
        num_qubits = gate_set.num_qubits
        prep_gate = self._prep_gate or default_prep_gate(gate_set)
        meas_gate = self._meas_gate or default_meas_gate(gate_set)

        indices_list = self._indices_list
        if indices_list is None:
            indices_list = [[i] for i in prep_gate.prep_idxs]

        for indices in indices_list:
            yield Path(
                start_fragment=[
                    FidelityIndex.from_gate(
                        gate=prep_gate,
                        pauli=QubitSparsePauli.identity(num_qubits),
                        in_bit_indices=frozenset(),
                        out_bit_indices=frozenset(indices),
                    )
                ],
                repeatable_fragment=[],
                end_fragment=[
                    FidelityIndex.from_gate(
                        gate=meas_gate,
                        pauli=QubitSparsePauli.identity(num_qubits),
                        in_bit_indices=frozenset(indices),
                        out_bit_indices=frozenset(),
                    )
                ],
                fragment_depth=0,
            )
