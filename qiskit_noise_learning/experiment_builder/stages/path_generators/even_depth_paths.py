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

"""EvenDepthPaths stage."""

from collections.abc import Iterator

import numpy as np
from qiskit.quantum_info import QubitSparsePauli, QubitSparsePauliList

from qiskit_noise_learning.gate_sets import ModelGate
from qiskit_noise_learning.sequences import FidelityIndex, Path

from ...experiment import Experiment
from ..utils import default_gates, default_meas_gate, default_prep_gate
from .add_paths import AddPaths


class EvenDepthPaths(AddPaths):
    """Generate unbound paths with repetitions of two applications of each target gate.

    For each target gate, generates all well-defined paths where the repeatable fragment
    consists of two applications of the gate with intermediate single-qubit Cliffords.

    Args:
        prep_gate: The preparation gate. If ``None``, defaults to the gate named ``"P"``.
        meas_gate: The measurement gate. If ``None``, defaults to the gate named ``"M"``.
        gates: Gates to generate paths for. If ``None``, defaults to all non-SPAM gates.
        input_paulis: Optional mapping from gate name to input Paulis.
        output_paulis: Optional mapping from gate name to output Paulis. Defaults to input.
    """

    @property
    def required_fields(self) -> tuple[str, ...]:
        if self._gates is None or self._prep_gate is None or self._meas_gate is None:
            return ("fidelity_model",)
        if self._input_paulis is None:
            return ("fidelity_model",)
        return ()

    def __init__(
        self,
        *,
        prep_gate: ModelGate | None = None,
        meas_gate: ModelGate | None = None,
        gates: list[ModelGate] | None = None,
        input_paulis: dict[str, QubitSparsePauliList] | None = None,
        output_paulis: dict[str, QubitSparsePauliList] | None = None,
    ):
        self._prep_gate = prep_gate
        self._meas_gate = meas_gate
        self._gates = gates
        self._input_paulis = input_paulis
        self._output_paulis = output_paulis

    def _generate_paths(self, experiment: Experiment) -> Iterator[Path]:
        gate_set = experiment.gate_set
        prep_gate = self._prep_gate or default_prep_gate(gate_set)
        meas_gate = self._meas_gate or default_meas_gate(gate_set)
        gates = self._gates or default_gates(gate_set)

        for gate in gates:
            if self._input_paulis is not None:
                in_paulis = self._input_paulis[gate.name]
            else:
                in_paulis = experiment.fidelity_model.generators[gate.name]

            out_paulis = None
            if self._output_paulis is not None:
                out_paulis = self._output_paulis.get(gate.name)

            yield from _even_depth_paths(prep_gate, meas_gate, gate, in_paulis, out_paulis)


def _even_depth_paths(
    prep_gate: ModelGate,
    meas_gate: ModelGate,
    gate: ModelGate,
    input_paulis: QubitSparsePauliList,
    output_paulis: QubitSparsePauliList | None = None,
) -> Iterator[Path]:
    output_paulis = output_paulis or input_paulis
    ident = QubitSparsePauli.identity(num_qubits=input_paulis.num_qubits)

    for input_pauli in input_paulis:
        input_fidelity = FidelityIndex.from_transition(
            gate=gate, in_pauli=input_pauli, out_pauli=gate.clifford_propagate(input_pauli)
        )
        for output_pauli in output_paulis:
            if not np.array_equal(input_pauli.indices, output_pauli.indices):
                continue
            output_fidelity = FidelityIndex.from_gate(gate, pauli=output_pauli)

            if np.array_equal(
                input_fidelity.transition[1].indices, output_fidelity.transition[0].indices
            ):
                yield Path(
                    start_fragment=[
                        FidelityIndex.from_gate(
                            prep_gate,
                            pauli=ident,
                            out_bit_indices=frozenset(input_fidelity.transition[0].indices),
                        )
                    ],
                    repeatable_fragment=[input_fidelity, output_fidelity],
                    end_fragment=[
                        FidelityIndex.from_gate(
                            meas_gate,
                            pauli=ident,
                            in_bit_indices=frozenset(output_fidelity.transition[1].indices),
                        )
                    ],
                )
