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

"""EvenDepthVanillaPaths stage."""

from collections.abc import Iterator

from qiskit.quantum_info import QubitSparsePauli, QubitSparsePauliList

from qiskit_noise_learning.gate_sets import ModelGate
from qiskit_noise_learning.sequences import FidelityIndex, Path

from ...experiment import Experiment
from ..utils import default_gates, default_meas_gate, default_prep_gate
from .add_paths import AddPaths


class FullRankPaths(AddPaths):
    """Generate unbound vanilla paths with repetitions of two gate applications.

    For each target gate, generates paths where the repeatable fragment consists of two
    applications of the gate.

    Args:
        prep_gate: The preparation gate. If ``None``, defaults to the gate named ``"P"``.
        meas_gate: The measurement gate. If ``None``, defaults to the gate named ``"M"``.
        gates: Gates to generate paths for. If ``None``, defaults to all non-SPAM gates in
            the gate set.
        fidelity_paulis: Optional mapping from gate name to the Paulis to use. If ``None``, uses
            ``fidelity_model.generators[gate_name]`` for each gate.
    """

    @property
    def required_fields(self) -> tuple[str, ...]:
        if self._gates is None or self._prep_gate is None or self._meas_gate is None:
            return ("fidelity_model",)
        if self._fidelity_paulis is None:
            return ("fidelity_model",)
        return ()

    def __init__(
        self,
        *,
        prep_gate: ModelGate | None = None,
        meas_gate: ModelGate | None = None,
        gates: list[ModelGate] | None = None,
        fidelity_paulis: dict[str, QubitSparsePauliList] | None = None,
        noise_after_gate: bool = True,
    ):
        self._prep_gate = prep_gate
        self._meas_gate = meas_gate
        self._gates = gates
        self._fidelity_paulis = fidelity_paulis
        self._noise_after_gate = noise_after_gate

    def _generate_paths(self, experiment: Experiment) -> Iterator[Path]:
        gate_set = experiment.gate_set
        prep_gate = self._prep_gate or default_prep_gate(gate_set)
        meas_gate = self._meas_gate or default_meas_gate(gate_set)
        gates = self._gates or default_gates(gate_set)

        for gate in gates:
            if self._fidelity_paulis is not None:
                paulis = self._fidelity_paulis[gate.name]
            else:
                paulis = experiment.fidelity_model.generators[gate.name]

            yield from _full_rank_paths(prep_gate, meas_gate, gate, paulis, self._noise_after_gate)


def _full_rank_paths(
    prep_gate: ModelGate,
    meas_gate: ModelGate,
    gate: ModelGate,
    fidelity_paulis: QubitSparsePauliList,
    noise_after_gate: bool,
) -> Iterator[Path]:
    ident = QubitSparsePauli.identity(num_qubits=fidelity_paulis.num_qubits)

    variable_depth_paulis = []
    depth_1_paulis = []

    done_paulis = set()

    for fidelity_pauli in fidelity_paulis:
        
        hashable = fidelity_pauli.to_pauli()

        if hashable in done_paulis:
            # Already being learned.
            continue

        conjugate = gate.clifford_propagate(fidelity_pauli, inverse = noise_after_gate)

        if conjugate == fidelity_pauli:
            # Best case: deep circuit reveals this fidelity unambiguously and efficiently.
            variable_depth_paulis.append(fidelity_pauli)

        elif conjugate in fidelity_paulis:
            # Mid case: deep circuit reveals product of two desired fidelities efficiently.
            #     Use inefficient (depth-1) circuit to disambiguate.
            variable_depth_paulis.append(fidelity_pauli)
            depth_1_paulis.append(conjugate) # arbitrary choice
            done_paulis.add(conjugate.to_pauli())

        else:
            # Worst case: deep circuit reveals nothing useful.
            #     Use inefficient (depth-1) circuit to learn.
            depth_1_paulis.append(fidelity_pauli)
        
        done_paulis.add(hashable)

    # Build Paths:
    fidelity_paulis = variable_depth_paulis+depth_1_paulis
    depths = [None] * len(variable_depth_paulis) + [1] * len(depth_1_paulis)
    for fidelity_pauli, depth in zip(fidelity_paulis, depths):
        desired_fidelity = FidelityIndex(gate, pauli=fidelity_pauli)
        conjugate = gate.clifford_propagate(fidelity_pauli, inverse = noise_after_gate)
        conjugate_fidelity = FidelityIndex(gate, pauli=conjugate)
        if not noise_after_gate:
            input_fidelity = desired_fidelity
            output_fidelity = conjugate_fidelity
        else:
            input_fidelity = conjugate_fidelity
            output_fidelity = desired_fidelity

        yield Path(
            start_fragment=[
                FidelityIndex(
                    prep_gate,
                    pauli=ident,
                    out_bit_indices=frozenset(input_fidelity.transition[0].indices),
                )
            ],
            repeatable_fragment=[input_fidelity, output_fidelity],
            end_fragment=[
                FidelityIndex(
                    meas_gate,
                    pauli=ident,
                    in_bit_indices=frozenset(output_fidelity.transition[1].indices),
                )
            ],
            depth=depth,
        )
