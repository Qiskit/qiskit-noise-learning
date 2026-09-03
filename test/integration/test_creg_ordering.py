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

"""Tests for creg ordering conventions across multiple classes."""

import numpy as np
from qiskit.quantum_info import QubitSparsePauli

from qiskit_noise_learning.circuit_generator import ExecutorCircuitGenerator
from qiskit_noise_learning.gate_sets import QiskitGateSet
from qiskit_noise_learning.sequences import ApplyGate, FidelityIndex, InstructionSequence


def test_fidelity_index_mask_vs_circuit_creg():
    """Test the ordering of FidelityIndex.mask vs the ExecutorCircuitGenerator circuit cregs when
    qubit_subsets is unordered.
    """

    # qubit_subset is NOT ascending -> the auto-built "M" gate has qubit_idxs == (1, 0)
    gate_set = QiskitGateSet(2, qubit_subset=[1, 0])
    with gate_set.build_new_gate() as builder:
        builder.circuit.cz(0, 1)

    seq = InstructionSequence(
        [ApplyGate("P")], [ApplyGate("L0")], [ApplyGate("M")], fragment_depth=2
    )
    item, _, _ = ExecutorCircuitGenerator(gate_set).generate_samplex_item(
        [seq], num_randomizations=1
    )

    # which physical qubit the generated circuit measures into each bit of creg "meas0"
    creg = next(reg for reg in item.circuit.cregs if reg.name == "meas0")
    creg_order = [None] * len(creg)
    for instruction in item.circuit.data:
        if instruction.operation.name == "measure" and instruction.clbits[0] in creg:
            bit = list(creg).index(instruction.clbits[0])
            creg_order[bit] = item.circuit.find_bit(instruction.qubits[0]).index

    # a <Z> observable on physical qubit 0 alone
    fidelity_index = FidelityIndex.from_gate(
        gate_set.model_gate_set["M"], pauli=QubitSparsePauli.identity(2), in_z_idxs=frozenset({0})
    )
    selected = [q for q, keep in zip(creg_order, fidelity_index.mask) if keep]
    assert selected == fidelity_index.observable_idxs  # AssertionError: [1] != [0]


def test_the_second():
    """Need to conceptualize what this is actually testing before merging."""

    gate_set = QiskitGateSet(2, qubit_subset=[1, 0])
    with gate_set.build_new_gate() as builder:
        builder.circuit.cz(0, 1)

    seq = InstructionSequence(
        [ApplyGate("P")], [ApplyGate("L0")], [ApplyGate("M")], fragment_depth=2
    )
    item, creg_names, measurement_map = ExecutorCircuitGenerator(gate_set).generate_samplex_item(
        [seq], num_randomizations=1
    )

    # ground truth: which physical qubit is measured into each bit of each creg
    actual = {
        reg.name: [None] * len(reg) for reg in item.circuit.cregs if reg.name in measurement_map
    }
    for instruction in item.circuit.data:
        if instruction.operation.name == "measure":
            qubit = item.circuit.find_bit(instruction.qubits[0]).index
            reg, bit = item.circuit.find_bit(instruction.clbits[0]).registers[0]
            if reg.name in actual:
                actual[reg.name][bit] = qubit

    for name in measurement_map:
        np.testing.assert_array_equal(measurement_map[name], actual[name])  # [0, 1] != [1, 0]
