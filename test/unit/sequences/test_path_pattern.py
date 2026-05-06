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

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, XGate
from qiskit.quantum_info import Clifford, PhasedQubitSparsePauli, QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionPattern,
    PartialPauliPermutation,
    PathPattern,
)


@pytest.fixture()
def gate_set_1q():
    model_gate_set = ModelGateSet(1)
    ident = [((0,), Clifford(QuantumCircuit(1)))]
    model_gate_set.add_gate(ModelGate("P", ident, prep_idxs=range(1)))
    model_gate_set.add_gate(ModelGate("M", ident, meas_idxs=range(1)))
    # Clifford maps X -> -Y, Y -> Z, Z -> -X
    model_gate_set.add_gate(
        ModelGate("L0", [((0,), Clifford([[True, True, True], [True, False, True]]))])
    )
    model_gate_set.add_gate(ModelGate("L1", [((0,), Clifford(XGate()))]))
    return model_gate_set


@pytest.fixture()
def gate_set_cz():
    model_gate_set = ModelGateSet(2)
    model_gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    model_gate_set.add_gate(
        ModelGate("P", [((0, 1), Clifford(QuantumCircuit(2)))], prep_idxs=range(2))
    )
    model_gate_set.add_gate(
        ModelGate("M", [((0, 1), Clifford(QuantumCircuit(2)))], meas_idxs=range(2))
    )
    return model_gate_set


def test_construction():
    """Test construction and attributes."""
    ident = [((0,), Clifford(QuantumCircuit(1)))]
    prep = ModelGate("P", ident, qubit_idxs=range(2), prep_idxs=range(2))
    gate = ModelGate("L0", ident, qubit_idxs=range(2))
    meas = ModelGate("M", ident, qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex(prep, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2)))
    ]
    repeatable_fragment = [FidelityIndex(gate, pauli=QubitSparsePauli("IX"))] * 2
    end_fragment = [
        FidelityIndex(meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2)))
    ]

    path_pattern = PathPattern(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )

    assert path_pattern.start_fragment == start_fragment
    assert path_pattern.repeatable_fragment == repeatable_fragment
    assert path_pattern.end_fragment == end_fragment


def test_observable_indices(gate_set_1q):
    """Test the *_fragment_observable_indices properties."""
    pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
            )
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )

    assert pattern.start_fragment_observable_indices == [[]]
    assert pattern.repeatable_fragment_observable_indices == [[]]
    assert pattern.end_fragment_observable_indices == [[0]]

    # example with MCM
    qc = QuantumCircuit(1)
    qc.sx(0)
    gate = ModelGate("L0", cliffords=[((1,), qc)], qubit_idxs=range(2), meas_idxs=[0])
    prep = ModelGate("P", cliffords=[], qubit_idxs=range(2), prep_idxs=[0, 1])
    meas = ModelGate("M", cliffords=[], qubit_idxs=range(2), meas_idxs=[0, 1])
    pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(prep, QubitSparsePauli("II"), QubitSparsePauli("ZZ"))
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(gate, QubitSparsePauli("XZ"), QubitSparsePauli("XZ"))
        ],
        end_fragment=[
            FidelityIndex.from_transition(meas, QubitSparsePauli("ZZ"), QubitSparsePauli("II"))
        ],
    )
    assert pattern.start_fragment_observable_indices == [[]]
    assert pattern.repeatable_fragment_observable_indices == [[]]
    assert pattern.end_fragment_observable_indices == [[0, 1]]


def test_to_instruction_single_qubit_single_box(gate_set_1q):
    """Tests for pattern to instruction with a single qubit and single layer."""
    # Clifford maps X -> -Y, Y -> Z, Z -> -X

    pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
            )
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    expected = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("Y", "X")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
            ApplyGate(gate_set_1q["M"]),
        ],
    )
    assert pattern.to_instruction_pattern() == expected

    # similar test but with no repeatable fragment
    pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    expected = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "Z")}]),
        ],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set_1q["M"])],
    )
    assert pattern.to_instruction_pattern() == expected


def test_to_instruction_single_qubit_single_box_deep_repetition(gate_set_1q):
    """Tests for pattern to instruction with a single qubit and single layer and a repeatable path
    fragment of length 3.
    """
    # Clifford maps X -> -Y, Y -> Z, Z -> -X
    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("Y"), QubitSparsePauli("Z")
            ),
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
            ),
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("Y"), QubitSparsePauli("Z")
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    expected = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("Y", "Y")}]),
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(gate_set_1q["M"]),
        ],
    )
    assert path_pattern.to_instruction_pattern() == expected


def test_to_instruction_single_qubit_two_boxes(gate_set_1q):
    """Tests for pattern to instruction with a single qubit and two layers."""
    # L0 Clifford maps X -> -Y, Y -> Z, Z -> -X, L1 is XGate
    pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
            ),
            FidelityIndex.from_transition(
                gate_set_1q["L1"], QubitSparsePauli("Z"), QubitSparsePauli("Z")
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    expected = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(gate_set_1q["L1"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
            ApplyGate(gate_set_1q["M"]),
        ],
    )
    assert pattern.to_instruction_pattern() == expected


def test_is_traversed_by(gate_set_1q):
    """Standard examples."""

    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L1"], QubitSparsePauli("Y"), QubitSparsePauli("Y")
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    inst_pattern = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate(gate_set_1q["L1"]),
            PartialPauliPermutation.from_sets([{("Y", "Y")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(gate_set_1q["M"]),
        ],
    )
    assert path_pattern.is_traversed_by(inst_pattern)

    wrong_inst_pattern = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate(gate_set_1q["L1"]),
            PartialPauliPermutation.from_sets([{("Y", "Y")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
            ApplyGate(gate_set_1q["M"]),
        ],
    )
    assert not path_pattern.is_traversed_by(wrong_inst_pattern)

    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
            ),
            FidelityIndex.from_transition(
                gate_set_1q["L1"], QubitSparsePauli("Z"), QubitSparsePauli("Z")
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    inst_pattern = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(gate_set_1q["L1"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
            ApplyGate(gate_set_1q["M"]),
        ],
    )

    assert path_pattern.is_traversed_by(inst_pattern)


def test_is_traversed_by_multiple_permutations(gate_set_1q):
    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L1"], QubitSparsePauli("Y"), QubitSparsePauli("Y")
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )

    inst_pattern = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate(gate_set_1q["L1"]),
            PartialPauliPermutation.from_sets([{("Y", "Y")}]),
            PartialPauliPermutation.from_sets([{("Y", "X")}]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(gate_set_1q["M"]),
        ],
    )
    assert path_pattern.is_traversed_by(inst_pattern)


def test_is_traversed_by_edge_cases(gate_set_1q, gate_set_cz):
    # no repeatable fragment
    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    inst_pattern = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
        ],
        repeatable_fragment=[],
        end_fragment=[
            ApplyGate(gate_set_1q["M"]),
        ],
    )
    assert path_pattern.is_traversed_by(inst_pattern)

    # meaningless pauli permutations in repeatable fragment
    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    inst_pattern = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
        ],
        repeatable_fragment=[
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
        ],
        end_fragment=[
            ApplyGate(gate_set_1q["M"]),
        ],
    )
    assert path_pattern.is_traversed_by(inst_pattern)

    # only start fragment
    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            ),
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            ),
        ],
        repeatable_fragment=[],
        end_fragment=[],
    )
    inst_pattern = InstructionPattern(
        start_fragment=[ApplyGate(gate_set_1q["P"]), ApplyGate(gate_set_1q["M"])],
        repeatable_fragment=[],
        end_fragment=[],
    )
    assert path_pattern.is_traversed_by(inst_pattern)

    # more apply gates than fidelity indices in same fragment
    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
            )
        ],
        repeatable_fragment=[],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
            )
        ],
    )
    instruction_pattern = InstructionPattern(
        start_fragment=[ApplyGate(gate_set_cz["P"]), ApplyGate(gate_set_cz["CZ"])],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set_cz["M"])],
    )
    assert not path_pattern.is_traversed_by(instruction_pattern)


def test_is_traversed_by_errors(gate_set_1q):
    # does not begin with identity
    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L1"], QubitSparsePauli("Y"), QubitSparsePauli("Y")
            )
        ],
        repeatable_fragment=[],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    inst_pattern = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
        ],
        repeatable_fragment=[],
        end_fragment=[
            ApplyGate(gate_set_1q["M"]),
        ],
    )
    with pytest.raises(ValueError, match="Path pattern does not begin with identity."):
        path_pattern.is_traversed_by(inst_pattern)

    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("Z")
            )
        ],
    )

    with pytest.raises(ValueError, match="Path pattern does not end with identity."):
        path_pattern.is_traversed_by(inst_pattern)


def test_extend_permutations(gate_set_cz):
    # test case on standard structures
    path_pattern_IX = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["P"],
                in_pauli=QubitSparsePauli("II"),
                out_pauli=QubitSparsePauli("IZ"),
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("IX"),
                out_pauli=QubitSparsePauli("ZX"),
            ),
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("ZX"),
                out_pauli=QubitSparsePauli("IX"),
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["M"],
                in_pauli=QubitSparsePauli("IZ"),
                out_pauli=QubitSparsePauli("II"),
            )
        ],
    )

    path_pattern_XI = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["P"],
                in_pauli=QubitSparsePauli("II"),
                out_pauli=QubitSparsePauli("ZI"),
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("XI"),
                out_pauli=QubitSparsePauli("XZ"),
            ),
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("XZ"),
                out_pauli=QubitSparsePauli("XI"),
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["M"],
                in_pauli=QubitSparsePauli("ZI"),
                out_pauli=QubitSparsePauli("II"),
            )
        ],
    )

    inst_pattern_IX = path_pattern_IX.to_instruction_pattern()
    inst_pattern_XI = path_pattern_XI.to_instruction_pattern()

    assert path_pattern_IX.extend_permutations(inst_pattern_XI) == inst_pattern_IX.merge(
        inst_pattern_XI
    )

    # test for manually built paths without the standard structure
    inst_pattern_IX = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_cz["P"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}, set()]),
        ],
        repeatable_fragment=[ApplyGate(gate_set_cz["CZ"])] * 2,
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}, set()]),
            ApplyGate(gate_set_cz["M"]),
        ],
    )
    inst_pattern_XI = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_cz["P"]),
            PartialPauliPermutation.from_sets([set(), {("Z", "X")}]),
        ],
        repeatable_fragment=[ApplyGate(gate_set_cz["CZ"])] * 2,
        end_fragment=[
            PartialPauliPermutation.from_sets([set(), {("X", "Z")}]),
            ApplyGate(gate_set_cz["M"]),
        ],
    )

    assert path_pattern_IX.extend_permutations(inst_pattern_XI) == inst_pattern_IX.merge(
        inst_pattern_XI
    )


def test_extend_permutations_edge_cases(gate_set_cz):
    # no repeatable fragment
    path_pattern_IX = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["P"],
                in_pauli=QubitSparsePauli("II"),
                out_pauli=QubitSparsePauli("IZ"),
            )
        ],
        repeatable_fragment=[],
        end_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["M"],
                in_pauli=QubitSparsePauli("IZ"),
                out_pauli=QubitSparsePauli("II"),
            )
        ],
    )

    path_pattern_XI = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["P"],
                in_pauli=QubitSparsePauli("II"),
                out_pauli=QubitSparsePauli("ZI"),
            )
        ],
        repeatable_fragment=[],
        end_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["M"],
                in_pauli=QubitSparsePauli("ZI"),
                out_pauli=QubitSparsePauli("II"),
            )
        ],
    )

    inst_pattern_IX = path_pattern_IX.to_instruction_pattern()
    inst_pattern_XI = path_pattern_XI.to_instruction_pattern()

    assert path_pattern_IX.extend_permutations(inst_pattern_XI) == inst_pattern_IX.merge(
        inst_pattern_XI
    )


def test_extend_permutations_failures(gate_set_cz):
    """Test situations in which extension is not possible."""

    # more apply gates than fidelity indices in same fragment
    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
            )
        ],
        repeatable_fragment=[],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
            )
        ],
    )
    instruction_pattern = InstructionPattern(
        start_fragment=[ApplyGate(gate_set_cz["P"]), ApplyGate(gate_set_cz["CZ"])],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set_cz["M"])],
    )
    assert path_pattern.extend_permutations(instruction_pattern) is None


def test_sign_flips_case1(gate_set_1q):
    """Single gate application."""
    # L0 maps X -> -Y, Y -> Z, Z -> -X

    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
            )
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    instruction_pattern = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("Y", "X")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
            ApplyGate(gate_set_1q["M"]),
        ],
    ).complete()

    # expected flips
    start_flip = (
        instruction_pattern.start_fragment[1].propagate(PhasedQubitSparsePauli("Z")).phase == 2
    )
    # the gate flips the sign as well
    repeatable_flip = not (
        instruction_pattern.repeatable_fragment[1].propagate(PhasedQubitSparsePauli("Y")).phase == 2
    )
    end_flip = instruction_pattern.end_fragment[0].propagate(PhasedQubitSparsePauli("X")).phase == 2
    assert path_pattern.sign_flips(instruction_pattern) == (start_flip ^ end_flip, repeatable_flip)


def test_sign_flips_case2(gate_set_1q):
    """Multiple gates, one flips sign one doesn't."""

    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("Y"), QubitSparsePauli("Z")
            ),
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("Z"), QubitSparsePauli("X")
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    instruction_pattern = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("Z", "Z")}]),
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(gate_set_1q["M"]),
        ],
    ).complete()

    # expected flips
    start_flip = (
        instruction_pattern.start_fragment[1].propagate(PhasedQubitSparsePauli("Z")).phase == 2
    )
    # the gate also flips the sign
    repeatable_flip = not (
        (
            instruction_pattern.repeatable_fragment[1].propagate(PhasedQubitSparsePauli("Z")).phase
            == 2
        )
        ^ (
            instruction_pattern.repeatable_fragment[3].propagate(PhasedQubitSparsePauli("X")).phase
            == 2
        )
    )
    end_flip = instruction_pattern.end_fragment[0].propagate(PhasedQubitSparsePauli("Y")).phase == 2
    assert path_pattern.sign_flips(instruction_pattern) == (start_flip ^ end_flip, repeatable_flip)


def test_sign_flips_case3(gate_set_1q):
    """Multiple gates, both flip sign."""

    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
            ),
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("Z"), QubitSparsePauli("X")
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    instruction_pattern = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("X", "X")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
            ApplyGate(gate_set_1q["M"]),
        ],
    ).complete()

    # expected flips
    start_flip = (
        instruction_pattern.start_fragment[1].propagate(PhasedQubitSparsePauli("Z")).phase == 2
    )
    # signs from gates should cancel out
    repeatable_flip = (
        instruction_pattern.repeatable_fragment[1].propagate(PhasedQubitSparsePauli("Y")).phase == 2
    ) ^ (
        instruction_pattern.repeatable_fragment[3].propagate(PhasedQubitSparsePauli("X")).phase == 2
    )
    end_flip = instruction_pattern.end_fragment[0].propagate(PhasedQubitSparsePauli("X")).phase == 2
    assert path_pattern.sign_flips(instruction_pattern) == (start_flip ^ end_flip, repeatable_flip)


def test_sign_flips_case4(gate_set_1q):
    """Gate sign flips in start fragment."""

    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            ),
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
            ),
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("Z"), QubitSparsePauli("X")
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    instruction_pattern = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
        ],
        repeatable_fragment=[
            ApplyGate(gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Z", "Z")}]),
            ApplyGate(gate_set_1q["M"]),
        ],
    ).complete()

    # expected sign flips
    start_flip = not (
        (instruction_pattern.start_fragment[1].propagate(PhasedQubitSparsePauli("Z")).phase == 2)
        ^ (instruction_pattern.start_fragment[3].propagate(PhasedQubitSparsePauli("Y")).phase == 2)
    )
    repeatable_flip = not (
        instruction_pattern.repeatable_fragment[1].propagate(PhasedQubitSparsePauli("X")).phase == 2
    )
    end_flip = instruction_pattern.end_fragment[0].propagate(PhasedQubitSparsePauli("Z")).phase == 2
    assert path_pattern.sign_flips(instruction_pattern) == (start_flip ^ end_flip, repeatable_flip)


def test_sign_flips_errors(gate_set_1q):
    path_pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            ),
        ],
        repeatable_fragment=[],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )
    instruction_pattern = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[],
        end_fragment=[
            ApplyGate(gate_set_1q["M"]),
        ],
    ).complete()

    with pytest.raises(ValueError, match="Cannot compute signs"):
        path_pattern.sign_flips(instruction_pattern)

    path_pattern.start_fragment[0] = FidelityIndex.from_transition(
        gate_set_1q["L0"], QubitSparsePauli("Z"), QubitSparsePauli("X")
    )

    with pytest.raises(ValueError, match="Path pattern does not begin with identity"):
        path_pattern.sign_flips(instruction_pattern)


def test_hash():
    prep = ModelGate("P", [], qubit_idxs=range(2), prep_idxs=range(2))
    gate = ModelGate("L0", [], qubit_idxs=range(2))
    meas = ModelGate("M", [], qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex(prep, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2)))
    ]
    repeatable_fragment = [FidelityIndex(gate, pauli=QubitSparsePauli("IX"))] * 2
    end_fragment = [
        FidelityIndex(meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2)))
    ]

    path_pattern = PathPattern(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )
    assert isinstance(hash(path_pattern), int)
