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
from qiskit.quantum_info import Clifford, PhasedQubitSparsePauli, QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate
from qiskit_noise_learning.sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionSequence,
    PartialPauliPermutation,
    Path,
)


def test_construction():
    """Test construction and attributes."""
    ident = [((0,), Clifford(QuantumCircuit(1)))]
    prep = ModelGate("P", ident, qubit_idxs=range(2), prep_idxs=range(2))
    gate = ModelGate("L0", ident, qubit_idxs=range(2))
    meas = ModelGate("M", ident, qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex.from_gate(
            prep, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2))
        )
    ]
    repeatable_fragment = [FidelityIndex.from_gate(gate, pauli=QubitSparsePauli("IX"))] * 2
    end_fragment = [
        FidelityIndex.from_gate(
            meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2))
        )
    ]

    path = Path(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )

    assert path.start_fragment == start_fragment
    assert path.repeatable_fragment == repeatable_fragment
    assert path.end_fragment == end_fragment
    assert path.fragment_depth is None


def test_construction_with_depth():
    """Test construction with a specified fragment_depth."""
    ident = [((0, 1), Clifford(QuantumCircuit(2)))]
    prep = ModelGate("P", ident, qubit_idxs=range(2), prep_idxs=range(2))
    gate = ModelGate("L0", ident, qubit_idxs=range(2))
    meas = ModelGate("M", ident, qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex.from_gate(
            prep, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2))
        )
    ]
    repeatable_fragment = [FidelityIndex.from_gate(gate, pauli=QubitSparsePauli("IX"))] * 2
    end_fragment = [
        FidelityIndex.from_gate(
            meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2))
        )
    ]

    path = Path(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
        fragment_depth=3,
    )

    assert path.fragment_depth == 3
    assert len(path) == 8


def test_iter():
    """Test __iter__ yields elements in the correct order."""

    ident = [((0, 1), Clifford(QuantumCircuit(2)))]
    prep = ModelGate("S", ident, qubit_idxs=range(2), prep_idxs=range(2))
    gate0 = ModelGate("G0", ident, qubit_idxs=range(2))
    gate1 = ModelGate("G1", ident, qubit_idxs=range(2))
    meas = ModelGate("M", ident, qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex.from_gate(
            prep, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2))
        )
    ]
    repeatable_fragment = [
        FidelityIndex.from_gate(gate0, pauli=QubitSparsePauli("IX")),
        FidelityIndex.from_gate(gate1, pauli=QubitSparsePauli("IY")),
    ]
    end_fragment = [
        FidelityIndex.from_gate(
            meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2))
        )
    ]

    path = Path(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
        fragment_depth=2,
    )
    items = list(path)

    assert len(items) == 6
    assert items[0] == start_fragment[0]
    assert items[1] == repeatable_fragment[0]
    assert items[2] == repeatable_fragment[1]
    assert items[3] == repeatable_fragment[0]
    assert items[4] == repeatable_fragment[1]
    assert items[5] == end_fragment[0]


def test_getitem():
    """Test __getitem__ returns correct elements by index."""

    ident = [((0, 1), Clifford(QuantumCircuit(2)))]
    prep0 = ModelGate("S0", ident, qubit_idxs=range(2), prep_idxs=range(2))
    prep1 = ModelGate("S1", ident, qubit_idxs=range(2), prep_idxs=range(2))
    gate0 = ModelGate("G0", ident, qubit_idxs=range(2))
    gate1 = ModelGate("G1", ident, qubit_idxs=range(2))
    meas = ModelGate("M", ident, qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex.from_gate(
            prep0, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2))
        ),
        FidelityIndex.from_gate(
            prep1, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2))
        ),
    ]
    repeatable_fragment = [
        FidelityIndex.from_gate(gate0, pauli=QubitSparsePauli("IX")),
        FidelityIndex.from_gate(gate1, pauli=QubitSparsePauli("IY")),
    ]
    end_fragment = [
        FidelityIndex.from_gate(
            meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2))
        )
    ]

    path = Path(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
        fragment_depth=3,
    )

    # start fragment
    assert path[0] == start_fragment[0]
    assert path[1] == start_fragment[1]
    # repeatable fragment (3 repetitions)
    assert path[2] == repeatable_fragment[0]
    assert path[3] == repeatable_fragment[1]
    assert path[4] == repeatable_fragment[0]
    assert path[5] == repeatable_fragment[1]
    assert path[6] == repeatable_fragment[0]
    assert path[7] == repeatable_fragment[1]
    # end fragment
    assert path[8] == end_fragment[0]


def test_getitem_out_of_bounds():
    """Test __getitem__ raises IndexError for out of bounds indices."""

    ident = [((0, 1), Clifford(QuantumCircuit(2)))]
    prep = ModelGate("S", ident, qubit_idxs=range(2), prep_idxs=range(2))
    gate = ModelGate("G", ident, qubit_idxs=range(2))
    meas = ModelGate("M", ident, qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex.from_gate(
            prep, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2))
        )
    ]
    repeatable_fragment = [FidelityIndex.from_gate(gate, pauli=QubitSparsePauli("IX"))]
    end_fragment = [
        FidelityIndex.from_gate(
            meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2))
        )
    ]

    path = Path(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
        fragment_depth=2,
    )
    assert len(path) == 4

    with pytest.raises(IndexError, match="exceeds length"):
        path[4]

    with pytest.raises(IndexError, match="negative"):
        path[-1]


def test_observable_indices(gate_set_1q):
    """Test the *_fragment_observable_indices properties."""
    path = Path(
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

    assert path.start_fragment_observable_indices == [[]]
    assert path.repeatable_fragment_observable_indices == [[]]
    assert path.end_fragment_observable_indices == [[0]]

    # example with MCM
    qc = QuantumCircuit(1)
    qc.sx(0)
    gate = ModelGate("L0", cliffords=[((1,), qc)], qubit_idxs=range(2), meas_idxs=[0])
    prep = ModelGate("P", cliffords=[], qubit_idxs=range(2), prep_idxs=[0, 1])
    meas = ModelGate("M", cliffords=[], qubit_idxs=range(2), meas_idxs=[0, 1])
    path = Path(
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
    assert path.start_fragment_observable_indices == [[]]
    assert path.repeatable_fragment_observable_indices == [[]]
    assert path.end_fragment_observable_indices == [[0, 1]]


def test_to_instruction_sequence_single_qubit_single_box(gate_set_1q):
    """Tests for path to instruction sequence with a single qubit and single layer."""
    # Clifford maps X -> -Y, Y -> Z, Z -> -X

    path = Path(
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
    expected = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("Y", "X")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
            ApplyGate("M"),
        ],
    )
    assert path.to_instruction_sequence() == expected

    # similar test but with no repeatable fragment
    path = Path(
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
    expected = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "Z")}]),
        ],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
    )
    assert path.to_instruction_sequence() == expected


def test_to_instruction_sequence_preserves_depth(gate_set_1q):
    """Test that to_instruction_sequence preserves the fragment_depth."""
    path = Path(
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
        fragment_depth=5,
    )
    assert path.to_instruction_sequence().fragment_depth == 5


def test_to_instruction_sequence_single_qubit_single_box_deep_repetition(gate_set_1q):
    """Tests for path to instruction with a single qubit and single layer and a repeatable path
    fragment of length 3.
    """
    # Clifford maps X -> -Y, Y -> Z, Z -> -X
    path = Path(
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
    expected = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("Y", "Y")}]),
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate("M"),
        ],
    )
    assert path.to_instruction_sequence() == expected


def test_to_instruction_sequence_single_qubit_two_boxes(gate_set_1q):
    """Tests for path to instruction with a single qubit and two layers."""
    # L0 Clifford maps X -> -Y, Y -> Z, Z -> -X, L1 is XGate
    path = Path(
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
    expected = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate("L1"),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
            ApplyGate("M"),
        ],
    )
    assert path.to_instruction_sequence() == expected


def test_is_traversed_by(gate_set_1q):
    """Standard examples."""

    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L1"),
            PartialPauliPermutation.from_sets([{("Y", "Y")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate("M"),
        ],
    )
    assert path.is_traversed_by(inst_seq)

    wrong_inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L1"),
            PartialPauliPermutation.from_sets([{("Y", "Y")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
            ApplyGate("M"),
        ],
    )
    assert not path.is_traversed_by(wrong_inst_seq)

    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate("L1"),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
            ApplyGate("M"),
        ],
    )

    assert path.is_traversed_by(inst_seq)


def test_is_traversed_by_depth_mismatch(gate_set_1q):
    """Test that is_traversed_by returns False for fragment_depth mismatch."""
    path = Path(
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
        fragment_depth=3,
    )
    inst_seq = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
        fragment_depth=4,
    )
    assert not path.is_traversed_by(inst_seq)


def test_is_traversed_by_multiple_permutations(gate_set_1q):
    path = Path(
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

    inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L1"),
            PartialPauliPermutation.from_sets([{("Y", "Y")}]),
            PartialPauliPermutation.from_sets([{("Y", "X")}]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate("M"),
        ],
    )
    assert path.is_traversed_by(inst_seq)


def test_is_traversed_by_edge_cases(gate_set_1q, gate_set_cz):
    # no repeatable fragment
    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
        ],
        repeatable_fragment=[],
        end_fragment=[
            ApplyGate("M"),
        ],
    )
    assert path.is_traversed_by(inst_seq)

    # meaningless pauli permutations in repeatable fragment
    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
        ],
        repeatable_fragment=[
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
        ],
        end_fragment=[
            ApplyGate("M"),
        ],
    )
    assert path.is_traversed_by(inst_seq)

    # only start fragment
    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[ApplyGate("P"), ApplyGate("M")],
        repeatable_fragment=[],
        end_fragment=[],
    )
    assert path.is_traversed_by(inst_seq)

    # more apply gates than fidelity indices in same fragment
    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[ApplyGate("P"), ApplyGate("CZ")],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
    )
    assert not path.is_traversed_by(inst_seq)


def test_is_traversed_by_errors(gate_set_1q):
    # does not begin with identity
    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
        ],
        repeatable_fragment=[],
        end_fragment=[
            ApplyGate("M"),
        ],
    )
    with pytest.raises(ValueError, match="Path does not begin with identity."):
        path.is_traversed_by(inst_seq)

    path = Path(
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

    with pytest.raises(ValueError, match="Path does not end with identity."):
        path.is_traversed_by(inst_seq)


def test_extend_permutations(gate_set_cz):
    # test case on standard structures
    path_IX = Path(
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

    path_XI = Path(
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

    inst_seq_IX = path_IX.to_instruction_sequence()
    inst_seq_XI = path_XI.to_instruction_sequence()

    assert path_IX.extend_permutations(inst_seq_XI) == inst_seq_IX.merge(inst_seq_XI)

    # test for manually built paths without the standard structure
    inst_seq_IX = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "X")}, set()]),
        ],
        repeatable_fragment=[ApplyGate("CZ")] * 2,
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}, set()]),
            ApplyGate("M"),
        ],
    )
    inst_seq_XI = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([set(), {("Z", "X")}]),
        ],
        repeatable_fragment=[ApplyGate("CZ")] * 2,
        end_fragment=[
            PartialPauliPermutation.from_sets([set(), {("X", "Z")}]),
            ApplyGate("M"),
        ],
    )

    assert path_IX.extend_permutations(inst_seq_XI) == inst_seq_IX.merge(inst_seq_XI)


def test_extend_permutations_edge_cases(gate_set_cz):
    # no repeatable fragment
    path_IX = Path(
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

    path_XI = Path(
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

    inst_seq_IX = path_IX.to_instruction_sequence()
    inst_seq_XI = path_XI.to_instruction_sequence()

    assert path_IX.extend_permutations(inst_seq_XI) == inst_seq_IX.merge(inst_seq_XI)


def test_extend_permutations_failures(gate_set_cz):
    """Test situations in which extension is not possible."""

    # more apply gates than fidelity indices in same fragment
    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[ApplyGate("P"), ApplyGate("CZ")],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
    )
    assert path.extend_permutations(inst_seq) is None


def test_fragment_sign_flips_case1(gate_set_1q):
    """Single gate application."""
    # L0 maps X -> -Y, Y -> Z, Z -> -X

    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("Y", "X")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
            ApplyGate("M"),
        ],
    ).complete()

    # expected flips
    start_flip = inst_seq.start_fragment[1].propagate(PhasedQubitSparsePauli("Z")).phase == 2
    # the gate flips the sign as well
    repeatable_flip = not (
        inst_seq.repeatable_fragment[1].propagate(PhasedQubitSparsePauli("Y")).phase == 2
    )
    end_flip = inst_seq.end_fragment[0].propagate(PhasedQubitSparsePauli("X")).phase == 2
    assert path.fragment_sign_flips(inst_seq) == (start_flip ^ end_flip, repeatable_flip)


def test_fragment_sign_flips_case2(gate_set_1q):
    """Multiple gates, one flips sign one doesn't."""

    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("Z", "Z")}]),
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate("M"),
        ],
    ).complete()

    # expected flips
    start_flip = inst_seq.start_fragment[1].propagate(PhasedQubitSparsePauli("Z")).phase == 2
    # the gate also flips the sign
    repeatable_flip = not (
        (inst_seq.repeatable_fragment[1].propagate(PhasedQubitSparsePauli("Z")).phase == 2)
        ^ (inst_seq.repeatable_fragment[3].propagate(PhasedQubitSparsePauli("X")).phase == 2)
    )
    end_flip = inst_seq.end_fragment[0].propagate(PhasedQubitSparsePauli("Y")).phase == 2
    assert path.fragment_sign_flips(inst_seq) == (start_flip ^ end_flip, repeatable_flip)


def test_fragment_sign_flips_case3(gate_set_1q):
    """Multiple gates, both flip sign."""

    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("X", "X")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
            ApplyGate("M"),
        ],
    ).complete()

    # expected flips
    start_flip = inst_seq.start_fragment[1].propagate(PhasedQubitSparsePauli("Z")).phase == 2
    # signs from gates should cancel out
    repeatable_flip = (
        inst_seq.repeatable_fragment[1].propagate(PhasedQubitSparsePauli("Y")).phase == 2
    ) ^ (inst_seq.repeatable_fragment[3].propagate(PhasedQubitSparsePauli("X")).phase == 2)
    end_flip = inst_seq.end_fragment[0].propagate(PhasedQubitSparsePauli("X")).phase == 2
    assert path.fragment_sign_flips(inst_seq) == (start_flip ^ end_flip, repeatable_flip)


def test_fragment_sign_flips_case4(gate_set_1q):
    """Gate sign flips in start fragment."""

    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("X", "Z")}]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Z", "Z")}]),
            ApplyGate("M"),
        ],
    ).complete()

    # expected sign flips
    start_flip = not (
        (inst_seq.start_fragment[1].propagate(PhasedQubitSparsePauli("Z")).phase == 2)
        ^ (inst_seq.start_fragment[3].propagate(PhasedQubitSparsePauli("Y")).phase == 2)
    )
    repeatable_flip = not (
        inst_seq.repeatable_fragment[1].propagate(PhasedQubitSparsePauli("X")).phase == 2
    )
    end_flip = inst_seq.end_fragment[0].propagate(PhasedQubitSparsePauli("Z")).phase == 2
    assert path.fragment_sign_flips(inst_seq) == (start_flip ^ end_flip, repeatable_flip)


def test_fragment_sign_flips_errors(gate_set_1q):
    path = Path(
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
    inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[],
        end_fragment=[
            ApplyGate("M"),
        ],
    ).complete()

    with pytest.raises(ValueError, match="Cannot compute signs"):
        path.fragment_sign_flips(inst_seq)

    path.start_fragment[0] = FidelityIndex.from_transition(
        gate_set_1q["L0"], QubitSparsePauli("Z"), QubitSparsePauli("X")
    )

    with pytest.raises(ValueError, match="Path does not begin with identity"):
        path.fragment_sign_flips(inst_seq)

    path = Path(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            ),
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
    inst_seq = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[],
        end_fragment=[
            ApplyGate("M"),
        ],
    ).complete()

    with pytest.raises(ValueError, match="repeatable fragment"):
        path.fragment_sign_flips(inst_seq)


def test_hash():
    prep = ModelGate("P", [], qubit_idxs=range(2), prep_idxs=range(2))
    gate = ModelGate("L0", [], qubit_idxs=range(2))
    meas = ModelGate("M", [], qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex.from_gate(
            prep, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2))
        )
    ]
    repeatable_fragment = [FidelityIndex.from_gate(gate, pauli=QubitSparsePauli("IX"))] * 2
    end_fragment = [
        FidelityIndex.from_gate(
            meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2))
        )
    ]

    path = Path(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )
    assert isinstance(hash(path), int)

    # with fragment_depth
    path_with_depth = Path(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
        fragment_depth=5,
    )
    assert isinstance(hash(path_with_depth), int)
    # different fragment_depth should give different hash
    assert hash(path) != hash(path_with_depth)


def test_bind_at(gate_set_1q):
    """Test bind_at returns a new Path with the specified fragment_depth."""
    ident = [((0,), Clifford(QuantumCircuit(1)))]
    prep = ModelGate("P", ident, qubit_idxs=range(1), prep_idxs=range(1))
    gate = ModelGate("L0", ident, qubit_idxs=range(1))
    meas = ModelGate("M", ident, qubit_idxs=range(1), meas_idxs=range(1))

    start_fragment = [FidelityIndex.from_gate(prep, pauli=QubitSparsePauli("I"))]
    repeatable_fragment = [FidelityIndex.from_gate(gate, pauli=QubitSparsePauli("X"))]
    end_fragment = [FidelityIndex.from_gate(meas, pauli=QubitSparsePauli("I"))]

    path = Path(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )
    assert path.fragment_depth is None

    bound = path.bind_at(3)
    assert bound.fragment_depth == 3
    assert bound.start_fragment == start_fragment
    assert bound.repeatable_fragment == repeatable_fragment
    assert bound.end_fragment == end_fragment
    assert isinstance(bound, Path)

    # unbind reverses it
    unbound = bound.unbind()
    assert unbound.fragment_depth is None
    assert unbound == path
