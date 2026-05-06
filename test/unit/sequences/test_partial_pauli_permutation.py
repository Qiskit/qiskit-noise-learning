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

from itertools import product

import numpy as np
import pytest
from qiskit.quantum_info import (
    Clifford,
    Pauli,
    PhasedQubitSparsePauli,
    QubitSparsePauli,
    QubitSparsePauliList,
)

from qiskit_noise_learning.sequences import PartialPauliPermutation
from qiskit_noise_learning.sequences.partial_pauli_permutation import (
    COMPLETE_TO_C1_TABLEAU,
    clifford_1q_inverse_representations,
    clifford_1q_representations,
    compose_matrix,
    full_set_map,
    inverse_vector,
    num_partial_permutations,
)


def test_convention_identity():
    """Tests that the completion of anything consistent with the identity is the identity."""
    for pauli_str in ["Z", "X", "Y"]:
        assert PartialPauliPermutation.from_sets(
            [{(pauli_str, pauli_str)}]
        ).complete() == PartialPauliPermutation([0])

    assert PartialPauliPermutation.empty(1).complete() == PartialPauliPermutation([0])
    assert PartialPauliPermutation([0]).complete() == PartialPauliPermutation([0])


def test_convention_inverse_complete_commutation():
    """Test that inversion and completion are commutative operations."""
    for idx in range(num_partial_permutations()):
        permutation = PartialPauliPermutation([idx])
        assert permutation.inverse.complete() == permutation.complete().inverse


def test_convention_compose_complete_commutation():
    """Test that composition and completion are commutative operations."""
    composition_matrix = compose_matrix()
    for idx0, idx1 in product(range(num_partial_permutations()), range(num_partial_permutations())):
        if composition_matrix[idx0, idx1] > 0:
            perm0 = PartialPauliPermutation([idx0])
            perm1 = PartialPauliPermutation([idx1])
            assert perm0.compose(perm1).complete() == perm0.complete().compose(perm1.complete())


def test_convention_clifford_inversion():
    """Test that implied Clifford of inverse is inverse of implied Clifford."""
    for idx in range(6):
        inverse_idx = inverse_vector()[idx]
        assert Clifford(COMPLETE_TO_C1_TABLEAU[idx]).compose(
            Clifford(COMPLETE_TO_C1_TABLEAU[inverse_idx])
        ) == Clifford([[True, False, False], [False, True, False]])


def test_construction():
    """Test construction and attributes."""

    pauli_permutation = PartialPauliPermutation(np.array([0, 1, 2]))

    assert pauli_permutation.num_qubits == 3
    assert (pauli_permutation.partial_permutation_indices == np.array([0, 1, 2])).all()
    assert pauli_permutation.is_complete


def test_empty():
    """Test function constructing empty partial permutations."""
    empty = PartialPauliPermutation.empty(4)
    expected = PartialPauliPermutation.from_sets([set() for _ in range(4)])
    assert empty == expected


def test_from_sets():
    """Test set construction."""

    permutation_sets = [{("X", "Y"), ("Y", "Z")}, set(), {("Z", "Y")}]
    expected = PartialPauliPermutation([full_set_map()[frozenset(s)] for s in permutation_sets])

    assert PartialPauliPermutation.from_sets(permutation_sets) == expected


def test_to_sets():
    """Test to_sets."""

    # the class will complete the two-entry permutation
    permutation_sets = [{("X", "Y"), ("Y", "Z")}, set(), {("Z", "Y")}]
    expected = [{("X", "Y"), ("Y", "Z"), ("Z", "X")}, set(), {("Z", "Y")}]
    assert PartialPauliPermutation.from_sets(permutation_sets).to_sets() == expected


def test_from_qubit_sparse_paulis():
    """Test construction from QubitSparsePauli instances."""

    partial_permutation = PartialPauliPermutation.from_qubit_sparse_paulis(
        in_pauli=QubitSparsePauli("IXY"), out_pauli=QubitSparsePauli("IXZ")
    )
    assert partial_permutation.num_qubits == 3
    assert partial_permutation == PartialPauliPermutation.from_sets(
        [{("Y", "Z")}, {("X", "X")}, set()]
    )

    # num_qubits error
    with pytest.raises(ValueError, match="in_pauli and out_pauli have mismatching qubit numbers"):
        PartialPauliPermutation.from_qubit_sparse_paulis(
            in_pauli=QubitSparsePauli("IY"), out_pauli=QubitSparsePauli("IXZ")
        )

    # indices error
    with pytest.raises(ValueError, match="in_pauli.indices and out_pauli.indices must match"):
        PartialPauliPermutation.from_qubit_sparse_paulis(
            in_pauli=QubitSparsePauli("YXI"), out_pauli=QubitSparsePauli("IXZ")
        )


def test_from_qubit_sparse_pauli_list():
    """Test construction from QubitSparsePauliList instances."""

    partial_permutation = PartialPauliPermutation.from_qubit_sparse_pauli_lists(
        in_paulis=QubitSparsePauliList(["IXY", "ZYI"]),
        out_paulis=QubitSparsePauliList(["IXZ", "XZI"]),
    )
    assert partial_permutation.num_qubits == 3
    assert partial_permutation == PartialPauliPermutation.from_sets(
        [{("Y", "Z")}, {("X", "X"), ("Y", "Z")}, {("Z", "X")}]
    )

    # num_qubits error
    with pytest.raises(ValueError, match="in_paulis and out_paulis have mismatching qubit numbers"):
        PartialPauliPermutation.from_qubit_sparse_pauli_lists(
            in_paulis=QubitSparsePauliList(["XY", "YI"]),
            out_paulis=QubitSparsePauliList(["IXZ", "XZI"]),
        )

    # indices error
    with pytest.raises(ValueError, match="in_pauli.indices and out_pauli.indices must match"):
        PartialPauliPermutation.from_qubit_sparse_pauli_lists(
            in_paulis=QubitSparsePauliList(["XIY", "ZYI"]),
            out_paulis=QubitSparsePauliList(["IXZ", "XZI"]),
        )


def test_is_mergeable_with():
    """Test mergeability checking."""

    # test when mergable
    partial_permutation0 = PartialPauliPermutation.from_qubit_sparse_paulis(
        in_pauli=QubitSparsePauli("IXY"), out_pauli=QubitSparsePauli("IXZ")
    )
    partial_permutation1 = PartialPauliPermutation.from_qubit_sparse_paulis(
        in_pauli=QubitSparsePauli("IZY"), out_pauli=QubitSparsePauli("IYZ")
    )
    assert partial_permutation0.is_mergeable_with(partial_permutation1)

    # test when not mergable due to Paulis
    partial_permutation2 = PartialPauliPermutation.from_qubit_sparse_paulis(
        in_pauli=QubitSparsePauli("IZY"), out_pauli=QubitSparsePauli("IXZ")
    )
    assert not partial_permutation2.is_mergeable_with(partial_permutation1)

    # test when not mergable due to qubit counts
    assert not partial_permutation0.is_mergeable_with(
        PartialPauliPermutation.from_qubit_sparse_paulis(
            in_pauli=QubitSparsePauli("IX"), out_pauli=QubitSparsePauli("IX")
        )
    )


def test_merge():
    """Test merging of two partial permutations."""

    partial_permutation0 = PartialPauliPermutation.from_qubit_sparse_paulis(
        in_pauli=QubitSparsePauli("IXY"), out_pauli=QubitSparsePauli("IXZ")
    )
    partial_permutation1 = PartialPauliPermutation.from_qubit_sparse_paulis(
        in_pauli=QubitSparsePauli("IZY"), out_pauli=QubitSparsePauli("IYZ")
    )
    out = partial_permutation0.merge(partial_permutation1)

    assert out.num_qubits == 3
    assert out == PartialPauliPermutation.from_sets([{("Y", "Z")}, {("X", "X"), ("Z", "Y")}, set()])


def test_merge_errors():
    """Test merging of incompatible partial permutations."""

    with pytest.raises(
        ValueError, match="Cannot merge permutations with different numbers of qubits."
    ):
        PartialPauliPermutation.from_sets([set(), set()]).merge(
            PartialPauliPermutation.from_sets([set()])
        )

    with pytest.raises(ValueError, match="Cannot merge inconsistent partial permutations."):
        PartialPauliPermutation.from_sets([{("X", "X")}]).merge(
            PartialPauliPermutation.from_sets([{("X", "Y")}])
        )


def test_complete():
    """Test completion."""

    # test identity-consistent permtuations are mapped to identity
    expected = PartialPauliPermutation.from_sets([{("Z", "Z"), ("X", "X"), ("Y", "Y")}])
    for P in ["Z", "X", "Y"]:
        assert PartialPauliPermutation.from_sets([{(P, P)}]).complete() == expected

    # test multiqubit example
    partial_partial_permutation = PartialPauliPermutation.from_sets([{("Z", "X")}, {("X", "Y")}])
    completion = partial_partial_permutation.complete()
    assert completion.is_complete
    assert partial_partial_permutation.is_mergeable_with(completion)


def test_propagate():
    paulis, signs = clifford_1q_representations()
    pauli_str = ["Z", "X", "Y"]

    pauli = PhasedQubitSparsePauli("XY")
    out = PartialPauliPermutation([2, 5]).propagate(pauli)
    sign = "-" if signs[5, 1] * signs[2, 2] < 0 else ""
    assert out == PhasedQubitSparsePauli(
        Pauli(sign + pauli_str[paulis[5, 1]] + pauli_str[paulis[2, 2]])
    )

    # phaseless test
    pauli = QubitSparsePauli("XY")
    out = PartialPauliPermutation([2, 5]).propagate(pauli)
    assert out == QubitSparsePauli(pauli_str[paulis[5, 1]] + pauli_str[paulis[2, 2]])

    # check imaginary phases handled correctly
    pauli = PhasedQubitSparsePauli(Pauli("iXY"))
    out = PartialPauliPermutation([2, 5]).propagate(pauli)
    sign = "-i" if signs[5, 1] * signs[2, 2] < 0 else "i"
    assert out == PhasedQubitSparsePauli(
        Pauli(sign + pauli_str[paulis[5, 1]] + pauli_str[paulis[2, 2]])
    )

    # test with inverses
    paulis, signs = clifford_1q_inverse_representations()
    pauli_str = ["Z", "X", "Y"]

    pauli = PhasedQubitSparsePauli("XY")
    out = PartialPauliPermutation([2, 5]).propagate(pauli, inverse=True)
    sign = "-" if signs[5, 1] * signs[2, 2] < 0 else ""
    assert out == PhasedQubitSparsePauli(
        Pauli(sign + pauli_str[paulis[5, 1]] + pauli_str[paulis[2, 2]])
    )

    # phaseless test
    pauli = QubitSparsePauli("XY")
    out = PartialPauliPermutation([2, 5]).propagate(pauli, inverse=True)
    assert out == QubitSparsePauli(pauli_str[paulis[5, 1]] + pauli_str[paulis[2, 2]])

    # check imaginary phases handled correctly
    pauli = PhasedQubitSparsePauli(Pauli("iXY"))
    out = PartialPauliPermutation([2, 5]).propagate(pauli, inverse=True)
    sign = "-i" if signs[5, 1] * signs[2, 2] < 0 else "i"
    assert out == PhasedQubitSparsePauli(
        Pauli(sign + pauli_str[paulis[5, 1]] + pauli_str[paulis[2, 2]])
    )

    # test partial propagation of QubitSparsePauli
    pauli = QubitSparsePauli("XY")
    out = PartialPauliPermutation.from_sets([{("Y", "Z")}, {("X", "Z")}]).propagate(pauli)
    assert out == QubitSparsePauli("ZZ")


def test_propagate_errors():
    with pytest.raises(
        ValueError,
        match="PartialPauliPermutation is undefined on given pauli.",
    ):
        PartialPauliPermutation.empty(2).propagate(QubitSparsePauli("XY"))

    with pytest.raises(
        ValueError, match="pauli must have the same number of qubits as the permutation."
    ):
        PartialPauliPermutation([0]).propagate(QubitSparsePauli("XY"))

    with pytest.raises(ValueError, match="only be called on complete permutations"):
        PartialPauliPermutation([11, 12]).propagate(PhasedQubitSparsePauli("XY"))


def test_inverse():
    permutation = PartialPauliPermutation.from_sets([{("X", "Z")}, {("Z", "X"), ("Y", "Z")}, {}])
    assert permutation.inverse == PartialPauliPermutation.from_sets(
        [{("Z", "X")}, {("X", "Z"), ("Z", "Y")}, {}]
    )


def test_compose():
    perm0 = PartialPauliPermutation.from_sets([{("X", "Z")}])
    perm1 = PartialPauliPermutation.from_sets([{("Z", "X")}])
    assert perm0.compose(perm1) == PartialPauliPermutation.from_sets([{("Z", "Z")}])

    perm0 = PartialPauliPermutation.from_sets([{("X", "Z")}]).complete()
    perm1 = PartialPauliPermutation.from_sets([{("Z", "X")}]).complete()
    assert perm0.compose(perm1) == PartialPauliPermutation([0])

    perm0 = PartialPauliPermutation.from_sets([{("X", "Z"), ("Z", "Y")}])
    perm1 = PartialPauliPermutation.from_sets([{("Z", "X"), ("X", "Z")}])
    assert perm0.compose(perm1) == PartialPauliPermutation.from_sets([{("Z", "Z"), ("X", "Y")}])


def test_compose_errors():
    # incomplete composed with complete
    perm0 = PartialPauliPermutation.from_sets([{("X", "Z")}])
    perm1 = PartialPauliPermutation([0])
    with pytest.raises(ValueError, match="Cannot compose incompatible"):
        perm1.compose(perm0)

    # both incomplete but incompatible
    perm0 = PartialPauliPermutation.from_sets([{("X", "Z")}])
    perm1 = PartialPauliPermutation.from_sets([{("Y", "Z")}])
    with pytest.raises(ValueError, match="Cannot compose incompatible"):
        perm1.compose(perm0)
