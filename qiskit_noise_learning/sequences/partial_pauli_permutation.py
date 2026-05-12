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


"""PartialPauliPermutation"""

import functools
from itertools import combinations, product
from types import MappingProxyType
from typing import Self, overload

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import (
    Clifford,
    Pauli,
    PhasedQubitSparsePauli,
    QubitSparsePauli,
    QubitSparsePauliList,
)

from .instruction import Instruction

pauli_strings = ["Z", "X", "Y"]


COMPLETE_TO_C1_TABLEAU = np.array(
    [
        [[True, False, False], [False, True, False]],
        [[False, True, True], [True, False, True]],
        [[True, True, False], [True, False, False]],
        [[True, False, True], [True, True, True]],
        [[False, True, False], [True, True, False]],
        [[True, True, True], [False, True, True]],
    ],
    np.bool_,
)
"""An array containing a choice of single-qubit Clifford tableau for each complete permutation.

Ignoring signs, the Cliffords correspond to :math:`I`, :math:`H`, :math:`HS`, :math:`HSH`,
:math:`HSHS`, and :math:`HSHSH`. The specific tableau for each permutation is chosen so that this
set is a subgroup.
"""


@functools.lru_cache
def partial_permutation_sets() -> tuple[frozenset[tuple[str, str]]]:
    """Return all possible single-qubit partial pauli permutations.

    The set format for a partial permutation is a set of elements of the form ``(p0, p1)``, where
    ``p0`` and ``p1`` are drawn from ``["Z", "X", "Y"]``, and the pair ``(p0, p1)`` means ``p0``
    is mapped to ``p1``.

    The ordering given here is fixed, enabling ``PartialPauliPermutation`` to internally store a
    list of indices into this list. Furthermore, some functionality depends on the specifics of the
    ordering. For example, the three-element/complete permutations are the first 6 elements of the
    list, and as such checking for completeness can be done by checking if the index is < 6. The
    ordering of the 6 fully specified permutations are based on the ordering of the blocks in
    ``COMPLETE_TO_C1_TABLEAU``.

    Note that the list returned by this function contains only the empty, single element, or three
    element partial permutations. The two element partial permutations are omitted as they are
    equivalent to three element permutation (as the set of non-identity single-qubit Paulis only has
    3 elements).
    """

    def tableau_to_permutation_set(tableau):
        cliff = Clifford(tableau)
        return frozenset(
            (Pauli(pauli_str).to_label(), Pauli(pauli_str).evolve(cliff, frame="s").to_label()[-1])
            for pauli_str in pauli_strings
        )

    complete_permutations_sets = [tableau_to_permutation_set(x) for x in COMPLETE_TO_C1_TABLEAU]
    single_mappings_sets = [frozenset([(a, b)]) for (a, b) in product(pauli_strings, pauli_strings)]
    return tuple(complete_permutations_sets + single_mappings_sets + [frozenset()])


@functools.lru_cache
def num_partial_permutations() -> int:
    """Return the total number of partial permutations.

    In other words, return the length of :func:`~.partial_permutation_sets`.
    """
    return len(partial_permutation_sets())


@functools.lru_cache
def partial_pauli_maps() -> np.ndarray[int]:
    """Returns a representation of phaseless partial mappings on ["Z", "X", "Y"]."""

    pauli_maps = -1 * np.ones((num_partial_permutations(), 3), dtype=np.int8)

    for perm_idx, permutation_set in enumerate(partial_permutation_sets()):
        for in_pauli, out_pauli in permutation_set:
            pauli_maps[perm_idx, pauli_strings.index(in_pauli)] = pauli_strings.index(out_pauli)
    return pauli_maps


@functools.lru_cache
def full_set_map() -> MappingProxyType[frozenset[tuple[str, str]], int]:
    """Return a map from possible partial-permutation sets to their index.

    Indices are defined by the ordering of :func:`~.partial_permutation_sets`. However,
    in addition to the sets in :func:`~.partial_permutation_sets`, which contain either no, one, or
    three elements, this mapping contains two-element sets, which are mapped to the index of their
    three-element completions.
    """
    # initialize witht he existing ordering for non-two-element sets
    set_map = {frozenset(s): idx for idx, s in enumerate(partial_permutation_sets())}

    # add the two-element sets
    for in_paulis, unordered_out_paulis in product(combinations(pauli_strings, 2), repeat=2):
        for out_paulis in [
            (unordered_out_paulis[0], unordered_out_paulis[1]),
            (unordered_out_paulis[1], unordered_out_paulis[0]),
        ]:
            new_permutation_set = set()
            for in_pauli, out_pauli in zip(in_paulis, out_paulis):
                new_permutation_set.add((in_pauli, out_pauli))

            for idx, permutation_set in enumerate(partial_permutation_sets()):
                if new_permutation_set.issubset(permutation_set):
                    set_map[frozenset(new_permutation_set)] = idx
                    break

    return MappingProxyType(set_map)


@functools.lru_cache
def consistency_matrix() -> np.ndarray[tuple[16, 16], np.dtype[bool]]:
    """Return a truth table specifying pairwise partial permutation consistency.

    Two partial permutations are consistent if there exists a single-qubit Clifford that implements
    both of their permutations, which without loss of generality is just to say that one doesn't map
    a Pauli to a different Pauli than the other.
    For partial permutations specified as indices ``idx0`` and ``idx1``,
    ``consistency_matrix()[idx0, idx1]`` gives whether or not they are consistent.
    """
    consistency_matrix = np.array(
        [[False] * num_partial_permutations()] * num_partial_permutations(), dtype=bool
    )
    # all full permutations are inconsistent with eachother
    consistency_matrix[:6, :6] = np.eye(6) > 0
    # the empty permutation is consistent with everything
    consistency_matrix[-1] = np.array([True] * num_partial_permutations())
    consistency_matrix[:, -1] = np.array([True] * num_partial_permutations())

    # fill out the rest of these matrices, looping through single-entry partial permutations
    for idx0 in range(6, num_partial_permutations() - 1):
        current_perm = partial_permutation_sets()[idx0]
        # first loop over complete permutations
        for idx1 in range(6):
            if current_perm.issubset(partial_permutation_sets()[idx1]):
                consistency_matrix[idx0, idx1] = True
                consistency_matrix[idx1, idx0] = True

        (current_entry,) = current_perm
        for idx1 in range(6, num_partial_permutations() - 1):
            (compare_entry,) = partial_permutation_sets()[idx1]

            # if unequal input and unequal output, can merge
            if (current_entry[0] != compare_entry[0] and current_entry[1] != compare_entry[1]) or (
                current_entry == compare_entry
            ):
                consistency_matrix[idx0, idx1] = True
                consistency_matrix[idx1, idx0] = True

    # make this array read only
    consistency_matrix.setflags(write=False)

    return consistency_matrix


@functools.lru_cache
def merge_matrix() -> np.ndarray[tuple[16, 16], np.dtype[np.int8]]:
    """Return a rule, specified as a matrix, for how to merge two partial permutations.

    Given two permutations specified as indices ``idx0`` and ``idx1``, the index of their merging
    is given by ``merge_matrix()[idx0, idx1]``. If ``idx0`` and ``idx1`` are inconsistent,
    ``merge_matrix()[idx0, idx1]`` returns ``-1``.
    """
    merge_matrix = -1 * np.ones(
        (num_partial_permutations(), num_partial_permutations()), dtype=np.int8
    )
    # merging full permutations with eachother are invalid unless merging a permutation with itself
    merge_matrix[:6, :6] = np.diag(np.arange(1, 7, dtype=np.int8)) - np.ones((6, 6), dtype=np.int8)
    # merging with the empty permutation is the identity
    merge_matrix[-1] = np.arange(num_partial_permutations(), dtype=np.int8)
    merge_matrix[:, -1] = np.arange(num_partial_permutations(), dtype=np.int8)

    for idx0 in range(6, num_partial_permutations() - 1):
        # first loop over complete
        for idx1 in range(0, num_partial_permutations() - 1):
            if consistency_matrix()[idx0, idx1]:
                idx_merged_set = full_set_map()[
                    frozenset(
                        partial_permutation_sets()[idx0].union(partial_permutation_sets()[idx1])
                    )
                ]
                merge_matrix[idx0, idx1] = idx_merged_set
                merge_matrix[idx1, idx0] = idx_merged_set

    # make this array read only
    merge_matrix.setflags(write=False)

    return merge_matrix


@functools.lru_cache
def compose_matrix() -> np.ndarray[np.dtype[np.int8]]:
    """A 2d array encoding the composition rule."""

    composition_matrix = -1 * np.ones(
        (num_partial_permutations(), num_partial_permutations()), dtype=np.int8
    )

    for (idx0, perm_set0), (idx1, perm_set1) in product(
        enumerate(partial_permutation_sets()), enumerate(partial_permutation_sets())
    ):
        if len(perm_set0) != len(perm_set1):
            continue

        # build set of possible composition
        comp_perm_list = []

        for input0, output0 in perm_set0:
            for input1, output1 in perm_set1:
                if output1 == input0:
                    comp_perm_list.append((input1, output0))
                    continue

        if len(comp_perm_list) != len(perm_set0):
            continue

        composition_matrix[idx0, idx1] = full_set_map()[frozenset(comp_perm_list)]

    return composition_matrix


@functools.lru_cache
def inverse_vector() -> np.ndarray[np.dtype[np.int8]]:
    """Return a 1d array encoding the inversion of a partial permutation."""
    permutation_sets = partial_permutation_sets()

    inverse_vector = np.empty(len(permutation_sets), dtype=np.int8)
    for idx, permutation_set in enumerate(permutation_sets):
        inverse_set = frozenset((x[1], x[0]) for x in permutation_set)
        inverse_vector[idx] = full_set_map()[inverse_set]

    return inverse_vector


@functools.lru_cache
def completion_vector() -> np.ndarray[np.dtype[np.int8]]:
    """Return a 1d array encoding the choice of "completion" for each partial permutation index,
    according to the set ordering in ``partial_permutation_sets``.

    A completion of a partial Pauli permutation is a fully-specified permutation that agrees with
    the partial Pauli permutation. For example, the identity permutation is a completion of the
    partial Pauli permutation which only specifies X -> X. Specified in terms of indices for the
    specific ordering given by ``partial_permutation_sets``, this function returns a 1d array
    encoding the specific choices of completion for all partial Pauli permutations, i.e., the
    completion of the partial Permutation specified as a set ``partial_permutation_sets()[idx]``, is
    ``partial_permutation_sets()[completion_vector()[idx]]``.

    The convention used is that a non-complete permutation is mapped to the first complete
    permutation it is consistent with in the list ``partial_permutation_sets``, except if the
    completion of the inverse is already specified, in which case it is set to the inverse of the
    completion of the inverse. An important property is that anything consistent with the identity
    is mapped to the identity.
    """
    permutation_sets = partial_permutation_sets()
    complete_permutation_sets = permutation_sets[:6]
    completion_vector = -1 * np.ones(len(permutation_sets), dtype=np.int8)
    for idx, permutation_set in enumerate(permutation_sets):
        # if completion of inverse is already specified, set the completion to the inverse of that
        inverse_completion_idx = completion_vector[inverse_vector()[idx]]
        if inverse_completion_idx != -1:
            completion_vector[idx] = inverse_vector()[inverse_completion_idx]
            continue

        for complete_idx, complete_permutation_set in enumerate(complete_permutation_sets):
            if permutation_set.issubset(complete_permutation_set):
                completion_vector[idx] = complete_idx
                break

    return completion_vector


@functools.lru_cache
def clifford_1q_representations() -> tuple[np.ndarray[int], np.ndarray[int]]:
    """Return single-qubit Cliffords consistent with each possible partial Pauli permutation.

    Note that the ordering of the Cliffords is given by ``COMPLETE_TO_C1_TABLEAU``, which is
    consistent with the function :func:`partial_permutation_sets`.

    Returns:
        A pair of 2d arrays, ``(paulis, signs)`` which encode, respectively, the permutation and the
        signs. For the Clifford at index ``clifford_idx``, its mapping of the Pauli at ``pauli_idx``
        in the list ``["Z", "X", "Y"]`` is given by ``paulis[clifford_idx, pauli_idx]`` with sign
        ``signs[clifford_idx, pauli_idx]``.
    """

    cliffords = [Clifford(x) for x in COMPLETE_TO_C1_TABLEAU]

    paulis = np.empty((6, 3), dtype=np.uint8)
    signs = np.ones((6, 3), dtype=np.int8)

    for clifford_idx, clifford in enumerate(cliffords):
        for pauli_idx, pauli in enumerate(pauli_strings):
            out_label = Pauli(pauli).evolve(clifford, frame="s").to_label()

            if len(out_label) == 2:
                signs[clifford_idx, pauli_idx] = -1

            paulis[clifford_idx, pauli_idx] = pauli_strings.index(out_label[-1])

    return paulis, signs


@functools.lru_cache
def clifford_1q_inverse_representations() -> tuple[np.ndarray[int], np.ndarray[int]]:
    """The same as :func:`clifford_1q_representions`, except for the inverse of the Clifford."""

    cliffords = [Clifford(x) for x in COMPLETE_TO_C1_TABLEAU]

    paulis = np.empty((6, 3), dtype=np.uint8)
    signs = np.ones((6, 3), dtype=np.int8)

    for clifford_idx, clifford in enumerate(cliffords):
        for pauli_idx, pauli in enumerate(pauli_strings):
            out_label = Pauli(pauli).evolve(clifford, frame="h").to_label()

            if len(out_label) == 2:
                signs[clifford_idx, pauli_idx] = -1

            paulis[clifford_idx, pauli_idx] = pauli_strings.index(out_label[-1])

    return paulis, signs


class PartialPauliPermutation(Instruction):
    r"""Partially-specified permutations of the single-qubit phaseless Paulis on ``n`` qubits.

    A :class:`PartialPauliPermutation` represents a partial-specification of a layer of single qubit
    Cliffords for situations where the specific phases of the Pauli group need not be constrained.
    The partial nature of the specification is to enable progressively building such layers. Once
    a partial permutation is "complete" in the sense that is a full specification of a permutation,
    as indicated by the ``bool`` property :attr:`PartialPauliPermutation.is_complete`, a
    default Clifford implementing the permutation is assigned to each qubit according to the
    ordering in ``COMPLETE_TO_C1_TABLEAU``.

    Two partial permutations on a qubit are mergeable (see :meth:`~is_mergeable_with` and
    :meth:`~merge`\) if there exists a single-qubit Clifford that implements both of their
    permutations, which without loss of generality is the statement that one doesn't map a Pauli to
    a different Pauli than the other.

    The main data representation of the class is a list of integers, where each integer indexes a
    particular single-qubit partial Pauli permutation given in ``partial_permutation_sets()``, which
    provides a fixed ordering.

    However, a human-readable ``set``-based representation can also be used for construction via the
    :meth:`.PartialPauliPermutation.from_sets` class method, or can be retrieved from an instance
    via the :meth:`.PartialPauliPermutation.to_sets`` method. For a single qubit, the partial
    permutation is specified as a ``set`` whose entries are ``tuple``\s of the form ``(p0, p1)``,
    where ``p0`` and ``p1`` are strings drawn from ``["Z", "X", "Y"]``. This ``tuple`` indicates
    that ``p0`` is mapped by the permutation to ``p1``.

    Args:
        partial_permutation_indices: A numpy array of index-specified partial permutations. The
            number of qubits is determined from the length.
    """

    def __init__(self, partial_permutation_indices: NDArray[np.int8]):
        self._partial_permutation_indices = np.array(partial_permutation_indices, dtype=np.int8)

    @property
    def inverse(self) -> Self:
        """Return the inversion of this partial permutation.

        This returns a partially-specified inversion: only the existing mappings in this
        instance will be inverted. Note that the completion convention has been chosen to be
        consistent with inversion, in the sense that
        ``self.inverse.complete() == self.complete().inverse``. Furthermore, the Clifford implied
        by ``COMPLETE_TO_C1_TABLEAU`` is the inverse of the implied Clifford.
        """
        return PartialPauliPermutation(inverse_vector()[self.partial_permutation_indices])

    @property
    def is_complete(self) -> bool:
        """Whether self represents a complete specification of single-qubit Pauli permutations."""
        return (self.partial_permutation_indices < 6).all()

    @property
    def num_qubits(self) -> int:
        return len(self.partial_permutation_indices)

    @property
    def partial_permutation_indices(self) -> NDArray[np.int8]:
        """Raw numerical format of the partial permutation."""
        return self._partial_permutation_indices

    def complete(self) -> Self:
        """Return a new partial Permutation that is complete and consistent with self.

        Note that the conventions have been chosen to ensure that:
        * Any partially-specified permutation consistent with the identity is mapped to the
          identity, and
        * ``self.inverse.complete() == self.complete().inverse``.

        Returns:
            A new :class:`PartialPauliPermutation` containing the completion of ``self``.
        """
        return PartialPauliPermutation(completion_vector()[self.partial_permutation_indices])

    def compose(self, other: Self) -> Self:
        """Compose with another partial permutation.

        For complete permutations, ``self.compose(other)`` returns the permutation assciated with
        ``C1 @ C2``, where ``C1`` and ``C2`` are the Cliffords associated, respectively, with
        ``self`` and ``other``. Partially specified permutations only contain a single mapping, and
        the composition is defined in the natural way only when the output of ``other`` is the input
        of ``self``. Note finally that composition is not defined if one of ``self`` and ``other``
        is incomplete, and the other is complete. This is due to the inability to ensure the
        commutation of completion and composition, described below.

        Note that the completion convention has been chosen to be
        consistent with composition, in the sense that
        ``self.compose(other).complete() == self.complete().compose(other.complete())``.
        Furthermore, for complete permutations, the mapping to the Clifford implied by
        ``COMPLETE_TO_C1_TABLEAU`` is a group homomorphism (preserves multiplication).

        Args:
            other: The other to compose with.

        Returns:
            The composed permutation.

        Raises:
            ValueError: If the composition of ``self`` with ``other`` is undefined.
        """

        if self.num_qubits != other.num_qubits:
            raise ValueError("Cannot compose permutations with different numbers of qubits.")

        new_indices = compose_matrix()[
            self.partial_permutation_indices, other.partial_permutation_indices
        ]

        if (new_indices == -1).any():
            raise ValueError("Cannot compose incompatible partial permutations.")

        return PartialPauliPermutation(new_indices)

    @classmethod
    def empty(cls, num_qubits: int) -> Self:
        """Generate the completely unspecified instance on ``num_qubits``.

        Args:
            num_qubits: Number of qubits.

        Returns:
            A new, trivial :class:`~.PartialPauliPermutation`.
        """
        empty_idx = full_set_map()[frozenset()]
        return cls([empty_idx] * num_qubits)

    @classmethod
    def from_qubit_sparse_paulis(
        cls, in_pauli: QubitSparsePauli, out_pauli: QubitSparsePauli
    ) -> Self:
        """Construct a ``PartialPauliPermutation`` that maps ``in_pauli`` to ``out_pauli``.

        Args:
            in_pauli: The Pauli to be mapped.
            out_pauli: The Pauli to be mapped to.

        Returns:
            A new :class:`~.PartialPauliPermutation` that maps ``in_pauli`` to ``out_pauli``.

        Raises:
            ValueError: If ``in_pauli`` and ``out_pauli`` are not on the same number of qubits, or
                if they do not act on the same qubits.
        """

        if in_pauli.num_qubits != out_pauli.num_qubits:
            raise ValueError("in_pauli and out_pauli have mismatching qubit numbers.")

        if any(in_pauli.indices != out_pauli.indices):
            raise ValueError("in_pauli.indices and out_pauli.indices must match.")

        sets = [set() for _ in range(in_pauli.num_qubits)]
        for idx, p0, p1 in zip(in_pauli.indices, in_pauli.paulis, out_pauli.paulis):
            sets[idx].add((pauli_strings[p0 - 1], pauli_strings[p1 - 1]))

        return cls.from_sets(sets)

    @classmethod
    def from_qubit_sparse_pauli_lists(
        cls, in_paulis: QubitSparsePauliList, out_paulis: QubitSparsePauliList
    ) -> Self:
        """Construct a ``PartialPauliPermutation`` that maps ``in_paulis`` to ``out_paulis``.

        Args:
            in_paulis: The Paulis to be mapped.
            out_paulis: The Paulis to be mapped to.

        Returns:
            A new :class:`~.PartialPauliPermutation`.

        Raises:
            ValueError: If the number of qubits are inconsistent, or the implied permutations are
                inconsistent.
        """

        if in_paulis.num_qubits != out_paulis.num_qubits:
            raise ValueError("in_paulis and out_paulis have mismatching qubit numbers.")

        partial_pauli_permutation = PartialPauliPermutation.empty(num_qubits=in_paulis.num_qubits)

        for in_pauli, out_pauli in zip(in_paulis, out_paulis):
            partial_pauli_permutation = partial_pauli_permutation.merge(
                PartialPauliPermutation.from_qubit_sparse_paulis(in_pauli, out_pauli)
            )

        return partial_pauli_permutation

    @classmethod
    def from_sets(cls, sets: list[frozenset[tuple[str, str]]]) -> Self:
        """Construct from a list of sets.

        See the class documentation for a description of the expected format.

        Args:
            sets: The sets specifying the partial permutation.

        Returns:
            A new instance.

        Raises:
            ValueError: If any of the sets are not valid.
        """

        set_mapping = full_set_map()

        indices = []
        for s in sets:
            try:
                indices.append(set_mapping[frozenset(s)])
            except ValueError:
                raise ValueError(f"{s} is not a valid Partial Pauli Permutation set.")

        return cls(indices)

    def is_mergeable_with(self, other):
        if self.num_qubits != other.num_qubits:
            return False

        return consistency_matrix()[
            self.partial_permutation_indices, other.partial_permutation_indices
        ].all()

    def merge(self, other):
        if self.num_qubits != other.num_qubits:
            raise ValueError("Cannot merge permutations with different numbers of qubits.")

        new_indices = merge_matrix()[
            self.partial_permutation_indices, other.partial_permutation_indices
        ]

        if (new_indices == -1).any():
            raise ValueError("Cannot merge inconsistent partial permutations.")

        return PartialPauliPermutation(new_indices)

    @overload
    def propagate(self, pauli: QubitSparsePauli, inverse: bool = False) -> QubitSparsePauli: ...

    @overload
    def propagate(
        self, pauli: PhasedQubitSparsePauli, inverse: bool = False
    ) -> PhasedQubitSparsePauli: ...

    def propagate(self, pauli, inverse=False):
        """Given a Pauli, propagate it through the Clifford implied by this permutation.

        This method works for both phased and unphased propagation depending on the type of the
        Pauli supplied. Unphased propagation can be performed on incomplete permutations, so long
        as the permutation is defined on the Pauli. Phased propagation requires ``self`` to be
        complete, so that an explicit Clifford can be associated with this instance.

        Args:
            pauli: The Pauli to apply the layer to.

        Returns:
            The evolved Pauli.

        Raises:
            ValueError: If ``pauli.num_qubits != self.num_qubits``, or if
                ``isinstance(pauli, PhasedQubitSparsePauli and not self.is_complete``, or if
                ``pauli`` is unphased and this instance is undefined on it.
            TypeError: If ``pauli`` is an invalid type.
        """

        if inverse:
            return self.inverse.propagate(pauli)

        if pauli.num_qubits != self.num_qubits:
            raise ValueError("pauli must have the same number of qubits as the permutation.")

        if isinstance(pauli, QubitSparsePauli):
            pauli_map = partial_pauli_maps()

            # note that the pauli index convention of sparse Paulis is shifted by +1
            new_paulis = (
                pauli_map[self.partial_permutation_indices[pauli.indices], pauli.paulis - 1] + 1
            )

            if (new_paulis < 1).any():
                raise ValueError("PartialPauliPermutation is undefined on given pauli.")

            return QubitSparsePauli.from_raw_parts(
                num_qubits=pauli.num_qubits, paulis=new_paulis, indices=pauli.indices
            )
        elif isinstance(pauli, PhasedQubitSparsePauli):
            if not self.is_complete:
                raise ValueError(
                    "PartialPauliPermutation.propagate on PhasedQubitSparsePauli can only be "
                    "called on complete permutations."
                )

            pauli_map, signs = clifford_1q_representations()

            # note that the pauli index convention of sparse Paulis is shifted by +1
            new_paulis = (
                pauli_map[self.partial_permutation_indices[pauli.indices], pauli.paulis - 1] + 1
            )

            phase_change = np.prod(
                signs[self.partial_permutation_indices[pauli.indices], pauli.paulis - 1]
            )
            new_phase = pauli.phase + 2 if phase_change == -1 else pauli.phase
            return PhasedQubitSparsePauli.from_raw_parts(
                num_qubits=pauli.num_qubits,
                paulis=new_paulis,
                indices=pauli.indices,
                phase=new_phase,
            )

    def to_sets(self) -> list[frozenset[tuple[str, str]]]:
        """Return the set representation."""
        return [partial_permutation_sets()[index] for index in self.partial_permutation_indices]

    def __eq__(self, other):
        return (
            isinstance(other, PartialPauliPermutation)
            and (self.partial_permutation_indices == other.partial_permutation_indices).all()
        )

    def __repr__(self):
        s = "PartialPauliPermutation(\n"
        s += f"    num_qubits={self.num_qubits}\n"
        for qubit_idx, single_set in enumerate(self.to_sets()):
            if len(single_set) > 0:
                input, output = zip(*single_set, strict=True)
                if input != output:
                    s += f"    {qubit_idx}: "
                    s += ''.join(input) + ' -> ' + ''.join(output) + "\n"
        s += ")"
        return s
