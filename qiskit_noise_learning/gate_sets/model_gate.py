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

"""ModelGate"""

from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import Self, overload

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import (
    Clifford,
    Pauli,
    PhasedQubitSparsePauli,
    QubitSparsePauli,
    SparseObservable,
)

from qiskit_noise_learning.gate_sets import Gate


class ModelGate(Gate):
    """A model for a gate of the form Clifford - MCM - reset.

    Args:
        cliffords: An iterable of tuples of physical qubit indices and :class:`Clifford`s
            corresponding to the ideal Clifford layer. The order of the iterable should start with
            the first applied Clifford and proceed temporally, and ``None`` is interpreted as the
            identity.
        qubit_idxs: The physical qubit indices.
        meas_idxs: The physical qubit indices that this gate measures.
        prep_idxs: The physical qubit indices that this gate prepares, or resets.

    Raises:
        ValueError: If both ``cliffords`` and ``qubit_idxs`` are not specified.
        ValueError: If any of the elements of ``cliffords`` have mismatched length of qubit indices
            and number of qubits of the Clifford.
    """

    def __init__(
        self,
        name: str,
        cliffords: Iterable[tuple[tuple[int, ...], Clifford]] | None = None,
        qubit_idxs: Iterable[int] | None = None,
        meas_idxs: Iterable[int] = (),
        prep_idxs: Iterable[int] = (),
    ):
        if cliffords is None and qubit_idxs is None:
            raise ValueError("At least one of 'cliffords' or 'qubit_idxs' must be specified.")

        clifford_qubit_idxs = set()
        self._cliffords = []
        for idxs, clifford in cliffords if cliffords is not None else []:
            if len(idxs) != clifford.num_qubits:
                raise ValueError(
                    f"Encountered a '{clifford.num_qubits}' qubit Clifford acting on "
                    f"'{idxs}' qubits."
                )
            self._cliffords.append((idxs, clifford))
            clifford_qubit_idxs.update(idx for idx in idxs)

        qubit_idxs = qubit_idxs if qubit_idxs is not None else list(clifford_qubit_idxs)
        if not clifford_qubit_idxs.issubset(qubit_idxs):
            raise ValueError("The qubit indices do not contain all the Clifford indices.")

        super().__init__(name=name, qubit_idxs=qubit_idxs, prep_idxs=prep_idxs, meas_idxs=meas_idxs)

    @cached_property
    def clifford(self) -> Clifford:
        """The overall Clifford operation for this gate."""
        this_clifford = Clifford(QuantumCircuit(max(self.qubit_idxs) + 1))
        for idxs, clifford in self._cliffords:
            this_clifford = this_clifford.compose(clifford, qargs=idxs)
        return this_clifford

    @property
    def cliffords(self) -> list[tuple[tuple[int, ...], Clifford]]:
        """A list of tuples of qubit indices and a Clifford that acts on them.

        The list is in temporal order. Each element contains the qubit indices that the
        corresponding Clifford acts on."""
        return self._cliffords

    @property
    def model_gate(self) -> Self:
        return self

    @property
    def constituent_gate_idxs(self) -> Iterator[tuple[int, ...]]:
        for idxs, _ in self._cliffords:
            yield idxs

    @overload
    def clifford_propagate(
        self, pauli: QubitSparsePauli, inverse: bool = False
    ) -> QubitSparsePauli: ...

    @overload
    def clifford_propagate(
        self, pauli: PhasedQubitSparsePauli, inverse: bool = False
    ) -> PhasedQubitSparsePauli: ...

    def clifford_propagate(self, pauli, inverse=False):
        r"""Given a Pauli, propagate it through the ideal Clifford operation.

        If ``inverse == False`` (the default), then this method returns :math:`C P C^\dagger`, and
        otherwise returns :math:`C^\dagger P C`.

        If the input is a phaseless Pauli type, the propagation will be phaseless.

        Note that the Clifford will be applied to on ``self.qubit_idxs``.

        Args:
            pauli: The Pauli to propagate.
            inverse: Whether to apply the gate or its inverse.

        Returns:
            The propagated result, which has the same type as the input.

        Raises:
            TypeError: If invalid type supplied.
        """
        # this is a hack
        if isinstance(pauli, QubitSparsePauli):
            pauli_str, indices = pauli.to_qubit_sparse_pauli_list().to_sparse_list()[0]
            dense_pauli = SparseObservable.from_sparse_list(
                [(pauli_str, indices, 1)], num_qubits=pauli.num_qubits
            ).pauli_bases()[0]

            frame = "h" if inverse else "s"
            pauli_indices = set(pauli.indices)

            for qubit_idxs, clifford in self.cliffords:
                if not pauli_indices.intersection(qubit_idxs):
                    continue
                dense_pauli = dense_pauli.evolve(clifford, qargs=qubit_idxs, frame=frame)
                pauli_indices = {
                    idx for idx, p in enumerate(dense_pauli.to_label()[::-1]) if p != "I"
                }

            return QubitSparsePauli(dense_pauli.to_label().replace("-", "").replace("i", ""))
        elif isinstance(pauli, PhasedQubitSparsePauli):
            phased_idx, pauli_str, indices = (
                pauli.to_phased_qubit_sparse_pauli_list().to_sparse_list()[0]
            )
            dense_pauli = SparseObservable.from_sparse_list(
                [(pauli_str, indices, (-1j) ** phased_idx)], num_qubits=pauli.num_qubits
            ).pauli_bases()[0]

            frame = "h" if inverse else "s"
            pauli_indices = set(pauli.indices)

            for qubit_idxs, clifford in self.cliffords:
                if not pauli_indices.intersection(qubit_idxs):
                    continue
                dense_pauli = dense_pauli.evolve(clifford, qargs=qubit_idxs, frame=frame)
                pauli_indices = {
                    idx for idx, p in enumerate(dense_pauli.to_label()[::-1]) if p != "I"
                }

            return PhasedQubitSparsePauli(Pauli(dense_pauli.to_label()))
        else:
            raise TypeError("Invalid type for ModelGate.clifford_propagate.")

    def __eq__(self, other):
        if not (
            isinstance(other, ModelGate)
            and self.name == other.name
            and self.meas_idxs == other.meas_idxs
            and self.prep_idxs == other.prep_idxs
            # unordered comparison of qubit_idxs as we'll be testing permuted clifford equality
            and set(self.qubit_idxs) == set(other.qubit_idxs)
        ):
            return False

        return self.cliffords == other.cliffords

    def __hash__(self) -> int:
        if not hasattr(self, "_hash"):
            self._hash = hash(f"ModelGate({self.name})")
        return self._hash
