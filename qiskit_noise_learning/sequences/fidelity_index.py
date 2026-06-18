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

"""FidelityIndex"""

from collections.abc import Container
from functools import cached_property
from typing import Self

import numpy as np
from qiskit.quantum_info import PhasedQubitSparsePauli, QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate


class FidelityIndex:
    r"""Index data for a fidelity in a Pauli-MCM-reset gate set.

    Let :math:`K` be the number of qubits, :math:`[K] = {0, ..., K-1}`, :math:`M\subset [K]` denote
    the measured qubits, and :math:`R \subset [K]` the reset qubits for the gate. For a given gate,
    each fidelity is indexed by:
    - A Pauli on the unmeasured and unreset qubits :math:`Q in P^{[K]\setminus (M \cup R)}`,
    - A list of "input bits" on the measured qubits :math:`x in Z_2^M`, and
    - A list of "output bits" on the measured and reset qubits :math:`y in Z_2^{M \cup R}`.

    In addition to the above data uniquely specifying a fidelity, this class has several derived
    properties phrased in terms of the Pauli operator transition implied by the index data. While
    these properties can be computed from the index data and the properties of the gate, this class
    is setup primarily as a data class, with all required data specified as arguments to the
    :meth:`FidelityIndex.__init__`, which is viewed as a "low-level" constructor without validation.
    The constructors :meth:`FidelityIndex.from_gate` and :meth:`FidelityIndex.from_transition` offer
    higher-level and easier to use construction methods.

    Args:
        gate_name: The name of the gate.
        pauli: A Pauli operator with support on unmeasured and unreset qubits. Note that
            ``pauli.num_qubits`` controls the size of the operators returned by ``self.transition``.
        in_bit_indices: The qubit indices of the non-zero "input bits".
        out_bit_indices: The qubit indices of the non-zero "output bits".
        input_pauli: The input Pauli of the implied transition.
        output_pauli: The output Pauli of the implied transition.
        sign_flip: Whether the implied transition involves a sign flip.
        meas_idxs: The measurement qubit indices for the gate.
    """

    def __init__(
        self,
        gate_name: str,
        pauli: QubitSparsePauli,
        in_bit_indices: frozenset[int],
        out_bit_indices: frozenset[int],
        input_pauli: QubitSparsePauli,
        output_pauli: QubitSparsePauli,
        sign_flip: bool,
        meas_idxs: frozenset[int],
    ):
        self._gate_name = gate_name
        self._pauli = pauli
        self._in_bit_indices = in_bit_indices
        self._out_bit_indices = out_bit_indices
        self._input_pauli = input_pauli
        self._output_pauli = output_pauli
        self._sign_flip = sign_flip
        self._meas_idxs = meas_idxs

    @classmethod
    def from_gate(
        cls,
        gate: ModelGate,
        pauli: QubitSparsePauli,
        in_bit_indices: frozenset[int] = frozenset(),
        out_bit_indices: frozenset[int] = frozenset(),
    ) -> Self:
        """Construct a fidelity index from a gate and unique index data.

        This constructor includes validation of the index data against the :class:`ModelGate`
        properties.

        Args:
            gate: The model gate.
            pauli: A Pauli operator with support on unmeasured and unreset qubits.
            in_bit_indices: The qubit indices of the non-zero "input bits".
            out_bit_indices: The qubit indices of the non-zero "output bits".

        Raises:
            ValueError: If the provided data is inconsistent with the gate.
        """
        meas_and_prep_qubits = gate.meas_idxs.union(gate.prep_idxs)

        if not frozenset(pauli.indices).issubset(
            frozenset(gate.qubit_idxs).difference(meas_and_prep_qubits)
        ):
            raise ValueError(
                "pauli.indices must lie within the unreset and unmeasured qubits of gate."
            )

        if not in_bit_indices.issubset(gate.meas_idxs):
            raise ValueError("in_bit_indices must be a subset of gate.meas_idxs")

        if not out_bit_indices.issubset(meas_and_prep_qubits):
            raise ValueError(
                "out_bit_indices must be a subset of gate.meas_idxs.union(gate.prep_idxs)"
            )

        input_pauli, output_pauli, sign_flip = _compute_transition(
            gate, pauli, in_bit_indices, out_bit_indices
        )

        return cls(
            gate_name=gate.name,
            pauli=pauli,
            in_bit_indices=in_bit_indices,
            out_bit_indices=out_bit_indices,
            input_pauli=input_pauli,
            output_pauli=output_pauli,
            sign_flip=sign_flip,
            meas_idxs=gate.meas_idxs,
        )

    @classmethod
    def from_transition(
        cls, gate: ModelGate, in_pauli: QubitSparsePauli, out_pauli: QubitSparsePauli
    ) -> Self:
        """Construct a fidelity index from a Pauli transition on the quantum registers.

        This constructor deduces the Pauli and bit indices of a :class:`FidelityIndex` from the
        given Pauli transition.

        Args:
            gate: The model gate.
            in_pauli: The input Pauli on the quantum register.
            out_pauli: The output Pauli on the quantum register.

        Raises:
            ValueError: If the pair of Pauli operators do not imply a valid :class:`FidelityIndex`.
        """

        if not set(in_pauli.indices).issubset(gate.qubit_idxs):
            raise ValueError("in_pauli.indices is not contained in gate.qubit_idxs.")
        elif not set(out_pauli.indices).issubset(gate.qubit_idxs):
            raise ValueError("out_pauli.indices is not contained in gate.qubit_idxs.")

        prep_meas_idxs = gate.meas_idxs.union(gate.prep_idxs)

        # output restricted to measured and reset qubits should only have I, Z components
        out_Z = _restrict_pauli(out_pauli, prep_meas_idxs)
        if not (out_Z.paulis == 1).all():
            raise ValueError(
                "out_pauli restricted to measured and reset qubits must only have I and Z "
                "components."
            )
        out_bit_indices = frozenset(int(x) for x in out_Z.indices)

        mapped_input = gate.clifford_propagate(pauli=in_pauli, inverse=False)

        # mapped_input restricted to measured qubits must only have I, Z components
        in_Z = _restrict_pauli(mapped_input, gate.meas_idxs)
        if not (in_Z.paulis == 1).all():
            raise ValueError(
                "in_pauli mapped by Clifford and restricted to measured qubits must only have I "
                "and Z components."
            )
        in_bit_indices = frozenset(int(x) for x in in_Z.indices)

        # mapped_input restricted to unmeasured and reset qubits should be identity
        if (
            len(
                _restrict_pauli(
                    mapped_input,
                    set(gate.qubit_idxs).difference(gate.meas_idxs).intersection(gate.prep_idxs),
                ).paulis
            )
            > 0
        ):
            raise ValueError(
                "in_pauli mapped by Clifford and restricted to unmeasured and reset qubits must be "
                "the identity."
            )

        # out_pauli and mapped_input must agree on unmeasured and unreset qubits
        unmeas_unprep_idxs = set(gate.qubit_idxs).difference(prep_meas_idxs)
        pauli = _restrict_pauli(out_pauli, unmeas_unprep_idxs)
        if pauli != _restrict_pauli(mapped_input, unmeas_unprep_idxs):
            raise ValueError(
                "out_pauli and in_pauli mapped by Clifford must agree on unmeasured and unreset "
                "qubits"
            )

        input_pauli, output_pauli, sign_flip = _compute_transition(
            gate, pauli, in_bit_indices, out_bit_indices
        )

        return cls(
            gate_name=gate.name,
            pauli=pauli,
            in_bit_indices=in_bit_indices,
            out_bit_indices=out_bit_indices,
            input_pauli=input_pauli,
            output_pauli=output_pauli,
            sign_flip=sign_flip,
            meas_idxs=gate.meas_idxs,
        )

    @property
    def gate_name(self) -> str:
        """The name of the gate."""
        return self._gate_name

    @property
    def pauli(self) -> QubitSparsePauli:
        """The Pauli operator on the Clifford portion of the model gate."""
        return self._pauli

    @property
    def in_bit_indices(self) -> frozenset[int]:
        """The input bits corresponding to the measurement indices."""
        return self._in_bit_indices

    @property
    def out_bit_indices(self) -> frozenset[int]:
        """The output bits corresponding to the reset indices."""
        return self._out_bit_indices

    @property
    def sign_flip(self) -> bool:
        """Whether the transition associated with this fidelity involves a sign flip."""
        return self._sign_flip

    @property
    def transition(self) -> tuple[QubitSparsePauli, QubitSparsePauli]:
        """The phaseless Pauli operator transition associated with this fidelity index."""
        return self._input_pauli, self._output_pauli

    @cached_property
    def mask(self) -> np.ndarray[np.bool_]:
        """The mask for marginalizing measurement outcomes."""
        sorted_meas_idxs = sorted(self._meas_idxs)
        mask = np.zeros(len(self._meas_idxs), dtype=np.bool_)
        mask[
            [
                idx
                for idx, meas_idx in enumerate(sorted_meas_idxs)
                if meas_idx in self.observable_indices
            ]
        ] = True
        return mask

    @cached_property
    def observable_indices(self) -> list[int]:
        """Qubit indices of the associated Z observable in ascending order."""
        return sorted(
            self.out_bit_indices.intersection(self._meas_idxs).symmetric_difference(
                self.in_bit_indices
            )
        )

    def __eq__(self, other: "FidelityIndex") -> bool:
        return (
            isinstance(other, FidelityIndex)
            and self.gate_name == other.gate_name
            and self.pauli == other.pauli
            and self.in_bit_indices == other.in_bit_indices
            and self.out_bit_indices == other.out_bit_indices
        )

    def __hash__(self) -> int:
        if not hasattr(self, "_hash"):
            self._hash = hash(
                (
                    self.gate_name,
                    (tuple(self.pauli.paulis), tuple(self.pauli.indices), self.pauli.num_qubits),
                    self.in_bit_indices,
                    self.out_bit_indices,
                )
            )
        return self._hash

    def __repr__(self) -> str:
        s = "FidelityIndex(\n"
        s += f"    gate_name='{self.gate_name}',\n"
        s += f"    pauli={self.pauli},\n"
        s += f"    in_bit_indices={self.in_bit_indices},\n"
        s += f"    out_bit_indices={self.out_bit_indices},\n"
        s += ")"
        return s


def _compute_transition(
    gate: ModelGate,
    pauli: QubitSparsePauli,
    in_bit_indices: frozenset[int],
    out_bit_indices: frozenset[int],
) -> tuple[QubitSparsePauli, QubitSparsePauli, bool]:
    """Compute the transition Paulis and sign flip for a fidelity index."""
    phased_input_pauli = gate.clifford_propagate(
        pauli=PhasedQubitSparsePauli(pauli.to_pauli())
        @ PhasedQubitSparsePauli.from_sparse_label(
            (0, "Z" * len(in_bit_indices), list(in_bit_indices)),
            num_qubits=pauli.num_qubits,
        ),
        inverse=True,
    )
    input_pauli = QubitSparsePauli.from_raw_parts(
        num_qubits=phased_input_pauli.num_qubits,
        paulis=phased_input_pauli.paulis,
        indices=phased_input_pauli.indices,
    )
    sign_flip = phased_input_pauli.phase == 2

    output_pauli = pauli @ QubitSparsePauli.from_sparse_label(
        ("Z" * len(out_bit_indices), list(out_bit_indices)),
        num_qubits=pauli.num_qubits,
    )

    return input_pauli, output_pauli, sign_flip


def _restrict_pauli(pauli: QubitSparsePauli, indices: Container[int]) -> QubitSparsePauli:
    """Restrict pauli to the qubits specified in indices."""

    new_pauli_idxs = []
    new_qubit_idxs = []
    for pauli_idx, qubit_idx in zip(pauli.paulis, pauli.indices):
        if qubit_idx in indices:
            new_qubit_idxs.append(qubit_idx)
            new_pauli_idxs.append(pauli_idx)

    return QubitSparsePauli.from_raw_parts(
        num_qubits=pauli.num_qubits, paulis=new_pauli_idxs, indices=new_qubit_idxs
    )
