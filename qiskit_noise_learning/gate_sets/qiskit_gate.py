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

"""QiskitGate"""

from collections.abc import Iterable, Iterator, Sequence
from functools import cached_property
from itertools import chain

from qiskit.circuit import (
    Annotation,
    CircuitInstruction,
    Operation,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.quantum_info import Clifford
from samplomatic import Twirl

from .gate import Gate, int_sequence_to_str
from .model_gate import ModelGate


def _is_prep(instr: CircuitInstruction) -> bool:
    # the qubit length check is just an early exit. we check by operation name because some
    # backends may support multiple alternate reset operations
    return len(instr.qubits) == 1 and instr.operation.name.startswith("reset")


def _is_meas(instr: CircuitInstruction) -> bool:
    # the qubit length check is just an early exit. we check by operation name because some
    # backends may support multiple alternate reset operations
    return len(instr.qubits) == 1 and instr.operation.name.startswith("meas")


class QiskitGate(Gate):
    """Represents a single gate in a :class:`~.QiskitGateSet`.

    It is assumed that the gate consists of a sequence of unitary operations, followed by
    measurements, with no qubit being measured more than once, and finally preparations or resets.
    This is not currently validated.

    In many ways, this class is similar to a :class:`~qiskit.circuit.CircuitInstruction` containing
    a :class:`~qiskit.circuit.BoxOp` in that it represents the action on some subset of qubits, and
    possibly with some of those qubits idling. It differs in that the physical qubits are
    represented as integers rather than :class:`~qiskit.circuit.Qubit` objects, and that we
    explicitly store which physical qubit indices perform preparation (or reset) operations.

    This class also implements the equality operation, where two :class:`~.QiskitGate` instances
    are equal whenever their physical indices are equal, and their circuits are equal taking into
    account permutations of the lists ``qubit_idxs`` and ``circuit.qubits``.

    Args:
        name: The name for the gate.
        circuit: The quantum circuit.
        qubit_idxs: The physical qubit indices that ``circuit.qubits`` act on.
        prep_idxs: The physical qubit indices that this gate prepares, or resets. This is included
            because explicitly using the :class:`Reset` instruction is not common.
        annotations: The annotations that describe how to implement the gate. If ``None``, this
            defaults to Pauli twirling.
    """

    def __init__(
        self,
        name: str,
        circuit: QuantumCircuit,
        qubit_idxs: Iterable[int],
        prep_idxs: Iterable[int] = (),
        annotations: Sequence[Annotation] | None = None,
    ):
        meas_idxs = []
        other_preps = []
        qubit_map = dict(zip(circuit.qubits, qubit_idxs))
        for instr in circuit:
            if _is_prep(instr):
                other_preps.extend(map(qubit_map.get, instr.qubits))
            elif _is_meas(instr):
                meas_idxs.extend(map(qubit_map.get, instr.qubits))

        if circuit.num_qubits != len(qubit_idxs):
            raise ValueError("`qubit_idxs` must have a length equal to `circuit.num_qubits`.")

        super().__init__(
            name=name,
            qubit_idxs=qubit_idxs,
            prep_idxs=chain(prep_idxs, other_preps),
            meas_idxs=meas_idxs,
        )
        self._qubit_map = qubit_map
        self._circuit = circuit
        self._annotations = [Twirl()] if annotations is None else list(annotations)
        if not any(isinstance(annotation, Twirl) for annotation in self._annotations):
            raise ValueError("Annotations must include a ''Twirl'' annotation.")

    @property
    def circuit(self) -> QuantumCircuit:
        """A circuit representation of this gate.

        .. note::

            The ``circuit.qubits`` are completely irrelevant and do not, for example, represent
            physical qubits. Instead, the mappnig ``dict(zip(circuit.qubits, qubit_idxs))`` provides
            the recipe for which physical qubits each qubit in the circuit corresponds to. See also
            :meth:`~.iter_ops`.

        """
        return self._circuit

    @property
    def annotations(self) -> list[Annotation]:
        """The annotations to use with this gate."""
        return self._annotations

    @property
    def constituent_gate_idxs(self) -> Iterator[tuple[int, ...]]:
        for instr in self._circuit:
            if _is_meas(instr) or _is_prep(instr):
                continue
            yield tuple(map(self._qubit_map.get, instr.qubits))

    def iter_ops(self) -> Iterable[tuple[tuple[int, ...], Operation]]:
        """Iterate through the operations that compose this gate in circuit order.

        Yields:
            Tuples ``(physical_qubits, operation)`` for each instruction in :attr:`~.circuit`.
        """
        for instr in self._circuit:
            yield tuple(map(self._qubit_map.get, instr.qubits)), instr.operation

    @cached_property
    def model_gate(self) -> ModelGate:
        """The model for this gate."""

        unitary_part = []
        reset_qubits = set()
        meas_qubits = set()

        for inst in self.circuit:
            if inst.name.startswith("reset"):
                if (qubit := inst.qubits[0]) in reset_qubits:
                    raise ValueError(
                        f"Cannot convert gate {self.name} into a ModelGate, "
                        f"as two resets occur on {qubit}."
                    )
                reset_qubits.add(qubit)
            elif inst.name.startswith("meas"):
                if (qubit := inst.qubits[0]) in meas_qubits:
                    raise ValueError(
                        f"Cannot convert QiskitGate {self.name} into a ModelGate, "
                        f"as two measurements occur on {qubit}."
                    )
                elif qubit in reset_qubits:
                    raise ValueError(
                        f"Cannot convert QiskitGate {self.name} into a ModelGate, "
                        f"as a measurement occurs after reset on {qubit}."
                    )
                meas_qubits.add(qubit)
            else:
                for qubit in inst.qubits:
                    if (qubit := inst.qubits[0]) in reset_qubits:
                        raise ValueError(
                            f"Cannot convert QiskitGate {self.name} into a ModelGate, as an "
                            f"instruction occurs after a reset on {qubit}."
                        )
                    elif qubit in meas_qubits:
                        raise ValueError(
                            f"Cannot convert QiskitGate {self.name} into a ModelGate, as a "
                            f"non-reset instruction occurs after a measurement on {qubit}."
                        )
                qubits = tuple(self._qubit_map[q] for q in inst.qubits)
                unitary_part.append((qubits, Clifford.from_circuit(inst.operation)))

        return ModelGate(
            self.name,
            unitary_part,
            qubit_idxs=self.qubit_idxs,
            meas_idxs=self.meas_idxs,
            prep_idxs=self.prep_idxs,
        )

    def draw(self, *args, **kwargs):
        """Draw this gate as a circuit diagram.

        Wire labels display the mapping from virtual qubit indices (the circuit's qubit
        ordering) to physical qubit indices, using the Qiskit :class:`~.TranspileLayout`
        convention, e.g. ``v_0 -> 5``.

        Args:
            *args: Positional keyword arguments forwarded to
                :meth:`~qiskit.circuit.QuantumCircuit.draw`.
            **kwargs: Keyword arguments forwarded to
                :meth:`~qiskit.circuit.QuantumCircuit.draw`.

        Returns:
            Text, matplotlib figure, or latex depending on the ``output`` kwarg.
        """
        from qiskit.transpiler import Layout, TranspileLayout

        num_physical = max(self.qubit_idxs) + 1
        qc = QuantumCircuit(num_physical)
        for creg in self.circuit.cregs:
            qc.add_register(creg)

        clbits = [b for creg in self.circuit.cregs for b in creg]
        qc.compose(self.circuit, qubits=list(self.qubit_idxs), clbits=clbits, inplace=True)

        active_idxs = set(self.qubit_idxs)
        num_idle = num_physical - self.num_qubits
        virt_reg = QuantumRegister(self.num_qubits, name="v")
        idle_reg = QuantumRegister(num_idle, name="idle") if num_idle else []
        layout_dict = dict(zip(virt_reg, self.qubit_idxs))
        layout_dict.update(zip(idle_reg, (i for i in range(num_physical) if i not in active_idxs)))
        initial_layout = Layout(layout_dict)
        input_qubit_mapping = dict(zip(virt_reg, range(self.num_qubits)))
        qc._layout = TranspileLayout(  # noqa: SLF001
            initial_layout=initial_layout,
            input_qubit_mapping=input_qubit_mapping,
            final_layout=None,
            _input_qubit_count=self.num_qubits,
            _output_qubit_list=qc.qubits,
        )

        kwargs.setdefault("idle_wires", False)
        return qc.draw(*args, **kwargs)

    def __eq__(self, other):
        # if self.qubit_idxs and self.circuit.qubits are permuted in the same way, then the
        # instruction has not changed. this equality implementation accounts for this permutation
        # freedom. note that QuantumCircuit equality is based on DAGCircuit equality, so that things
        # like instruction ordering are accounted for.

        if not (
            isinstance(other, QiskitGate)
            and self.name == other.name
            and self.meas_idxs == other.meas_idxs
            and self.prep_idxs == other.prep_idxs
            and self.annotations == other.annotations
            # qubit_idxs get unordered comparison because we'll be testing permuted circuit equality
            and set(self.qubit_idxs) == set(other.qubit_idxs)
        ):
            return False

        # we use compose() to perform circuit permutations before comparison
        self_circuit = QuantumCircuit(max_qubit := max(self.qubit_idxs) + 1)
        self_circuit.compose(self.circuit, qubits=self.qubit_idxs, inplace=True)
        other_circuit = QuantumCircuit(max_qubit)
        other_circuit.compose(other.circuit, qubits=other.qubit_idxs, inplace=True)

        return self_circuit == other_circuit

    def __repr__(self):
        qubits = int_sequence_to_str("qubits", self.qubit_idxs)
        prep = f", {int_sequence_to_str('prep', self.sorted_prep_idxs)}" if self.prep_idxs else ""
        meas = f", {int_sequence_to_str('meas', self.sorted_meas_idxs)}" if self.meas_idxs else ""
        return (
            f"QiskitGate(<name={self.name}, {qubits}, "
            f"ops={dict(self.circuit.count_ops())}{prep}{meas}>)"
            f"@{hex(id(self))}"
        )
