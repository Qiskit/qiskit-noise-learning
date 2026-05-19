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

"""QiskitGateSet"""

from collections.abc import Iterable
from itertools import count

from qiskit.circuit import (
    Annotation,
    CircuitInstruction,
    Measure,
    Operation,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.transpiler import Target
from samplomatic import Twirl

from .gate_set import GateSet
from .model_gate_set import ModelGateSet
from .qiskit_gate import QiskitGate


class GateBuilder:
    """Helper context to build new gates for a :class:`~.QiskitGateSet`."""

    def __init__(self, gate_set: "QiskitGateSet", name: str, idle_unused: bool = True):
        self.name = name
        self._gate_set = gate_set
        self._idle_unused = idle_unused
        self.circuit = QuantumCircuit(gate_set.num_qubits)
        self._box = None

    def __enter__(self):
        # as an implementation detail, we choose to use box because it give us access to noop()
        self._box = self.circuit.box()
        self._box.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._idle_unused:
            self.circuit.noop(sorted(self._gate_set.qubit_subset))
        try:
            self._box.__exit__(exc_type, exc_value, traceback)
        finally:
            self._box = None
        self._gate_set.add_box_as_gate(self.circuit[0], name=self.name)
        return False


class QiskitGateSet(GateSet[QiskitGate]):
    """A gate set whose noise is to be learned and that is specified using Qiskit objects.

    Here, we are using "gate" in the context of noise learning, where our operations of interest are
    not typically the smallest operations that are discretely executed on a device (such as ``cz``
    gates), but rather collections of such operations whose noise will be learned together as a
    unit. To the point, typically, in this class, gates are layers. They need not be however;
    partial width layers and layers with overlapping operations within are valid. We require this
    object to represent a set of them because in certain cases their noise needs to be learned
    together in order to have a consistent gauge defined between their noise models, or to allow
    parametrizations of their noise models to be correlated.

    This object satisfies the Python mapping protocol so that, for example, gates can be extracted
    with dictionary syntax. The names of gates are always strings.

    .. code:: python

        >>> from qiskit_noise_learning.gate_sets import QiskitGateSet
        >>> from qiskit.circuit import QuantumCircuit

        >>> # instantiate a new gate set on 10 qubits
        >>> gate_set = QiskitGateSet(10)

        >>> # the gate set comes populated with preparation and measurement gates on all 10 qubits
        >>> assert len(gate_set) == 2
        >>> assert "P" in gate_set and "M" in gate_set

    Args:
        num_qubits: How many qubits the QPU of interest has. If a ``target`` is provided, this field
            may be omitted.
        target: An optional :class:`~.Target` against which operations will be validated whenever
            gates are added to the gate set. Its number of qubits must match ``num_qubits`` if both
            are present.
        qubit_subset: A subset of ``range(num_qubits)`` specifying the region of interest of the
            QPU. All gates added must act within this subset. By default, contains all qubits.
            When ``add_default_spam`` is ``True``, the iteration order of this argument determines
            the qubit ordering of the default preparation and measurement gates.
        add_default_spam: Whether to initialize the gateset with gates that respectively implement
            state preparation (given name ``"P"``) and state measurement (given name ``"M"``) on
            all qubits in the region of interest.
    """

    def __init__(
        self,
        num_qubits: int | None = None,
        *,
        target: Target | None = None,
        qubit_subset: Iterable[int] | None = None,
        add_default_spam: bool = True,
    ):
        if num_qubits is None and target is None:
            raise ValueError("At least one of `num_qubits` or `target` must be specified.")
        if num_qubits and target and num_qubits != target.num_qubits:
            raise ValueError("The value of `num_qubits` must match `target.num_qubits`.")

        qubit_subset = list(qubit_subset) if qubit_subset is not None else None
        super().__init__(
            num_qubits=num_qubits or target.num_qubits, qubit_subset=qubit_subset, target=target
        )

        self._name_iter = tuple(f"L{idx}" for idx in count())

        if add_default_spam:
            self.add_measurement(name="M", qubit_idxs=qubit_subset)
            self.add_preparation(name="P", qubit_idxs=qubit_subset)

    def _custom_html_columns(self):
        def twirl(gate: QiskitGate) -> str:
            for annotation in gate.annotations:
                if isinstance(annotation, Twirl):
                    return str(annotation)
            return ""

        return ["Twirl"], [twirl]

    @property
    def model_gate_set(self) -> ModelGateSet:
        """The model for this gate set."""

        model_gate_set = ModelGateSet(
            num_qubits=self.num_qubits,
            qubit_subset=self.qubit_subset,
            coupling_map=self._target.build_coupling_map() if self._target else None,
        )
        for gate in self._gates.values():
            model_gate_set.add_gate(gate=gate.model_gate)

        return model_gate_set

    @property
    def target(self) -> Target | None:
        """The target of this gateset, if one exists."""
        return self._target

    def add_gate(self, gate: QiskitGate):
        """Add a gate to the gate set.

        .. code:: python
            >>> from qiskit_noise_learning.gate_sets import QiskitGateSet, QiskitGate
            >>> from qiskit.circuit import QuantumCircuit

            >>> gate_set = QiskitGateSet(10)
            >>> circuit = QuantumCircuit(5)
            >>> circuit.cx(3, 4)
            >>> gate_set.add_gate(QiskitGate("gate0", circuit, [4, 5, 7, 8, 9]))

            >>> assert "gate0" in gate_set

        Args:
            gate: The gate to add.

        Raises:
            ValueError: If the gate acts on some qubits outside of the valid range, or this gate set
                has a target and some member of the gate does not comply, or if the name is already
                used in the gate set.
        """
        if self.target:
            for idxs, op in gate.iter_ops():
                if not self.target.instruction_supported(op.name, qargs=idxs):
                    raise ValueError(f"Operation {op} on {idxs} is not supported in the target.")

        super().add_gate(gate)

    def add_box_as_gate(self, box_instr: CircuitInstruction, *, name: str | None = None) -> str:
        """Add a Qiskit circuit instruction containing a box operation as a gate.

        .. code:: python
            >>> from qiskit_noise_learning.gate_sets import QiskitGateSet, QiskitGate
            >>> from qiskit.circuit import QuantumCircuit

            >>> gate_set = QiskitGateSet(10)
            >>> circuit = QuantumCircuit(10)
            >>> with circuit.box():
            ...     circuit.cx(3, 4)
            ...     # use noop to indicate that only the first 7 qubits will be part of the gate,
            ...     # otherwise it would be restricted to only qubits 3 and 4, and learned noise
            ...     # will only be with respect to those two qubits
            ...     circuit.noop(range(7))

            >>> name = gate_set.add_box_as_gate(circuit[0])

            >>> assert name in gate_set

        Args:
            box_instr: The circuit instruction containing a box operation.
            name: The name of the gate, or ``None`` to have a name chosen for you.

        Returns:
            The name of the added gate.

        Raises:
            ValueError: If the provided instruction does not contain a :class:`qiskit.circuit.BoxOp`
                operation, or if the instruction acts on non-physical qubits.
        """
        box = box_instr.operation
        if box.name != "box":
            raise ValueError("The provided instruction does not contain a `BoxOp` operation.")

        qreg = QuantumRegister(self.num_qubits, "q")
        try:
            qubit_idxs = [qreg.index(qubit) for qubit in box_instr.qubits]
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Cannot add the given box as a gate because it does not act on a single quantum "
                f"register named 'q' of size {self.num_qubits}."
            ) from exc

        name = name or next(self._name_iter)
        annotations = box.annotations or [Twirl()]
        self.add_gate(QiskitGate(name, box.body, qubit_idxs, annotations=annotations))
        return name

    def add_circuit_as_gate(
        self,
        circuit: QuantumCircuit,
        qubit_idxs: Iterable[int] | None = None,
        *,
        annotations: Iterable[Annotation] | None = None,
        name: str | None = None,
    ) -> str:
        """Add a quantum circuit object as a gate.

        This method is a thin wrapper for :meth:`~.add_gate` that constructs a :class:`~.QiskitGate`
        for you.

        .. code:: python
            >>> from qiskit_noise_learning.gate_sets import QiskitGateSet, QiskitGate
            >>> from qiskit.circuit import QuantumCircuit

            >>> gate_set = QiskitGateSet(10)
            >>> circuit = QuantumCircuit(5)
            >>> circuit.cx(3, 4)
            >>> name = gate_set.add_circuit_as_gate(circuit, [4, 5, 7, 8, 9])

            >>> assert name in gate_set

        .. note::

            The ``circuit.qubits`` are completely irrelevant and do not, for example, represent
            physical qubits. Instead, the mapping ``dict(zip(circuit.qubits, qubit_idxs))`` provides
            the recipe for which physical qubits each qubit in the circuit corresponds to.

        Args:
            circuit: The circuit to use as a gate.
            qubit_idxs: The physical qubits on which the circuit acts.
            annotations: The annotations that describe how to implement the circuit, or ``None``
                to use the default annotations of :class:`~.QiskitGate`.
            name: The name of the gate, or ``None`` to have a name chosen for you.

        Returns:
            The name of the added gate.
        """
        if qubit_idxs is None:
            qubit_idxs = range(circuit.num_qubits)
        name = name or next(self._name_iter)
        self.add_gate(QiskitGate(name, circuit, qubit_idxs, annotations=annotations))
        return name

    def build_new_gate(self, name: str | None = None, idle_unused: bool = True) -> GateBuilder:
        """Return a circuit builder whose contents will be added as a gate.

        .. code:: python
            >>> from qiskit_noise_learning.gate_sets import QiskitGateSet, QiskitGate
            >>> from qiskit.circuit import QuantumCircuit

            >>> gate_set = QiskitGateSet(10)
            >>> with gate_set.build_new_gate() as builder:
            ...    builder.circuit.cx(4, 5)

            >>> assert builder.name in gate_set
            >>> assert set(gate_set[builder.name].qubit_idxs) == set(range(10))


        Args:
            name: The name of the gate, or ``None`` to have a name chosen for you.
            idle_unused: Whether all qubits in :attr:`~.GateSet.qubit_subset` that are not
                already part of the gate will be automatically included as idling qubits.

        Returns:
            A circuit builder.
        """
        return GateBuilder(self, name or next(self._name_iter), idle_unused)

    def add_measurement(
        self,
        qubit_idxs: Iterable[int] | None = None,
        operation_type: type[Operation] = Measure,
        *,
        annotations: Iterable[Annotation] | None = None,
        name: str | None = None,
    ) -> str:
        """Add a gate to this gate set that measures specified qubits.

        Args:
            qubit_idxs: The physical qubit indices to measure.
            operation_type: The type of measurement operation to apply on each qubit.
            annotations: The annotations that describe how to implement the measurement, or ``None``
                to use the default annotations of :class:`~.QiskitGate`.
            name: The name of the gate, or ``None`` to have a name chosen for you.

        Returns:
            The name of the added gate.
        """
        qubit_idxs = sorted(self._qubit_subset) if qubit_idxs is None else list(qubit_idxs)
        circuit = QuantumCircuit(len(qubit_idxs), len(qubit_idxs))
        for idx in range(len(qubit_idxs)):
            circuit.append(operation_type(), [idx], [idx])

        name = name or next(self._name_iter)
        self.add_gate(QiskitGate(name, circuit, qubit_idxs, annotations=annotations))
        return name

    def add_preparation(
        self,
        qubit_idxs: Iterable[int] | None = None,
        *,
        annotations: Iterable[Annotation] | None = None,
        name: str | None = None,
    ) -> str:
        """Add a gate to this gate set that prepares (or resets) specified qubits.

        Args:
            qubit_idxs: The physical qubit indices to prepare.
            annotations: The annotations that describe how to implement the preparation, or ``None``
                to use the default annotations of :class:`~.QiskitGate`.
            name: The name of the gate, or ``None`` to have a name chosen for you.

        Returns:
            The name of the added gate.
        """
        qubit_idxs = sorted(self._qubit_subset) if qubit_idxs is None else list(qubit_idxs)
        circuit = QuantumCircuit(len(qubit_idxs))
        name = name or next(self._name_iter)
        self.add_gate(
            QiskitGate(name, circuit, qubit_idxs, prep_idxs=qubit_idxs, annotations=annotations),
        )
        return name
