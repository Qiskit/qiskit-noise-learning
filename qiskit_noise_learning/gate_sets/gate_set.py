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

"""GateSet"""

from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, TypeVar

from qiskit.transpiler import Target

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from .model_gate_set import ModelGateSet

from ..utils.html_repr import HTMLTable
from .gate import Gate

GateType = TypeVar("GateType", bound=Gate)


class GateSet(Mapping[str, GateType], metaclass=ABCMeta):
    """A mapping of names to gates with qubit metadata.

    Args:
        num_qubits: How many qubits the QPU of interest has.
        qubit_subset: A subset of ``range(num_qubits)`` specifying the region of interest of the
            QPU. All gates added must act within this subset. By default, contains all qubits.
        target: An optional :class:`~.Target` against which operations will be validated whenever
            gates are added to the gate set. Its number of qubits must match ``num_qubits`` if both
            are present.
        name: Name for this gate set. If ``None``, :attr:`name` falls back to the class name.
        latex_str: An optional LaTeX string for rendering this gate set.
    """

    def __init__(
        self,
        num_qubits: int,
        qubit_subset: Iterable[int] | None = None,
        target: Target | None = None,
        name: str | None = None,
        latex_str: str | None = None,
    ):
        self._num_qubits = num_qubits
        self._name = name
        self._latex_str = latex_str
        if qubit_subset is None:
            self._qubit_subset = frozenset(range(self._num_qubits))
        else:
            self._qubit_subset = frozenset(qubit_subset)
            if not self._qubit_subset.issubset(range(self._num_qubits)):
                raise ValueError(f"`qubit_subset` must be a subset of range({self._num_qubits}).")

        self._gates: dict[str, GateType] = {}
        self._target = target

    @property
    @abstractmethod
    def model_gate_set(self) -> "ModelGateSet":
        """Return a :class:`ModelGateSet` representing this gate set."""

    @property
    def name(self) -> str:
        """Name for this gate set, defaulting to the class name."""
        return self._name if self._name is not None else type(self).__name__

    @property
    def latex_str(self) -> str:
        """A LaTeX string for this gate set."""
        return self._latex_str

    @property
    def label(self) -> str:
        """A string label for use in plotter legends."""
        if self.latex_str:
            return f"${self.latex_str}$"

        return self.name

    @property
    def math_label(self) -> str:
        """A string label for use within latex math mode."""
        if self.latex_str:
            return self.latex_str
        return r"\text{" + self.name + r"}"

    @property
    def num_qubits(self) -> int:
        """The total number of qubits of the device this gateset acts on."""
        return self._num_qubits

    @property
    def qubit_subset(self) -> frozenset[int]:
        """The indices of the subset of device qubits that all gates act on."""
        return self._qubit_subset

    @property
    def target(self) -> Target | None:
        """The target of this gateset, if one exists."""
        return self._target

    def add_gate(self, gate: GateType):
        """Add a gate to the gate set.

        Args:
            gate: The gate to add.

        Raises:
            ValueError: If the gate acts on some qubits outside of the valid range, or if the gate
                name is already used in the gate set.
        """

        if not self._qubit_subset.issuperset(gate.qubit_idxs):
            raise ValueError("The provided gate acts on some qubits outside of the valid range.")

        if gate.name in self._gates:
            raise ValueError(f"The gate name {gate.name} is already used.")

        self._gates[gate.name] = gate

    def __getitem__(self, key: str) -> GateType:
        return self._gates[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._gates

    def __len__(self):
        return len(self._gates)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(<num_qubits={self.num_qubits}, num_gates={len(self)}, "
            f"names={sorted(self)}>)"
        )

    def draw(self) -> "go.Figure":
        """Draw the device topology with per-gate coloring.

        Each gate's 2-qubit interactions are drawn as colored edges on the device
        coupling graph. Gates that act only on single qubits (such as preparation
        and measurement) are shown as shaped markers on the relevant nodes.

        Returns:
            A plotly Figure.

        Raises:
            ValueError: If :attr:`target` is ``None``.
            ImportError: If ``plotly`` or ``qiskit-ibm-runtime`` is not installed.
        """
        from ..visualizations.gate_set_topology import gate_set_topology

        return gate_set_topology(self)

    def _custom_html_columns(self) -> tuple[list[str], list[Callable[[GateType], str]]]:
        """Subclass plugin to append custom columns to the HTML repr."""
        return [], []

    def _repr_html_(self) -> str:
        caption = f"{self.name} — {self.num_qubits} qubits, {len(self)} gates"

        table = (
            HTMLTable()
            .set_caption(caption)
            .extend_columns(["Name", "Num Qubits", "Gate Edges", "Prep", "Meas", "Idling"])
        )

        extra_cols, extra_fns = self._custom_html_columns()
        table.extend_columns(extra_cols)

        for gate in self._gates.values():
            row = [
                gate.name,
                str(gate.num_qubits),
                ", ".join("_".join(map(str, idxs)) for idxs in sorted(gate.constituent_gate_idxs)),
                ", ".join(str(q) for q in gate.sorted_prep_idxs),
                ", ".join(str(q) for q in gate.sorted_meas_idxs),
                ", ".join(str(q) for q in sorted(gate.idling_idxs)),
            ]
            row.extend(extra_fn(gate) for extra_fn in extra_fns)
            table.add_row(row)

        return table._repr_html_()
