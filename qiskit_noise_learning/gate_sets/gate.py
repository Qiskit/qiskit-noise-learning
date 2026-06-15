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

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model_gate import ModelGate


def int_sequence_to_str(desc: str, seq: Sequence[int]) -> str:
    """Briefly describe a short sequence of integers as a string."""
    return f"{desc}={seq}" if len(seq) < 10 else f"num_{desc}={len(seq)}"


class Gate(ABC):
    """Represents qubit metadata for a single gate in a gate set.

    Typically serves as a base class for types that implement non-trivial gates.

    Args:
        name: The gate name.
        qubit_idxs: The physical qubit indices that the gate acts on.
        prep_idxs: The physical qubit indices that this gate prepares, or resets to the 0 state.
        meas_idxs: The physical qubit indices that this gate measures.
        latex_str: An optional LaTeX string for rendering this gate. If ``None``,
            :attr:`latex_str` falls back to :attr:`name`.
    """

    def __init__(
        self,
        name: str,
        qubit_idxs: Iterable[int],
        prep_idxs: Iterable[int] = (),
        meas_idxs: Iterable[int] = (),
        latex_str: str | None = None,
    ):
        self._name = name
        self._latex_str = latex_str
        self._qubit_idxs = tuple(qubit_idxs)

        self._prep_idxs = frozenset(prep_idxs)
        self._meas_idxs = frozenset(meas_idxs)

        if not self._prep_idxs.issubset(self._qubit_idxs):
            raise ValueError("`prep_idxs` must be a subset of `qubit_idxs`.")
        if not self._meas_idxs.issubset(self._qubit_idxs):
            raise ValueError("`meas_idxs` must be a subset of `qubit_idxs`.")

    @property
    @abstractmethod
    def model_gate(self) -> "ModelGate":
        """Return a :class:`ModelGate` representing this gate."""

    @property
    def name(self) -> str:
        """The gate name."""
        return self._name

    @property
    def latex_str(self) -> str:
        """A LaTeX string for rendering this gate."""
        return self._latex_str if self._latex_str is not None else self._name

    @property
    def num_qubits(self) -> int:
        """The number of qubits this gate acts on."""
        return len(self._qubit_idxs)

    @property
    def qubit_idxs(self) -> tuple[int, ...]:
        """The physical qubit indices this gate acts on."""
        return self._qubit_idxs

    @property
    def meas_idxs(self) -> frozenset[int]:
        """The physical qubit indices that this gate measures."""
        return self._meas_idxs

    @cached_property
    def sorted_meas_idxs(self) -> list[int]:
        """The indices of the measured qubits in increasing order."""
        return sorted(self._meas_idxs)

    @property
    def prep_idxs(self) -> frozenset[int]:
        """The physical qubit indices that this gate prepares (or resets)."""
        return self._prep_idxs

    @cached_property
    def sorted_prep_idxs(self) -> list[int]:
        """The indices of the reset qubits in increasing order."""
        return sorted(self._prep_idxs)

    @property
    def idling_idxs(self) -> set[int]:
        """The physical qubit indices that this gate is idling on."""
        idxs = set(self.qubit_idxs)
        idxs.difference_update(self.prep_idxs)
        idxs.difference_update(self.meas_idxs)
        idxs.difference_update(self.gate_idxs)
        return idxs

    @cached_property
    def gate_idxs(self) -> frozenset[int]:
        """The physical indices where this gate undergoes unitary action."""
        return frozenset({idx for gate_idxs in self.constituent_gate_idxs for idx in gate_idxs})

    @property
    @abstractmethod
    def constituent_gate_idxs(self) -> Iterator[tuple[int, ...]]:
        """Iterator over tuples of physical indices that specify where constituent gates act.

        Some subclasses may not have a meaningful notion of what a "constituent gate" is, because
        they don't choose to represent unitary action by some seperable representation.
        The only contract they need to obey is that the union of all yielded integers is equal to
        :attr:`~.gate_idxs`.
        """

    def __repr__(self):
        qubits = int_sequence_to_str("qubits", self.qubit_idxs)
        prep = f", {int_sequence_to_str('prep', self.sorted_prep_idxs)}" if self.prep_idxs else ""
        meas = f", {int_sequence_to_str('meas', self.sorted_meas_idxs)}" if self.meas_idxs else ""
        return (
            f"{self.__class__.__name__}(<name={self.name}, {qubits}, "
            f"ops={prep}{meas}>)@{hex(id(self))}"
        )
