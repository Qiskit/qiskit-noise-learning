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

from collections.abc import Iterable
from typing import Self

from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.gate_sets import GateSet

from .model_gate import ModelGate


class ModelGateSet(GateSet[ModelGate]):
    """A set of Clifford - MCM - reset gates represented as :class:`.ModelGate` instances.

    Args:
        num_qubits: How many qubits the QPU of interest has.
        qubit_subset: A subset of ``range(num_qubits)`` specifying the region of interest of the
            QPU. All gates added must act within this subset. By default, contains all qubits.
        coupling_map: A coupling map for the device. Defaults to the full coupling map on
            ``num_qubits``.
        name: Name for this gate set. If ``None``, :attr:`name` falls back to the class name.
        latex_str: An optional LaTeX string for rendering this gate set.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        qubit_subset: Iterable[int] | None = None,
        coupling_map: CouplingMap | None = None,
        name: str | None = None,
        latex_str: str | None = None,
    ):
        self._coupling_map = coupling_map or CouplingMap.from_full(num_qubits)
        super().__init__(
            num_qubits=num_qubits, qubit_subset=qubit_subset, name=name, latex_str=latex_str
        )

    @property
    def coupling_map(self) -> CouplingMap:
        """The coupling map for the device this gate set is modeling."""
        return self._coupling_map

    @property
    def model_gate_set(self) -> Self:
        return self
