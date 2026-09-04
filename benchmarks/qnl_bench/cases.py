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

"""The benchmark case suite.

A case fully determines an experiment-building workload. Everything below is derived from a
topology family, a qubit count, and the model locality -- no backend service or credentials are
needed, so a case is reproducible on any machine.

The workload each case builds mirrors the experiment-builder cell of
``notebooks/utility_benchmark_v2/executor_utility_benchmark_qnl_advanced_more.ipynb``: one gate per
disjoint two-qubit layer of the coupling map, a 2-local Pauli-Lindblad model over those gates with
1-local preparation and measurement, then per-gate multiplicative and additive path generation, SPAM
paths, an overall rank reduction, and sequence finalization.
"""

from dataclasses import dataclass, field

from qiskit.transpiler import Target

from qiskit_noise_learning.gate_sets import QiskitGateSet
from qiskit_noise_learning.models import PauliLindbladModel

from .lattices import TOPOLOGIES, layer_couplings

#: Basis gates for the synthetic targets. Only ``cz`` matters for the gate sets that get built;
#: the rest are present so the target resembles a real device's.
BASIS_GATES = ("cz", "rz", "sx", "x", "id", "measure", "reset")

#: The fragment depths from the utility benchmark, used by every case unless overridden.
DEFAULT_FRAGMENT_DEPTHS = (0, 1, 2, 4, 6, 12, 24)


@dataclass(frozen=True)
class BenchmarkCase:
    """A single experiment-building workload.

    Args:
        name: Short identifier, used on the command line.
        topology: A key of :data:`~.lattices.TOPOLOGIES`.
        num_qubits: The exact number of qubits in the device and in the gate set's qubit subset.
        k: The locality of the Pauli-Lindblad model for the two-qubit layer gates.
        fragment_depths: The fragment depths bound at the end of the build.
        shots: Shots per randomization; carried through to the experiment but does not affect cost.
        randomizations: Randomizations; carried through but does not affect cost.
    """

    name: str
    topology: str
    num_qubits: int
    k: int = 2
    fragment_depths: tuple[int, ...] = DEFAULT_FRAGMENT_DEPTHS
    shots: int = 64
    randomizations: int = 50

    def __str__(self) -> str:
        return f"{self.name} ({self.topology}, {self.num_qubits}q, k={self.k})"


#: The benchmark suite: both topology families across four qubit counts.
#:
#: The four sizes are exact, so heavy-hex and grid cases at the same size differ only in coupling
#: density -- that is the comparison the suite is built to support.
SUITE: tuple[BenchmarkCase, ...] = tuple(
    BenchmarkCase(name=f"{prefix}{num_qubits}", topology=topology, num_qubits=num_qubits)
    for topology, prefix in (("heavy_hex", "hex"), ("grid", "grid"))
    for num_qubits in (32, 64, 128, 256)
)


def case_by_name(name: str) -> BenchmarkCase:
    """Look a case up in :data:`SUITE` by name.

    Args:
        name: The case's :attr:`~BenchmarkCase.name`.

    Returns:
        The matching case.

    Raises:
        KeyError: If no case has that name.
    """
    for case in SUITE:
        if case.name == name:
            return case
    raise KeyError(f"Unknown case '{name}'. Available: {', '.join(c.name for c in SUITE)}.")


@dataclass
class CaseTopology:
    """The device-level data derived from a :class:`BenchmarkCase`.

    Args:
        coupling_map: The device coupling map.
        layers: The disjoint two-qubit layers, one gate each.
        gate_names: The gate name given to each layer.
    """

    coupling_map: object
    layers: list[list[tuple[int, int]]]
    gate_names: list[str] = field(default_factory=list)

    @property
    def num_edges(self) -> int:
        """The number of undirected couplings."""
        return sum(len(layer) for layer in self.layers)


def build_gate_set(case: BenchmarkCase) -> tuple[QiskitGateSet, CaseTopology]:
    """Build the gate set for a case: one ``cz`` layer gate per edge colour, plus ``P`` and ``M``.

    Args:
        case: The case to build for.

    Returns:
        The gate set, and the topology data it was derived from.
    """
    coupling_map = TOPOLOGIES[case.topology](case.num_qubits)
    layers = layer_couplings(coupling_map)
    target = Target.from_configuration(
        basis_gates=list(BASIS_GATES),
        num_qubits=case.num_qubits,
        coupling_map=coupling_map,
    )

    qubit_subset = sorted({qubit for layer in layers for pair in layer for qubit in pair})
    gate_set = QiskitGateSet(case.num_qubits, target=target, qubit_subset=qubit_subset)

    gate_names = []
    for idx, layer in enumerate(layers):
        name = f"layer_{idx + 1}"
        gate_names.append(name)
        with gate_set.build_new_gate(name, latex_str=name) as builder:
            for left, right in layer:
                builder.circuit.cz(left, right)

    return gate_set, CaseTopology(coupling_map=coupling_map, layers=layers, gate_names=gate_names)


def build_model(
    case: BenchmarkCase, gate_set: QiskitGateSet, topology: CaseTopology
) -> PauliLindbladModel:
    """Build the ``k``-local Pauli-Lindblad model for a case.

    Layer gates get ``case.k``-local generators; preparation and measurement get 1-local.

    Args:
        case: The case to build for.
        gate_set: The gate set from :func:`build_gate_set`.
        topology: The topology data from :func:`build_gate_set`.

    Returns:
        The model.
    """
    gate_k = dict.fromkeys(topology.gate_names, case.k) | {"P": 1, "M": 1}
    return PauliLindbladModel.k_local(gate_set, gate_k=gate_k)
