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

"""Utility to build with preparation noise injection."""

from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp
from qiskit.converters import dag_to_circuit
from samplomatic import Twirl
from samplomatic.builders import pre_build
from samplomatic.constants import Direction
from samplomatic.partition import QubitIndicesPartition
from samplomatic.pre_samplex import PreCollect, PreEdge, PreInjectNoise, PreSamplex
from samplomatic.samplex import Samplex
from samplomatic.utils import get_annotation
from samplomatic.virtual_registers import VirtualType


def add_preparation_noise(pre_samplex: PreSamplex, noise_ref: str, modifier_ref: str = ""):
    """Adds an inject noise pre-node to the pre-samplex.

    This function does not work generically, and the pre-samplex is expected to begin with a
    pre-collect node that the inject noise pre-node will be attached to. This ensures that the
    pre-samplex can be finalized into a samplex. For this condition to hold, the circuit from
    which the pre-samplex was built to have its first twirled box to be full width and left-dressed.

    Args:
        pre_samplex: The pre-samplex to mutate in place.
        noise_ref: The name of the reference to the noise map.
        modifier_ref: The name used for the argument to modify the rates of the noise map.

    Raises:
        ValueError: If the first node in the pre-samplex is not a pre-collect. This should always
            be true for a pre-samplex that was built from a circuit that began with a left-dressed
            box.
        ValueError: If the `noise_ref` collides with any of those already present on the
            `pre-samplex`.

    """
    graph = pre_samplex.graph
    pre_node = graph.nodes()[0]
    if not isinstance(pre_node, PreCollect):
        raise ValueError(
            "Cannot add preparation noise to a pre-samplex without a pre-collect as its first node."
        )

    subsystems = pre_node.subsystems
    if (num_subsys := pre_samplex._pauli_lindblad_maps.get(noise_ref)) and num_subsys != len(  # noqa: SLF001
        subsystems
    ):
        raise ValueError(
            f"Cannot add Pauli-Lindblad map with reference `{noise_ref}` on `{subsystems}` and a "
            f"different subsystem with length `{num_subsys}`."
        )
    else:
        pre_samplex._pauli_lindblad_maps[noise_ref] = len(subsystems)  # noqa: SLF001

    # Ensure that the subsystems are in the order of the qubit map so that the map is addressing the
    # correct qubit indices
    subsystems = QubitIndicesPartition.from_elements(
        [i for i in pre_samplex.qubit_map.values() if (i,) in subsystems]
    )

    if modifier_ref:
        pre_samplex._noise_modifiers[noise_ref].add(modifier_ref)  # noqa: SLF001

    pre_inject_noise = PreInjectNoise(
        subsystems,
        Direction.RIGHT,
        VirtualType.PAULI,
        noise_ref,
        modifier_ref,
        next(pre_samplex._pauli_lindblad_map_count),  # noqa: SLF001
    )
    node_idx = graph.add_node(pre_inject_noise)
    graph.add_edge(node_idx, 0, PreEdge(subsystems, Direction.RIGHT))


def build_with_prep_noise(
    circuit: QuantumCircuit, prep_noise_ref: str = "prep", prep_modifier_ref: str = ""
) -> tuple[QuantumCircuit, Samplex]:
    """Build a template circuit with injectable preparation noise.

    This function uses the normal samplex build path except adds an additional node for injectable
    preparation noise.

    Args:
        circuit: The annotated circuit to build.
        noise_ref: The name of the reference to the noise map.
        modifier_ref: The name used for the argument to modify the rates of the noise map.

    Returns:
        The template circuit and the samplex.

    Raises:
        ValueError: If the circuit does not begin with a left-dressed twirl annotated box.
    """
    if len(circuit) == 0 or not isinstance(op := circuit[0].operation, BoxOp):
        raise ValueError("Circuit does not begin with a box operation.")

    if (twirl := get_annotation(op, Twirl)) is None:
        raise ValueError("Initial box operation does not contain a twirl annotaiton.")

    if twirl.dressing == "right":
        raise ValueError('Dressing of the initial box\'s twirl annotation must be `"left"`.')

    template_state, pre_samplex = pre_build(circuit)
    add_preparation_noise(pre_samplex, prep_noise_ref, prep_modifier_ref)
    return dag_to_circuit(template_state.template), pre_samplex.finalize().finalize()
