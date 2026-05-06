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

"""Utils."""

from itertools import product

from rustworkx import PyGraph


def generate_bases(graph: PyGraph) -> list[str]:
    """Generate the 9 basis strings for measuring single- and two-qubit Pauli fidelities.

    For triangle-free topologies, 9 bases are always sufficient to measure all 9 non-identity Pauli
    pair combinations on every edge. This function uses an alternating color strategy where
    adjacent qubits are assigned different colors.

    .. note::

        While 9 bases are sufficient, fewer bases may be possible in some cases.

    Args:
        graph: The graph to generate the bases for.

    Returns:
        The basis strings.
    """
    qubit_color = {}
    visited = set()

    for start_q in graph.node_indices():
        if start_q in visited:
            continue

        queue = [(start_q, 0)]
        visited.add(start_q)
        qubit_color[start_q] = 0

        while queue:
            q, color = queue.pop(0)

            for q1, q2 in graph.edge_list():
                neighbor = None
                if q1 == q and q2 not in visited:
                    neighbor = q2
                elif q2 == q and q1 not in visited:
                    neighbor = q1

                if neighbor:
                    visited.add(neighbor)
                    qubit_color[neighbor] = 1 - color
                    queue.append((neighbor, 1 - color))

    generated_bases = []
    all_pauli_pairs = list(product(["X", "Y", "Z"], repeat=2))

    for p0, p1 in all_pauli_pairs:
        basis_list = []
        for q in graph.node_indices():
            color = qubit_color[q]
            basis_list.append(p0 if color == 0 else p1)
        basis = "".join(basis_list[::-1])
        generated_bases.append(basis)

    return generated_bases
