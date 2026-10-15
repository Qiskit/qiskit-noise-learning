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

"""Synthetic device topologies for the experiment-building benchmarks.

Two families are provided, chosen to bracket the coupling-map densities that real IBM devices
present to noise learning:

* ``heavy_hex`` -- the IBM heavy-hex "brick" lattice used by Eagle/Heron processors. Maximum
  degree 3, roughly ``1.14`` edges per qubit, and 3 disjoint two-qubit layers.
  :func:`heavy_hex_lattice` reproduces ``FakeFez``'s coupling map exactly at
  ``(n_rows=8, row_len=16)``, which ``run_bench.py --self-check`` verifies.
* ``grid`` -- a square lattice, as on the Nighthawk-class ``ibm_miami`` (which is exactly a
  ``12 x 10`` grid). Maximum degree 4, roughly ``1.9`` edges per qubit, and 4 disjoint two-qubit
  layers.

Edge count per qubit is what drives the size of a 2-local Pauli-Lindblad model, so the grid family
carries roughly 1.6x the generators per qubit that the heavy-hex family does. Layer count sets how
many gates the experiment learns. Both axes matter for build cost, and the two families separate
them from qubit count.
"""

from collections.abc import Callable, Sequence

import rustworkx
from qiskit.transpiler import CouplingMap

#: Row length of the generated heavy-hex lattices. Real Eagle and Heron devices use 15 and 16
#: respectively; 16 is kept fixed so that generated lattices are prefixes of the same family.
HEAVY_HEX_ROW_LEN = 16

#: Spacing between rung columns in the heavy-hex lattice.
HEAVY_HEX_RUNG_PERIOD = 4


def heavy_hex_lattice(
    n_rows: int, row_len: int = HEAVY_HEX_ROW_LEN, rung_period: int = HEAVY_HEX_RUNG_PERIOD
) -> tuple[int, list[tuple[int, int]]]:
    """Build an IBM-style heavy-hex lattice.

    The lattice is ``n_rows`` horizontal chains of ``row_len`` qubits, joined by single "rung"
    qubits placed every ``rung_period`` columns. Rung columns alternate their offset between
    consecutive bands, which is what gives the lattice its hexagonal faces.

    Qubit indices follow the convention used by IBM devices: band ``b`` occupies a contiguous
    block holding its ``row_len`` chain qubits followed by the ``row_len // rung_period`` rung
    qubits that connect it to band ``b + 1``.

    Args:
        n_rows: The number of horizontal chains.
        row_len: The number of qubits per chain.
        rung_period: The column spacing between rungs.

    Returns:
        The number of qubits, and the undirected edge list.
    """
    n_rungs = row_len // rung_period
    period = row_len + n_rungs
    edges: list[tuple[int, int]] = []

    def row_q(band: int, col: int) -> int:
        return band * period + col

    def rung_q(band: int, idx: int) -> int:
        return band * period + row_len + idx

    for band in range(n_rows):
        for col in range(row_len - 1):
            edges.append((row_q(band, col), row_q(band, col + 1)))

    for band in range(n_rows - 1):
        # Alternating rung offsets; the values 3 and 1 are what reproduce IBM's layouts.
        offset = 3 if band % 2 == 0 else 1
        for idx in range(n_rungs):
            col = offset + rung_period * idx
            edges.append((row_q(band, col), rung_q(band, idx)))
            edges.append((rung_q(band, idx), row_q(band + 1, col)))

    num_qubits = (n_rows - 1) * period + row_len
    return num_qubits, edges


def _coupling_map_from_edges(num_qubits: int, edges: Sequence[tuple[int, int]]) -> CouplingMap:
    """Build a bidirectional :class:`~qiskit.transpiler.CouplingMap` from undirected edges."""
    coupling_map = CouplingMap()
    for qubit in range(num_qubits):
        coupling_map.add_physical_qubit(qubit)
    for left, right in edges:
        coupling_map.add_edge(left, right)
        coupling_map.add_edge(right, left)
    return coupling_map


def _truncate(
    num_qubits: int, edges: Sequence[tuple[int, int]], keep: int
) -> tuple[int, list[tuple[int, int]]]:
    """Restrict a lattice to a connected induced subgraph on exactly ``keep`` qubits.

    Qubits are kept in breadth-first order from qubit ``0``, so the result is always connected,
    and relabelled to ``range(keep)`` in that order.

    Args:
        num_qubits: The number of qubits in the full lattice.
        edges: The full lattice's undirected edge list.
        keep: The number of qubits to keep.

    Returns:
        The kept qubit count and the relabelled induced edge list.

    Raises:
        ValueError: If ``keep`` exceeds ``num_qubits``.
    """
    if keep > num_qubits:
        raise ValueError(f"Cannot keep {keep} of {num_qubits} qubits.")
    if keep == num_qubits:
        return num_qubits, list(edges)

    graph = rustworkx.PyGraph()
    graph.add_nodes_from(range(num_qubits))
    graph.add_edges_from([(left, right, None) for left, right in edges])

    order = [0]
    seen = {0}
    head = 0
    while len(order) < keep and head < len(order):
        for neighbour in sorted(graph.neighbors(order[head])):
            if neighbour not in seen:
                seen.add(neighbour)
                order.append(neighbour)
                if len(order) == keep:
                    break
        head += 1

    relabel = {old: new for new, old in enumerate(order)}
    new_edges = [
        (relabel[left], relabel[right])
        for left, right in edges
        if left in relabel and right in relabel
    ]
    return keep, new_edges


def heavy_hex_coupling_map(num_qubits: int, row_len: int = HEAVY_HEX_ROW_LEN) -> CouplingMap:
    """A heavy-hex coupling map on exactly ``num_qubits`` qubits.

    The smallest lattice from :func:`heavy_hex_lattice` with at least ``num_qubits`` qubits is
    generated and then truncated. Truncation removes at most ``row_len + row_len // 4 - 1``
    qubits, so the result is a full lattice plus a partial final band -- the same kind of ragged
    boundary a real device subset has.

    Args:
        num_qubits: The exact number of qubits wanted.
        row_len: The chain length of the underlying lattice.

    Returns:
        A bidirectional coupling map on ``range(num_qubits)``.
    """
    n_rows = 1
    while heavy_hex_lattice(n_rows, row_len)[0] < num_qubits:
        n_rows += 1
    total, edges = heavy_hex_lattice(n_rows, row_len)
    total, edges = _truncate(total, edges, num_qubits)
    return _coupling_map_from_edges(total, edges)


def grid_dimensions(num_qubits: int, max_aspect: float = 4.0) -> tuple[int, int]:
    """Pick the ``(rows, cols)`` of the grid used for ``num_qubits`` qubits.

    An exact factorization is preferred when one exists whose aspect ratio is at most
    ``max_aspect``; among those, the most square is chosen. Otherwise the most square grid with at
    least ``num_qubits`` sites is used and the surplus is truncated by
    :func:`grid_coupling_map`.

    Args:
        num_qubits: The number of qubits wanted.
        max_aspect: The largest tolerated ``cols / rows`` ratio for an exact factorization.

    Returns:
        The grid's row and column counts, with ``rows <= cols``.
    """
    exact = [
        (rows, num_qubits // rows)
        for rows in range(1, num_qubits + 1)
        if num_qubits % rows == 0
        and rows <= num_qubits // rows
        and (num_qubits // rows) / rows <= max_aspect
    ]
    if exact:
        return max(exact)

    rows = 1
    while rows * rows < num_qubits:
        rows += 1
    cols = -(-num_qubits // rows)
    return rows, cols


def grid_coupling_map(num_qubits: int) -> CouplingMap:
    """A square-lattice coupling map on exactly ``num_qubits`` qubits.

    Args:
        num_qubits: The exact number of qubits wanted.

    Returns:
        A bidirectional coupling map on ``range(num_qubits)``.
    """
    rows, cols = grid_dimensions(num_qubits)
    edges = [
        (row * cols + col, row * cols + col + 1) for row in range(rows) for col in range(cols - 1)
    ]
    edges += [
        (row * cols + col, (row + 1) * cols + col) for row in range(rows - 1) for col in range(cols)
    ]
    total, edges = _truncate(rows * cols, edges, num_qubits)
    return _coupling_map_from_edges(total, edges)


#: The topology families available to benchmark cases.
TOPOLOGIES: dict[str, Callable[[int], CouplingMap]] = {
    "heavy_hex": heavy_hex_coupling_map,
    "grid": grid_coupling_map,
}


def layer_couplings(coupling_map: CouplingMap) -> list[list[tuple[int, int]]]:
    """Split a coupling map into disjoint two-qubit gate layers.

    This is the same edge-colouring decomposition the utility-benchmark notebooks use, so the
    generated gate sets have the same shape as the ones built from a real backend.

    Args:
        coupling_map: The coupling map to decompose.

    Returns:
        One sorted list of qubit pairs per layer.
    """
    graph = coupling_map.graph.to_undirected(multigraph=False)
    colouring = rustworkx.graph_bipartite_edge_color(graph)

    layers: dict[int, list[tuple[int, int]]] = {}
    for edge_idx, colour in colouring.items():
        endpoints = graph.get_edge_endpoints_by_index(edge_idx)
        layers.setdefault(colour, []).append(tuple(sorted(endpoints)))

    return [sorted(layers[colour]) for colour in sorted(layers)]
