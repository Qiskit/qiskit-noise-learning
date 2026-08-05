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

"""Where to draw each qubit of a device on the plane.

Devices of a size IBM has built are drawn in the layout they are conventionally drawn in,
transcribed from ``qiskit_ibm_runtime.visualization.embeddings`` (Apache 2.0, the same project
family).  They are copied rather than imported to avoid dependence on a private member, as well as
its dependencies.

Any other device is laid out from its coupling graph instead, so that drawing a topology is
something this package can always do rather than something it can do for a list of sizes.  The
conventional layout still wins where there is one: it is the picture a reader recognizes, and a
derived layout is only ever a legible arrangement, not the familiar one.

Both paths return positions in **plot space** -- ``(x, y)``, y increasing upwards -- rather than the
``(row, column)`` the tables are written in, so that a caller does not have to know which of the two
it got.  They are also on a common scale: adjacent qubits sit about one unit apart.  Every geometric
constant in :mod:`~qiskit_noise_learning.visualizations.gate_set_topology` -- node radius, arc
radius, the offset between parallel edges, inches per unit -- is a length in that unit, so a layout
that ignored the convention would not merely look different, it would draw arcs larger than the
device.
"""

from collections.abc import Iterable, Sequence
from itertools import combinations, product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qiskit.transpiler import CouplingMap

#: Seed for the derived layout.  Fixed, because a spring layout is only reproducible if it is: an
#: unseeded one would move every qubit each time a figure was drawn, which for a docs build means a
#: page whose picture changes with no change to the source.
_LAYOUT_SEED = 1234

#: What a derived layout scales its median edge to, in the units described above.
_TARGET_EDGE_LENGTH = 1.0

#: Closest that any two qubits of a derived layout may end up, in the same units.  A spring layout
#: pulls a few pairs much tighter than its median edge, and two nodes closer than twice the arc
#: radius have overlapping arcs; this floor is just above that, and it is applied by scaling the
#: whole layout so that relative distances -- the part carrying the information -- are preserved.
_MIN_SEPARATION = 0.55


def qubit_coordinates(
    num_qubits: int, coupling_map: "CouplingMap | None" = None
) -> list[tuple[float, float]]:
    """Return the ``(x, y)`` position to draw each qubit of a device at.

    Args:
        num_qubits: Number of qubits in the device.
        coupling_map: The device's coupling map, used to lay out a device whose size is not one of
            the conventionally drawn ones.  Without it, such a device has no layout at all.

    Returns:
        One ``(x, y)`` position per qubit, indexed by qubit, with adjacent qubits about one unit
        apart.

    Raises:
        ValueError: If the device's size is not one of the conventionally drawn ones and no coupling
            map was given to derive a layout from, or if ``coupling_map`` describes a different
            number of qubits than ``num_qubits``.
    """
    # Checked before the layout is chosen, rather than where the coordinates are read: a map of the
    # wrong size is a caller error whichever layout this device happens to get, and a conventional
    # layout would otherwise absorb the disagreement and hand back positions for a different device.
    if coupling_map is not None and coupling_map.size() != num_qubits:
        raise ValueError(
            f"Coupling map covers {coupling_map.size()} qubits, but the device has {num_qubits}."
        )

    lattice = _lattice_coordinates(num_qubits)
    if lattice is not None:
        # A lattice row is drawn below the one before it, so the row index runs against the y axis.
        return [(float(column), float(-row)) for row, column in lattice]

    if coupling_map is None:
        raise ValueError(
            f"No conventional layout for a {num_qubits}-qubit device, and no coupling map to "
            "derive one from."
        )
    return _derived_coordinates(num_qubits, coupling_map)


def _derived_coordinates(num_qubits: int, coupling_map: "CouplingMap") -> list[tuple[float, float]]:
    """Lay a device out from its coupling graph, for sizes the tables do not cover.

    Args:
        num_qubits: Number of qubits in the device.
        coupling_map: The device's coupling map, which the caller has already checked covers exactly
            ``num_qubits`` qubits.

    Returns:
        One ``(x, y)`` position per qubit, indexed by qubit.
    """
    import rustworkx as rx

    graph = coupling_map.graph.to_undirected(multigraph=False)
    positions = rx.spring_layout(graph, seed=_LAYOUT_SEED)
    points = np.array([positions[qubit] for qubit in range(num_qubits)], dtype=float)
    points = _to_common_scale(points, list(graph.edge_list()))
    return [(float(x), float(y)) for x, y in points]


def _to_common_scale(points: np.ndarray, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    """Scale ``points`` onto the unit-spacing convention the drawing constants assume.

    A spring layout comes out normalized to roughly a unit box whatever the device, so its natural
    spacing shrinks as qubits are added -- the one property a caller measuring lengths in qubit
    spacings cannot tolerate.  Scaling is the whole adjustment: the arrangement itself is what the
    layout was for.

    Args:
        points: One ``(x, y)`` row per qubit.
        edges: The graph's edges, whose typical length sets the scale.  A device with none (no
            two-qubit gates at all) is scaled by the separation floor alone.

    Returns:
        The rescaled points.
    """
    lengths = [float(np.hypot(*(points[left] - points[right]))) for left, right in edges]
    positive = [length for length in lengths if length > 0]
    if positive:
        points = points * (_TARGET_EDGE_LENGTH / float(np.median(positive)))

    separations = [
        float(np.hypot(*(points[left] - points[right])))
        for left, right in combinations(range(len(points)), 2)
    ]
    closest = min((gap for gap in separations if gap > 0), default=None)
    if closest is not None and closest < _MIN_SEPARATION:
        points = points * (_MIN_SEPARATION / closest)
    return points


def _lattice_coordinates(num_qubits: int) -> list[tuple[int, int]] | None:
    """The conventional ``(row, column)`` layout for a device of this size, if there is one.

    Args:
        num_qubits: Number of qubits in the device.

    Returns:
        One ``(row, column)`` coordinate per qubit, indexed by qubit, or ``None`` if this size is
        not one of the conventionally drawn ones.
    """
    if num_qubits == 5:
        return [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1)]

    if num_qubits == 7:
        return _heavy_hex([range(3), [1], range(3)])

    if num_qubits == 15:
        return [(0, column) for column in range(7)] + [(1, column) for column in range(7, -1, -1)]

    if num_qubits == 16:
        return _heavy_hex([[3], range(7), [1, 5], range(1, 6), [3]], row_major=False)

    if num_qubits == 20:
        return _heavy_hex([range(5)] * 4)

    if num_qubits == 27:
        return _heavy_hex([[3, 7], range(10), [1, 5, 9], range(1, 11), [3, 7]], row_major=False)

    if num_qubits == 28:
        return _heavy_hex([range(2, 7), [2, 6], range(9), [0, 4, 8], range(9)])

    if num_qubits == 53:
        first = [range(9), [0, 4, 8]]
        second = [range(9), [2, 6]]
        return _heavy_hex([range(2, 7), [2, 6]] + first + second + first + second)

    if num_qubits == 65:
        repeated = [range(11), [2, 6, 10]]
        rows = [range(10), [0, 4, 8]] + repeated + [range(11), [0, 4, 8]] + repeated
        return _heavy_hex(rows + [range(1, 11)])

    if num_qubits == 120:
        return list(product(range(12), range(10)))

    if num_qubits == 127:
        first = [range(15), [2, 6, 10, 14]]
        second = [range(15), [0, 4, 8, 12]]
        rows = [range(14), [0, 4, 8, 12]] + first + second + first + second + first
        return _heavy_hex(rows + [range(1, 15)])

    if num_qubits == 133:
        first = [range(15), [0, 4, 8, 12]]
        second = [range(15), [2, 6, 10, 14]]
        return _heavy_hex((first + second) * 3 + first)

    if num_qubits == 156:
        first = [range(16), [3, 7, 11, 15]]
        second = [range(16), [1, 5, 9, 13]]
        return _heavy_hex((first + second) * 3 + first + [range(16)])

    return None


def _heavy_hex(rows: Sequence[Iterable[int]], row_major: bool = True) -> list[tuple[int, int]]:
    """Return the coordinates of a heavy-hex lattice described row by row.

    Args:
        rows: The column indices occupied by each row, ordered from the top down.
        row_major: Whether qubits are numbered along rows or down columns.

    Returns:
        One coordinate per qubit, indexed by qubit.
    """
    coordinates = [(row_index, column) for row_index, row in enumerate(rows) for column in row]
    if not row_major:
        coordinates.sort(key=lambda coordinate: (coordinate[1], coordinate[0]))
    return coordinates
