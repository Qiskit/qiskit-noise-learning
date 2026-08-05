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

The layouts are the ones IBM devices are conventionally drawn in, transcribed from
``qiskit_ibm_runtime.visualization.embeddings`` (Apache 2.0, the same project family).  They are
copied rather than imported to avoid dependence on a private member, as well as its dependencies
(e.g. plotly).
"""

from collections.abc import Iterable, Sequence
from itertools import product


def qubit_coordinates(num_qubits: int) -> list[tuple[int, int]]:
    """Return the ``(row, column)`` position of each qubit of a ``num_qubits``-qubit device.

    Args:
        num_qubits: Number of qubits in the device.

    Returns:
        One coordinate per qubit, indexed by qubit.

    Raises:
        ValueError: If no layout is known for a device of this size.
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

    raise ValueError(f"No known layout for a {num_qubits}-qubit device.")


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
