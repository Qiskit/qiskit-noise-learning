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

"""Array plumbing shared by every version of the serialized payload."""

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import DTypeLike, NDArray
from qiskit.quantum_info import QubitSparsePauli

IDX: TypeAlias = np.uint32
"""The dtype every index and offset array is written with."""

Row: TypeAlias = Sequence[int] | NDArray[Any]
"""One row handed to :func:`pack_ragged`, either a sequence of indices or an array of them."""


def pack_ragged(rows: Sequence[Row], dtype: DTypeLike = IDX) -> tuple[NDArray[Any], NDArray[IDX]]:
    """Concatenate rows of differing lengths into one array, plus the offsets that split it.

    Writing one long array rather than one array per row matters: the transport compresses each
    array separately, so many short arrays compress far worse than a few long ones.

    Args:
        rows: The rows to concatenate.
        dtype: The dtype of the concatenated array.

    Returns:
        A tuple of the concatenated values and the offsets into them, the latter having one more
        entry than there were rows.

    Examples:
        >>> from qiskit_noise_learning.circuit_generator.executor.arrays import pack_ragged
        >>> flat, offsets = pack_ragged([[1, 2, 3], [], [4]])
        >>> flat.tolist(), offsets.tolist()
        ([1, 2, 3, 4], [0, 3, 3, 4])
    """
    offsets = np.zeros(len(rows) + 1, dtype=IDX)
    if not rows:
        return np.empty(0, dtype=dtype), offsets
    offsets[1:] = np.cumsum([len(row) for row in rows])
    return np.concatenate([np.asarray(row, dtype=dtype) for row in rows]).astype(dtype), offsets


def unpack_ragged(flat: NDArray[Any], offsets: NDArray[IDX]) -> list[NDArray[Any]]:
    """Split a concatenated array back into the rows that :func:`pack_ragged` was given.

    Args:
        flat: The concatenated values.
        offsets: The offsets into them.

    Returns:
        The rows, as views onto ``flat``.
    """
    return [flat[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]


def pack_paulis(
    paulis: Sequence[QubitSparsePauli],
) -> tuple[NDArray[np.uint8], NDArray[IDX], NDArray[IDX]]:
    """Write Paulis as concatenated term and qubit-index arrays, plus the offsets that split them.

    The qubit count is not written, being the same for every Pauli in a payload and recorded once.

    Args:
        paulis: The Paulis to write.

    Returns:
        A tuple of the concatenated Pauli terms, their qubit indices, and the offsets into both.

    Examples:
        >>> from qiskit.quantum_info import QubitSparsePauli
        >>> from qiskit_noise_learning.circuit_generator.executor.arrays import pack_paulis
        >>> terms, indices, offsets = pack_paulis([QubitSparsePauli.from_label("IXZ")])
        >>> terms.tolist(), indices.tolist(), offsets.tolist()
        ([1, 2], [0, 1], [0, 2])
    """
    terms, offsets = pack_ragged([pauli.paulis for pauli in paulis], dtype=np.uint8)
    indices, _ = pack_ragged([pauli.indices for pauli in paulis], dtype=IDX)
    return terms, indices, offsets


def unpack_paulis(
    terms: NDArray[np.uint8],
    indices: NDArray[IDX],
    offsets: NDArray[IDX],
    num_qubits: int,
) -> list[QubitSparsePauli]:
    """Rebuild the Paulis that :func:`pack_paulis` was given.

    Args:
        terms: The concatenated Pauli terms.
        indices: The concatenated qubit indices.
        offsets: The offsets into both.
        num_qubits: The qubit count every rebuilt Pauli is given.

    Returns:
        The Paulis.
    """
    return [
        QubitSparsePauli.from_raw_parts(num_qubits, terms[start:stop], indices[start:stop])
        for start, stop in zip(offsets[:-1], offsets[1:])
    ]
