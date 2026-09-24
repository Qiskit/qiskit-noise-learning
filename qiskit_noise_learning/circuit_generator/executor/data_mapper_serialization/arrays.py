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
    """Concatenate rows of differing lengths into one array, plus each row's length.

    Two things about the transport shape this. Each array is compressed on its own, so one long
    array beats one array per row. And row *lengths* are written rather than offsets into the
    concatenation: lengths repeat, often taking only one or two distinct values, where offsets climb
    monotonically and share no structure. Measured over a 196-qubit payload, the offsets form cost
    158.9 KB against 4.8 KB for lengths.

    Args:
        rows: The rows to concatenate.
        dtype: The dtype of the concatenated array.

    Returns:
        A tuple of the concatenated values and the length of each row.

    Examples:
        >>> from qiskit_noise_learning.circuit_generator.executor.arrays import pack_ragged
        >>> flat, lengths = pack_ragged([[1, 2, 3], [], [4]])
        >>> flat.tolist(), lengths.tolist()
        ([1, 2, 3, 4], [3, 0, 1])
    """
    lengths = np.array([len(row) for row in rows], dtype=IDX)
    if not rows:
        return np.empty(0, dtype=dtype), lengths
    return np.concatenate([np.asarray(row, dtype=dtype) for row in rows]).astype(dtype), lengths


def unpack_ragged(flat: NDArray[Any], lengths: NDArray[IDX]) -> list[NDArray[Any]]:
    """Split a concatenated array back into the rows that :func:`pack_ragged` was given.

    Args:
        flat: The concatenated values.
        lengths: The length of each row.

    Returns:
        The rows, as views onto ``flat``.
    """
    bounds = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=bounds[1:])
    return [flat[start:stop] for start, stop in zip(bounds[:-1], bounds[1:])]


def pack_paulis(
    paulis: Sequence[QubitSparsePauli],
) -> tuple[NDArray[np.uint8], NDArray[IDX], NDArray[IDX]]:
    """Write Paulis as concatenated term and qubit-index arrays, plus each Pauli's term count.

    The qubit count is not written, being the same for every Pauli in a payload and recorded once.

    Args:
        paulis: The Paulis to write.

    Returns:
        A tuple of the concatenated Pauli terms, their qubit indices, and each Pauli's length.

    Examples:
        >>> from qiskit.quantum_info import QubitSparsePauli
        >>> from qiskit_noise_learning.circuit_generator.executor.arrays import pack_paulis
        >>> terms, indices, lengths = pack_paulis([QubitSparsePauli.from_label("IXZ")])
        >>> terms.tolist(), indices.tolist(), lengths.tolist()
        ([1, 2], [0, 1], [2])
    """
    terms, lengths = pack_ragged([pauli.paulis for pauli in paulis], dtype=np.uint8)
    indices, _ = pack_ragged([pauli.indices for pauli in paulis], dtype=IDX)
    return terms, indices, lengths


def unpack_paulis(
    terms: NDArray[np.uint8],
    indices: NDArray[IDX],
    lengths: NDArray[IDX],
    num_qubits: int,
) -> list[QubitSparsePauli]:
    """Rebuild the Paulis that :func:`pack_paulis` was given.

    Args:
        terms: The concatenated Pauli terms.
        indices: The concatenated qubit indices.
        lengths: The length of each Pauli.
        num_qubits: The qubit count every rebuilt Pauli is given.

    Returns:
        The Paulis.
    """
    bounds = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=bounds[1:])
    return [
        QubitSparsePauli.from_raw_parts(num_qubits, terms[start:stop], indices[start:stop])
        for start, stop in zip(bounds[:-1], bounds[1:])
    ]
