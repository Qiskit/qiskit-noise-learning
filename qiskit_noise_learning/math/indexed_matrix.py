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

"""IndexedMatrix"""

from collections.abc import Hashable, Mapping, Sequence
from typing import Generic, Self, TypeVar

import numpy as np

from .indexed_vector import IndexedVector

RowIndex = TypeVar("RowIndex", bound=Hashable)
ColumnIndex = TypeVar("ColumnIndex", bound=Hashable)
OtherColumnIndex = TypeVar("OtherColumnIndex", bound=Hashable)


class IndexedMatrix(Generic[RowIndex, ColumnIndex]):
    """A matrix with float entries and arbitrary row and column index data.

    Args:
        row_index_map: A mapping from row indices to the integer row axes of ``data``.
        column_index_map: A mapping from column indices to the integer column axes of ``data``.
        data: The array for the given row and column indices.

    Raises:
        ValueError: If the shape of ``data`` is inconsistent with the values of ``row_index_map``
            or ``column_index_map``.
    """

    def __init__(
        self,
        row_index_map: Mapping[RowIndex, int] | None = None,
        column_index_map: Mapping[ColumnIndex, int] | None = None,
        data: np.ndarray[float] | None = None,
    ):
        row_index_map = (
            dict() if row_index_map is None else {k: v for k, v in row_index_map.items()}
        )
        column_index_map = (
            dict() if column_index_map is None else {k: v for k, v in column_index_map.items()}
        )
        data = np.array([], dtype=float) if (data is None) else data

        # validate empty case
        if data.shape[0] == 0:
            if len(row_index_map) != 0 or len(column_index_map) != 0:
                raise ValueError(
                    "Non-empty row or column indices is inconsistent with data.shape == (0,)."
                )

        else:
            if set(row_index_map.values()) != set(range(data.shape[0])):
                raise ValueError(
                    "The values of row_index_map must be a permutation of range(data.shape[0])."
                )

            if set(column_index_map.values()) != set(range(data.shape[1])):
                raise ValueError(
                    "The values of column_index_map must be a permutation of range(data.shape[1])."
                )

        self._row_index_map = row_index_map
        self._column_index_map = column_index_map
        self._data = data
        self._rank = None

    @classmethod
    def from_index_lists(
        self,
        row_indices: Sequence[RowIndex],
        column_indices: Sequence[ColumnIndex],
        data: np.ndarray[float],
    ) -> Self:
        """Construct from ordered lists of row and column indices.

        Args:
            row_indices: The list of row indices for the row axes of ``data``.
            column_indices: The list of column indices for the column axes of ``data``.
            data: The data matrix.

        Returns:
            An :class:`IndexedMatrix` whose row and column index maps are built from ``row_indices``
            and ``column_indices``.
        """
        return IndexedMatrix(
            row_index_map={k: idx for idx, k in enumerate(row_indices)},
            column_index_map={k: idx for idx, k in enumerate(column_indices)},
            data=data,
        )

    @classmethod
    def from_rows(
        cls,
        row_indices: Sequence[RowIndex],
        rows: Sequence[IndexedVector[ColumnIndex]],
        tol: float = 1e-8,
    ) -> Self:
        """Construct from row indices and their sparse :class:`IndexedVector` rows.

        Args:
            row_indices: The index for each row.
            rows: The sparse rows, as :class:`IndexedVector` instances.
            tol: Tolerance below which row values are treated as ``0.0``.

        Returns:
            An :class:`IndexedMatrix` containing the (non-zero) rows.
        """
        matrix = cls()
        matrix.add_rows(row_indices=list(row_indices), rows=list(rows), tol=tol)
        return matrix

    @property
    def row_index_map(self) -> dict[RowIndex, int]:
        """Dictionary mapping row indices to the row axis integer of ``self.data``."""
        return self._row_index_map

    @property
    def column_index_map(self) -> dict[ColumnIndex, int]:
        """Dictionary mapping column indices to the column axis integer of ``self.data``."""
        return self._column_index_map

    @property
    def data(self) -> np.ndarray[float]:
        """The numerical data."""
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the matrix."""
        return self._data.shape

    @property
    def rank(self) -> int:
        """The rank of the matrix."""
        if self._rank is None:
            # handle empty cases
            if any(x == 0 for x in self._data.shape):
                self._rank = 0
            else:
                self._rank = np.linalg.matrix_rank(self._data)
        return self._rank

    def add_rows(
        self, row_indices: list[RowIndex], rows: list[IndexedVector[ColumnIndex]], tol=1e-8
    ):
        """Add rows to the matrix.

        Args:
            row_indices: A list of indices for the rows.
            rows: The list of rows.
            tol: Tolerance below which values in ``rows`` are assumed to be ``0.0``.

        Raises:
            ValueError: If any row index is duplicated in ``row_indices`` or is already present in
                this instance.
            ValueError: If the number of row indices does not match the number of rows.
        """

        if len(row_indices) != len(rows):
            raise ValueError(
                f"The number of row indices, '{len(row_indices)}', is not equal to the number of "
                f"rows, '{len(rows)}'."
            )

        # iterate through arguments once, building a list of non-empty rows, their corresponding
        # indices, and a list of the non-zero columns for each new row
        new_row_indices: list[RowIndex] = []
        new_rows: list[IndexedVector[ColumnIndex]] = []
        new_row_nonzero_columns: list[list[ColumnIndex]] = []
        new_column_count = 0
        for row_index, row in zip(row_indices, rows):
            if row_index in self._row_index_map:
                raise ValueError(f"Cannot add row with duplicate row index '{row_index}'.")

            current_row_nonzero_columns = []
            for column_index, value in row.items():
                if abs(value) > tol:
                    current_row_nonzero_columns.append(column_index)
                    if column_index not in self._column_index_map:
                        self._column_index_map[column_index] = len(self._column_index_map)
                        new_column_count += 1

            # if any non-trivial rows, add them
            if len(current_row_nonzero_columns) != 0:
                new_row_indices.append(row_index)
                new_rows.append(row)
                new_row_nonzero_columns.append(current_row_nonzero_columns)
                self._row_index_map[row_index] = len(self._row_index_map)

        # if no new non-zero rows exit early
        if len(new_rows) == 0:
            return

        # expand internal data array for any new columns
        if new_column_count != 0 and len(self._data) != 0:
            padded_data = np.append(
                self._data,
                np.zeros((self._data.shape[0], new_column_count), dtype=float),
                axis=1,
            )
        else:
            # if no new columns, or matrix is empty, do nothing
            padded_data = self._data

        # build array for new rows
        new_row_array = np.zeros((len(new_rows), len(self._column_index_map)), dtype=float)
        for array_row_idx, (row_index, row, nonzero_column_indices) in enumerate(
            zip(new_row_indices, new_rows, new_row_nonzero_columns)
        ):
            for column_index in nonzero_column_indices:
                new_row_array[array_row_idx, self._column_index_map[column_index]] = row[
                    column_index
                ]

        # update data
        self._data = (
            new_row_array
            if padded_data.shape == (0,)
            else np.append(padded_data, new_row_array, axis=0)
        )
        self._rank = None

    def linearly_independent_rows(self, tol=1e-8) -> Self:
        """Return a submatrix containing a maximal set of linearly independent rows.

        Rows are processed in order (according to the indices in ``self.row_index_map``): a row is
        kept if and only if it is linearly independent of all preceding kept rows. This guarantees
        earlier rows are always preferred.

        Args:
            tol: The tolerance for determining linear independence based on the norm of the
                component of a row orthogonal to the span of preceding kept rows.
        """
        n_rows, n_cols = self._data.shape
        selected_indices = []
        basis = np.empty((n_cols, 0), dtype=float)

        # identify first accepted row
        for idx, row in enumerate(self._data):
            norm = np.linalg.norm(row)
            if norm > tol:
                selected_indices.append(idx)
                basis = np.column_stack([basis, row / norm])
                break

        if len(selected_indices) == 0:
            return IndexedMatrix[RowIndex, ColumnIndex]()

        for idx in range(selected_indices[0] + 1, n_rows):
            row = self._data[idx]
            residual = row - basis @ (basis.T @ row)
            norm = np.linalg.norm(residual)
            if norm > tol:
                selected_indices.append(idx)
                basis = np.column_stack([basis, residual / norm])

        new_row_index_map = {}
        for row_index, array_idx in self._row_index_map.items():
            if array_idx in selected_indices:
                new_row_index_map[row_index] = selected_indices.index(array_idx)

        return IndexedMatrix[RowIndex, ColumnIndex](
            row_index_map=new_row_index_map,
            column_index_map=self._column_index_map.copy(),
            data=self._data[selected_indices],
        )

    def copy(self) -> Self:
        """Return a copy of self."""
        return IndexedMatrix(
            row_index_map=self._row_index_map.copy(),
            column_index_map=self.column_index_map.copy(),
            data=self._data.copy(),
        )

    def transpose(self) -> "IndexedMatrix[ColumnIndex, RowIndex]":
        """Return the transpose, swapping row and column indices."""
        if self._data.shape[0] == 0:
            return IndexedMatrix[ColumnIndex, RowIndex]()
        return IndexedMatrix[ColumnIndex, RowIndex](
            row_index_map=self._column_index_map.copy(),
            column_index_map=self._row_index_map.copy(),
            data=self._data.T.copy(),
        )

    @property
    def T(self) -> "IndexedMatrix[ColumnIndex, RowIndex]":
        """The transpose, swapping row and column indices."""
        return self.transpose()

    def __matmul__(
        self,
        other: "IndexedMatrix[ColumnIndex, OtherColumnIndex] | IndexedVector[ColumnIndex]",
    ) -> "IndexedMatrix[RowIndex, OtherColumnIndex] | IndexedVector[RowIndex]":
        """Index-aware matrix product.

        Contraction is performed over the index labels shared between this matrix's columns and the
        rows (or entries) of ``other``; labels present in only one operand are treated as zero. The
        intended contract is that ``other``'s row indices (or the vector's indices) are this
        matrix's column indices.

        Args:
            other: An :class:`IndexedMatrix` whose rows are indexed by this matrix's column
                indices, or an :class:`IndexedVector` indexed by this matrix's column indices.

        Returns:
            An :class:`IndexedMatrix` (matrix product, indexed by this matrix's rows and ``other``'s
            columns) or :class:`IndexedVector` (matrix-vector product, indexed by this matrix's
            rows).
        """
        if isinstance(other, IndexedVector):
            return self._matmul_vector(other)
        if isinstance(other, IndexedMatrix):
            return self._matmul_matrix(other)
        return NotImplemented

    def _matmul_vector(self, vector: IndexedVector[ColumnIndex]) -> IndexedVector[RowIndex]:
        result = IndexedVector[RowIndex]()
        if self._data.shape[0] == 0:
            return result

        x = np.zeros(len(self._column_index_map), dtype=float)
        for column_index, data_idx in self._column_index_map.items():
            if column_index in vector:
                x[data_idx] = vector[column_index]

        y = self._data @ x
        for row_index, data_idx in self._row_index_map.items():
            result[row_index] = float(y[data_idx])
        return result

    def _matmul_matrix(
        self, other: "IndexedMatrix[ColumnIndex, OtherColumnIndex]"
    ) -> "IndexedMatrix[RowIndex, OtherColumnIndex]":
        n_rows = len(self._row_index_map)
        if n_rows == 0:
            return IndexedMatrix[RowIndex, OtherColumnIndex]()

        n_cols = len(other.column_index_map)
        shared = [k for k in self._column_index_map if k in other.row_index_map]
        if shared:
            left = self._data[:, [self._column_index_map[k] for k in shared]]
            right = other.data[[other.row_index_map[k] for k in shared], :]
            data = left @ right
        else:
            data = np.zeros((n_rows, n_cols), dtype=float)

        return IndexedMatrix[RowIndex, OtherColumnIndex](
            row_index_map=self._row_index_map.copy(),
            column_index_map=dict(other.column_index_map),
            data=data,
        )

    def __getitem__(
        self, index: RowIndex | Sequence[RowIndex]
    ) -> IndexedVector[ColumnIndex] | Self:
        """Return a row based on index, or a submatrix based on a sequence of row indices."""

        # attempt to treat index as a single entry
        if isinstance(index, Hashable) and (
            (row_idx := self._row_index_map.get(index, None)) is not None
        ):
            return IndexedVector[ColumnIndex](
                {
                    col_idx: self._data[row_idx][data_idx]
                    for col_idx, data_idx in self._column_index_map.items()
                }
            )

        if not isinstance(index, Sequence):
            raise KeyError(index)

        new_row_index_map = dict()
        data_idxs = np.empty(len(index), dtype=int)
        for idx, row_index in enumerate(index):
            new_row_index_map[row_index] = idx
            data_idxs[idx] = self._row_index_map[row_index]

        return IndexedMatrix[RowIndex, ColumnIndex](
            row_index_map=new_row_index_map,
            column_index_map=self._column_index_map.copy(),
            data=self._data[data_idxs],
        )

    def __eq__(self, other: Self) -> bool:
        if not isinstance(other, IndexedMatrix) or len(self._row_index_map) != len(
            other.row_index_map
        ):
            return False

        for row_index in self._row_index_map:
            if row_index not in other.row_index_map or self[row_index] != other[row_index]:
                return False

        return True
