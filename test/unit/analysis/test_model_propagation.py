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

import numpy as np

from qiskit_noise_learning.analysis import propagate_model_data
from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.math import EnumeratedParameterSpace, IndexedVector, LinearMap


class _ExampleMap(LinearMap):
    """A linear map defined by an explicit mapping from output index to sparse row."""

    def __init__(self, rows: dict[str, IndexedVector]):
        self._rows = rows
        inputs = frozenset(idx for row in rows.values() for idx in row)
        super().__init__(
            input_space=EnumeratedParameterSpace(inputs),
            output_space=EnumeratedParameterSpace(frozenset(rows)),
        )

    def row(self, output_index: str) -> IndexedVector:
        return self._rows[output_index]


def _model_data():
    # params i0, i1 with values 5, 7, diagonal covariance diag(2, 3), and disjoint time windows.
    return ModelData.from_arrays(
        parameter_indices=["i0", "i1"],
        parameter_values=np.array([5.0, 7.0]),
        covariance=np.array([[2.0, 0.0], [0.0, 3.0]]),
        time_lbs=np.array(["2024-01-01", "2024-01-03"], dtype="datetime64[s]"),
        time_ubs=np.array(["2024-01-02", "2024-01-05"], dtype="datetime64[s]"),
    )


def test_propagates_values():
    # o0 = i0 + i1, o1 = 2 * i1
    linear_map = _ExampleMap(
        {
            "o0": IndexedVector({"i0": 1.0, "i1": 1.0}),
            "o1": IndexedVector({"i1": 2.0}),
        }
    )

    result = propagate_model_data(linear_map, _model_data(), ["o0", "o1"])

    labels = list(result.dataset.coords["parameter"].values)
    assert labels == ["o0", "o1"]
    values = dict(zip(labels, result.dataset["parameter_values"].values))
    assert values == {"o0": 12.0, "o1": 14.0}


def test_propagates_covariance():
    linear_map = _ExampleMap(
        {
            "o0": IndexedVector({"i0": 1.0, "i1": 1.0}),
            "o1": IndexedVector({"i1": 2.0}),
        }
    )

    result = propagate_model_data(linear_map, _model_data(), ["o0", "o1"])

    # M = [[1, 1], [0, 2]], cov = diag(2, 3) => M cov M^T = [[5, 6], [6, 12]]
    cov = result.dataset["covariance"].values
    assert np.allclose(cov, np.array([[5.0, 6.0], [6.0, 12.0]]))


def test_propagates_time_bounds_as_contributor_envelope():
    linear_map = _ExampleMap(
        {
            "o0": IndexedVector({"i0": 1.0, "i1": 1.0}),
            "o1": IndexedVector({"i1": 2.0}),
        }
    )

    result = propagate_model_data(linear_map, _model_data(), ["o0", "o1"])

    lbs = dict(zip(result.dataset.coords["parameter"].values, result.dataset["time_lbs"].values))
    ubs = dict(zip(result.dataset.coords["parameter"].values, result.dataset["time_ubs"].values))
    # o0 draws on both inputs -> widest envelope; o1 only on i1
    assert lbs["o0"] == np.datetime64("2024-01-01", "s")
    assert ubs["o0"] == np.datetime64("2024-01-05", "s")
    assert lbs["o1"] == np.datetime64("2024-01-03", "s")
    assert ubs["o1"] == np.datetime64("2024-01-05", "s")
