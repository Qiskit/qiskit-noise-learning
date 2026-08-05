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

from itertools import combinations

import numpy as np
import pytest
from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.visualizations._qubit_coordinates import qubit_coordinates

#: Every device size drawn in a conventional layout, so that a size dropped from the tables, or one
#: whose table stops covering all of its qubits, fails here.
_CONVENTIONAL_SIZES = (5, 7, 15, 16, 20, 27, 28, 53, 65, 120, 127, 133, 156)

#: How close two qubits of a derived layout may end up, restated from the module under test. The
#: figure draws arcs of radius 0.24 around every node, so nodes closer than twice that have
#: overlapping arcs; this floor sits just above.
_MIN_SEPARATION = 0.55

#: Tolerance for a length the layout scales to a target rather than computes exactly.
_ATOL = 1e-9


def _min_separation(coords):
    """The distance between the closest two qubits of a layout."""
    points = np.array(coords, dtype=float)
    return min(
        float(np.hypot(*(points[left] - points[right])))
        for left, right in combinations(range(len(points)), 2)
    )


def _median_edge_length(coords, coupling_map):
    """The typical distance between coupled qubits of a layout."""
    points = np.array(coords, dtype=float)
    return float(
        np.median(
            [float(np.hypot(*(points[left] - points[right]))) for left, right in coupling_map]
        )
    )


def _edgeless_map(num_qubits):
    """A coupling map covering ``num_qubits`` qubits with no couplings at all: a device whose gate
    set has no two-qubit gate."""
    coupling_map = CouplingMap()
    for qubit in range(num_qubits):
        coupling_map.add_physical_qubit(qubit)
    return coupling_map


# --------------------------------------------------------------------------------------------------
# conventional layouts
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_qubits", _CONVENTIONAL_SIZES)
def test_conventional_layout_places_every_qubit_once(num_qubits):
    """A conventional size is laid out with no coupling map at all, and its table gives one distinct
    finite position per qubit -- a transcription that dropped or repeated a qubit would leave the
    figure indexing positions that belong to another one."""
    coords = qubit_coordinates(num_qubits)
    assert len(coords) == num_qubits
    assert len(set(coords)) == num_qubits
    assert np.isfinite(np.array(coords, dtype=float)).all()


def test_conventional_five_qubit_layout_is_the_familiar_star():
    """The 5-qubit layout is the recognizable plus sign: qubit 1 at the centre, 0/2/3 across the
    middle row, 4 below."""
    assert qubit_coordinates(5) == [(0.0, -1.0), (1.0, 0.0), (1.0, -1.0), (2.0, -1.0), (1.0, -2.0)]


def test_conventional_rows_run_downwards():
    """The tables are written in ``(row, column)`` with row 0 at the top, so the returned positions
    put the first row highest: a 20-qubit device is 4 rows of 5, with ``y`` from 0 down to -3."""
    ys = [y for _, y in qubit_coordinates(20)]
    xs = [x for x, _ in qubit_coordinates(20)]
    assert (max(ys), min(ys)) == (0.0, -3.0)
    assert (min(xs), max(xs)) == (0.0, 4.0)


def test_conventional_layout_wins_over_a_coupling_map():
    """A device whose size has a conventional layout gets it even when a coupling graph is available
    to derive one from: the conventional picture is the one a reader recognizes."""
    assert qubit_coordinates(5, CouplingMap.from_line(5)) == qubit_coordinates(5)


# --------------------------------------------------------------------------------------------------
# derived layouts
# --------------------------------------------------------------------------------------------------


def test_unconventional_size_needs_a_coupling_map():
    """A size the tables do not cover has no layout at all without a graph to derive one from."""
    with pytest.raises(ValueError, match="no coupling map"):
        qubit_coordinates(4)


@pytest.mark.parametrize(
    "num_qubits, coupling_map",
    [
        pytest.param(4, CouplingMap.from_line(6), id="map-larger"),
        pytest.param(5, CouplingMap.from_line(2), id="map-smaller-conventional-size"),
    ],
)
def test_coupling_map_of_the_wrong_size_raises(num_qubits, coupling_map):
    """A coupling map describing a different device is rejected rather than half-used. The second
    case is a conventional size, where the layout does not consult the map: the disagreement is
    still a caller error, and absorbing it would hand back positions for another device."""
    with pytest.raises(ValueError, match="but the device has"):
        qubit_coordinates(num_qubits, coupling_map)


def test_derived_layout_places_every_qubit_once():
    """A size the tables do not cover is laid out from its coupling graph, one distinct finite
    position per qubit."""
    coords = qubit_coordinates(4, CouplingMap.from_line(4))
    assert len(coords) == 4
    assert len(set(coords)) == 4
    assert np.isfinite(np.array(coords, dtype=float)).all()


def test_derived_layout_is_reproducible():
    """The spring layout is seeded, so drawing the same device twice gives the same picture. Without
    this a docs build would produce a figure that moved with no change to its source."""
    assert qubit_coordinates(4, CouplingMap.from_line(4)) == qubit_coordinates(
        4, CouplingMap.from_line(4)
    )


def test_derived_coupled_qubits_sit_about_one_unit_apart():
    """A derived layout is rescaled onto the same unit spacing the conventional tables use, since
    every drawing constant -- node radius, arc radius, inches per unit -- is a length in that unit.
    A spring layout's own scale shrinks as qubits are added, which is what this corrects."""
    coupling_map = CouplingMap.from_grid(3, 3)
    coords = qubit_coordinates(9, coupling_map)
    assert np.isclose(_median_edge_length(coords, coupling_map.get_edges()), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "num_qubits, coupling_map",
    [
        pytest.param(4, CouplingMap.from_line(4), id="line-4"),
        pytest.param(6, CouplingMap.from_full(6), id="full-6"),
        pytest.param(9, CouplingMap([[0, q] for q in range(1, 9)]), id="star-9"),
        # A long line is where a spring layout pulls some pair much tighter than its median edge,
        # which is the case the separation floor exists for.
        pytest.param(60, CouplingMap.from_line(60), id="line-60"),
    ],
)
def test_derived_layout_keeps_every_pair_of_qubits_apart(num_qubits, coupling_map):
    """No two qubits of a derived layout end up closer than the separation floor, whatever the graph
    -- two nodes nearer than that have overlapping arcs, and the arcs are what carry the gates."""
    assert _min_separation(qubit_coordinates(num_qubits, coupling_map)) >= _MIN_SEPARATION - _ATOL


def test_device_with_no_couplings_is_still_laid_out():
    """A device with no two-qubit gates has no edge length to scale by, so the separation floor is
    the whole adjustment -- but it still gets a layout rather than an error."""
    coords = qubit_coordinates(4, _edgeless_map(4))
    assert len(set(coords)) == 4
    assert _min_separation(coords) >= _MIN_SEPARATION - _ATOL
