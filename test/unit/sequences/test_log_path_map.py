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

import math
from itertools import chain

from qiskit.quantum_info import QubitSparsePauli

from qiskit_noise_learning.math import IndexedVector
from qiskit_noise_learning.models import LogFidelitySpace
from qiskit_noise_learning.sequences import FidelityIndex, LogPathMap, LogPathSpace, Path


def _expected_unbound_vector(path):
    vector = IndexedVector()
    for fidelity in path.repeatable_fragment:
        vector[fidelity] = vector.get(fidelity, 0.0) + 1.0
    return vector


def _expected_bound_vector(path):
    vector = _expected_unbound_vector(path)
    vector = path.depth * vector
    for fidelity in chain(path.start_fragment, path.end_fragment):
        vector[fidelity] = vector.get(fidelity, 0.0) + 1.0
    return vector


# --------------------------------------------------------------------------------------------------
# LogPathSpace
# --------------------------------------------------------------------------------------------------


def test_log_path_space_fidelity_space(gate_set_cz):
    fidelity_space = LogFidelitySpace(gate_set_cz)
    assert LogPathSpace(fidelity_space).fidelity_space is fidelity_space


def test_log_path_space_dim_is_infinite(gate_set_cz):
    assert LogPathSpace(LogFidelitySpace(gate_set_cz)).dim == math.inf


def test_log_path_space_contains_real_path(gate_set_cz, make_cz_path):
    space = LogPathSpace(LogFidelitySpace(gate_set_cz))
    assert make_cz_path("XI") in space


def test_log_path_space_rejects_path_from_other_gate_set(gate_set_cz, gate_set_1q):
    space = LogPathSpace(LogFidelitySpace(gate_set_cz))
    other_fidelity = FidelityIndex.from_gate(gate=gate_set_1q["L0"], pauli=QubitSparsePauli("X"))
    path = Path(start_fragment=[], repeatable_fragment=[other_fidelity], end_fragment=[])
    assert path not in space


def test_log_path_space_rejects_non_path(gate_set_cz):
    assert "not a path" not in LogPathSpace(LogFidelitySpace(gate_set_cz))


# --------------------------------------------------------------------------------------------------
# LogPathMap
# --------------------------------------------------------------------------------------------------


def test_log_path_map_spaces(gate_set_cz):
    fidelity_space = LogFidelitySpace(gate_set_cz)
    log_path_map = LogPathMap(fidelity_space)
    assert log_path_map.input_space is fidelity_space
    assert isinstance(log_path_map.output_space, LogPathSpace)


def test_log_path_map_rows_unbound(gate_set_cz, make_cz_path):
    log_path_map = LogPathMap(LogFidelitySpace(gate_set_cz))
    path = make_cz_path("XI")  # unbound

    row = log_path_map.rows([path])[path]

    assert row == _expected_unbound_vector(path)


def test_log_path_map_rows_bound(gate_set_cz, make_cz_path):
    log_path_map = LogPathMap(LogFidelitySpace(gate_set_cz))
    path = make_cz_path("XI").bind_at(2)

    row = log_path_map.rows([path])[path]

    assert row == _expected_bound_vector(path)
