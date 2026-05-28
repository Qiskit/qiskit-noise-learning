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

from itertools import product

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, QubitSparsePauli, QubitSparsePauliList
from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.experiment_builder import (
    depth0_path_generator,
    depth1_path_generator,
    even_depth_path_generator,
    even_depth_vanilla_path_generator,
    standard_vanilla_path_generator,
)
from qiskit_noise_learning.experiment_builder.experiment_generators import (
    generate_vanilla_instruction_sequences,
    yield_matching_paths,
)
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import FidelityIndex, Path


@pytest.fixture()
def gate_set_cz():
    model_gate_set = ModelGateSet(2)
    model_gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    model_gate_set.add_gate(
        ModelGate("P", [((0, 1), Clifford(QuantumCircuit(2)))], prep_idxs=range(2))
    )
    model_gate_set.add_gate(
        ModelGate("M", [((0, 1), Clifford(QuantumCircuit(2)))], meas_idxs=range(2))
    )
    return model_gate_set


def test_depth0_path_generator(gate_set_cz):
    path_iterator = depth0_path_generator(
        prep_gate=gate_set_cz["P"],
        meas_gate=gate_set_cz["M"],
        indices_list=[[0], [1]],
        num_qubits=2,
    )
    expected_iterator = [
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0]),
                    )
                ],
                repeatable_fragment=[],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0]),
                    )
                ],
                depth=0,
            ),
            None,
        ),
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([1]),
                    )
                ],
                repeatable_fragment=[],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([1]),
                    )
                ],
                depth=0,
            ),
            None,
        ),
    ]

    assert list(path_iterator) == list(expected_iterator)


def test_depth1_path_generator(gate_set_cz):
    path_iterator = depth1_path_generator(
        prep_gate=gate_set_cz["P"],
        meas_gate=gate_set_cz["M"],
        gate=gate_set_cz["CZ"],
        input_paulis=QubitSparsePauliList(["IZ", "IX", "XX"]),
    )
    expected_iterator = [
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0]),
                    ),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("IZ")),
                ],
                repeatable_fragment=[],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0]),
                    )
                ],
                depth=0,
            ),
            None,
        ),
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0]),
                    ),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("ZX")),
                ],
                repeatable_fragment=[],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0, 1]),
                    )
                ],
                depth=0,
            ),
            None,
        ),
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0, 1]),
                    ),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("YY")),
                ],
                repeatable_fragment=[],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0, 1]),
                    )
                ],
                depth=0,
            ),
            None,
        ),
    ]

    assert list(path_iterator) == list(expected_iterator)


def test_even_depth_path_generator(gate_set_cz):
    path_iterator = even_depth_path_generator(
        prep_gate=gate_set_cz["P"],
        meas_gate=gate_set_cz["M"],
        gate=gate_set_cz["CZ"],
        input_paulis=QubitSparsePauliList(["IZ", "IX", "IY", "XX"]),
    )
    expected_iterator = [
        # IZ -> IZ
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0]),
                    ),
                ],
                repeatable_fragment=[
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("IZ")),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("IZ")),
                ],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0]),
                    )
                ],
            ),
            None,
        ),
        # IX -> IX
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0]),
                    ),
                ],
                repeatable_fragment=[
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("ZX")),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("IX")),
                ],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0]),
                    )
                ],
            ),
            None,
        ),
        # IX -> IY
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0]),
                    ),
                ],
                repeatable_fragment=[
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("ZX")),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("IY")),
                ],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0]),
                    )
                ],
            ),
            None,
        ),
        # IY -> IX
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0]),
                    ),
                ],
                repeatable_fragment=[
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("ZY")),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("IX")),
                ],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0]),
                    )
                ],
            ),
            None,
        ),
        # IY -> IY
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0]),
                    ),
                ],
                repeatable_fragment=[
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("ZY")),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("IY")),
                ],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0]),
                    )
                ],
            ),
            None,
        ),
        # XX -> XX
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0, 1]),
                    ),
                ],
                repeatable_fragment=[
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("YY")),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("XX")),
                ],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0, 1]),
                    )
                ],
            ),
            None,
        ),
    ]
    assert list(path_iterator) == list(expected_iterator)


def test_even_depth_vanilla_path_generator(gate_set_cz):
    path_iterator = even_depth_vanilla_path_generator(
        prep_gate=gate_set_cz["P"],
        meas_gate=gate_set_cz["M"],
        gate=gate_set_cz["CZ"],
        input_paulis=QubitSparsePauliList(["IZ", "IX", "IY", "XX"]),
    )
    expected_iterator = [
        # IZ -> IZ
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0]),
                    ),
                ],
                repeatable_fragment=[
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("IZ")),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("IZ")),
                ],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0]),
                    )
                ],
            ),
            None,
        ),
        # IX -> IX
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0]),
                    ),
                ],
                repeatable_fragment=[
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("ZX")),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("IX")),
                ],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0]),
                    )
                ],
            ),
            None,
        ),
        # IY -> IY
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0]),
                    ),
                ],
                repeatable_fragment=[
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("ZY")),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("IY")),
                ],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0]),
                    )
                ],
            ),
            None,
        ),
        # XX -> XX
        (
            Path(
                start_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["P"],
                        pauli=QubitSparsePauli("II"),
                        out_bit_indices=frozenset([0, 1]),
                    ),
                ],
                repeatable_fragment=[
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("YY")),
                    FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("XX")),
                ],
                end_fragment=[
                    FidelityIndex(
                        gate=gate_set_cz["M"],
                        pauli=QubitSparsePauli("II"),
                        in_bit_indices=frozenset([0, 1]),
                    )
                ],
            ),
            None,
        ),
    ]
    assert list(path_iterator) == list(expected_iterator)


def test_sufficient_bases_ring():
    """Test that the generation functions give the minimal number of bases."""
    edges = [(idx, (idx + 1) % 11) for idx in range(11)]
    coupling_map = CouplingMap(edges)

    gate_idxs = [(idx, idx + 1) for idx in range(0, 10, 2)]
    paulis = ["".join(p for p in comb) for comb in product("IXZY", repeat=2)]
    input_paulis = QubitSparsePauliList.from_sparse_list(
        [(pauli, idxs) for idxs in gate_idxs for pauli in paulis], num_qubits=11
    )
    gate = ModelGate("CZ", [(idxs, Clifford(CZGate())) for idxs in gate_idxs])
    prep = ModelGate("P", [(tuple(range(11)), Clifford(QuantumCircuit(11)))], prep_idxs=range(11))
    meas = ModelGate("M", [(tuple(range(11)), Clifford(QuantumCircuit(11)))], meas_idxs=range(11))

    instruction_sequences = generate_vanilla_instruction_sequences(prep, meas, gate, coupling_map)
    paths = list(even_depth_vanilla_path_generator(prep, meas, gate, input_paulis))
    matched_paths = list(yield_matching_paths(paths, instruction_sequences))

    assert len(instruction_sequences) == 9
    assert len(matched_paths) == len(paths)

    other_matched_paths = list(
        standard_vanilla_path_generator(prep, meas, gate, input_paulis, coupling_map)
    )

    assert other_matched_paths == matched_paths


def test_yield_matching_paths_errors(gate_set_cz):
    path = Path(
        start_fragment=[],
        repeatable_fragment=[FidelityIndex(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("IZ"))],
        end_fragment=[],
    )

    with pytest.raises(ValueError, match="a path that is not traversed"):
        list(yield_matching_paths([(path, None)], []))
