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

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, QubitSparsePauli, QubitSparsePauliList

from qiskit_noise_learning.experiment_builder import (
    depth0_path_generator,
    depth1_path_generator,
    even_depth_pattern_generator,
    even_depth_vanilla_pattern_generator,
)
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import FidelityIndex, Path, PathPattern


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
                PathPattern(
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
                ),
                depth=0,
            ),
            None,
        ),
        (
            Path(
                PathPattern(
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
                ),
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
                PathPattern(
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
                ),
                depth=0,
            ),
            None,
        ),
        (
            Path(
                PathPattern(
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
                ),
                depth=0,
            ),
            None,
        ),
        (
            Path(
                PathPattern(
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
                ),
                depth=0,
            ),
            None,
        ),
    ]

    assert list(path_iterator) == list(expected_iterator)


def test_even_depth_pattern_generator(gate_set_cz):
    pattern_iterator = even_depth_pattern_generator(
        prep_gate=gate_set_cz["P"],
        meas_gate=gate_set_cz["M"],
        gate=gate_set_cz["CZ"],
        input_paulis=QubitSparsePauliList(["IZ", "IX", "IY", "XX"]),
    )
    expected_iterator = [
        # IZ -> IZ
        (
            PathPattern(
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
            PathPattern(
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
            PathPattern(
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
            PathPattern(
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
            PathPattern(
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
            PathPattern(
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
    assert list(pattern_iterator) == list(expected_iterator)


def test_even_depth_vanilla_pattern_generator(gate_set_cz):
    pattern_iterator = even_depth_vanilla_pattern_generator(
        prep_gate=gate_set_cz["P"],
        meas_gate=gate_set_cz["M"],
        gate=gate_set_cz["CZ"],
        input_paulis=QubitSparsePauliList(["IZ", "IX", "IY", "XX"]),
    )
    expected_iterator = [
        # IZ -> IZ
        (
            PathPattern(
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
            PathPattern(
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
            PathPattern(
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
            PathPattern(
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
    assert list(pattern_iterator) == list(expected_iterator)
