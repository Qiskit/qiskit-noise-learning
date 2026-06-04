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
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CZGate, XGate
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.experiment_builder.experiment import Experiment
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import FidelityIndex, Path


@pytest.fixture()
def gate_set_1q():
    model_gate_set = ModelGateSet(1)
    ident = Clifford(QuantumCircuit(1))
    model_gate_set.add_gate(ModelGate("P", [((0,), ident)], prep_idxs=range(1)))
    model_gate_set.add_gate(ModelGate("M", [((0,), ident)], meas_idxs=range(1)))
    # Clifford maps X -> -Y, Y -> Z, Z -> -X
    model_gate_set.add_gate(
        ModelGate("L0", [((0,), Clifford([[True, True, True], [True, False, True]]))])
    )
    model_gate_set.add_gate(ModelGate("L1", [((0,), Clifford(XGate()))]))
    return model_gate_set


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


@pytest.fixture()
def unbound_path_ix(gate_set_cz):
    return Path(
        start_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["P"],
                in_pauli=QubitSparsePauli("II"),
                out_pauli=QubitSparsePauli("IZ"),
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("IX"),
                out_pauli=QubitSparsePauli("ZX"),
            ),
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("ZX"),
                out_pauli=QubitSparsePauli("IX"),
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["M"],
                in_pauli=QubitSparsePauli("IZ"),
                out_pauli=QubitSparsePauli("II"),
            )
        ],
    )


@pytest.fixture()
def unbound_path_xi(gate_set_cz):
    return Path(
        start_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["P"],
                in_pauli=QubitSparsePauli("II"),
                out_pauli=QubitSparsePauli("ZI"),
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("XI"),
                out_pauli=QubitSparsePauli("XZ"),
            ),
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("XZ"),
                out_pauli=QubitSparsePauli("XI"),
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["M"],
                in_pauli=QubitSparsePauli("ZI"),
                out_pauli=QubitSparsePauli("II"),
            )
        ],
    )


@pytest.fixture()
def unbound_path_xx(gate_set_cz):
    return Path(
        start_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["P"],
                in_pauli=QubitSparsePauli("II"),
                out_pauli=QubitSparsePauli("ZZ"),
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("XX"),
                out_pauli=QubitSparsePauli("YY"),
            ),
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("YY"),
                out_pauli=QubitSparsePauli("XX"),
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["M"],
                in_pauli=QubitSparsePauli("ZZ"),
                out_pauli=QubitSparsePauli("II"),
            )
        ],
    )


@pytest.fixture()
def simple_experiment_with_paths(gate_set_cz, unbound_path_ix, unbound_path_xi):
    """An experiment with a fidelity model and two unbound paths."""
    return Experiment(
        fidelity_model=gate_set_cz,
        paths=[unbound_path_ix, unbound_path_xi],
    )


@pytest.fixture()
def experiment_with_sequences(gate_set_cz, unbound_path_ix, unbound_path_xi):
    """An experiment with paths, instruction sequences, relations, and multipliers."""
    seq_ix = unbound_path_ix.to_instruction_sequence()
    seq_xi = unbound_path_xi.to_instruction_sequence()
    return Experiment(
        fidelity_model=gate_set_cz,
        paths=[unbound_path_ix, unbound_path_xi],
        instruction_sequences=[seq_ix, seq_xi],
        relations={(0, 0), (1, 1)},
        randomization_multipliers=[1, 1],
    )
