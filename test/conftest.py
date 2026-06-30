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

"""Pytest Configuration"""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, XGate
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionSequence,
    Path,
)

# enable scipy-style doctests
pytest_plugins = "scipy_doctest"


# --------------------------------------------------------------------------------------------------
# Gate sets
# --------------------------------------------------------------------------------------------------


@pytest.fixture()
def gate_set_1q():
    """A 1-qubit gate set with prep ``P``, measurement ``M``, and unitaries ``L0``/``L1``."""
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
    """A 2-qubit gate set with a ``CZ`` gate, pure preparation ``P``, and measurement ``M``."""
    model_gate_set = ModelGateSet(2)
    model_gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
    return model_gate_set


# --------------------------------------------------------------------------------------------------
# Factory fixtures
# --------------------------------------------------------------------------------------------------


@pytest.fixture()
def make_cz_path(gate_set_cz):
    """Return a builder for real, traversable :class:`~.Path` objects over ``gate_set_cz``.

    ``make(repeatable)`` builds the closed ``CZ`` orbit through one or more Pauli labels. A single
    label ``P`` is expanded to the orbit ``[P, CZ(P)]``; a list of labels is used as the explicit
    orbit, with each consecutive pair representing a transition. The resulting repeatable fragment
    is a closed loop, so the path is traversable at any depth.

    With ``spam=True`` (the default) the path gains a ``P`` start fragment and an ``M`` end fragment
    whose transition Paulis are ``Z`` on the support of the loop's starting Pauli. The returned path
    is unbound; bind a depth with ``.bind_at(depth)``.
    """

    def _make(repeatable, spam=True):
        cz = gate_set_cz["CZ"]

        if isinstance(repeatable, str):
            primary = QubitSparsePauli(repeatable)
            nodes = [primary, cz.clifford_propagate(pauli=primary, inverse=False)]
        else:
            nodes = [QubitSparsePauli(label) for label in repeatable]

        n = len(nodes)
        repeatable_fragment = [
            FidelityIndex.from_transition(gate=cz, in_pauli=nodes[i], out_pauli=nodes[(i + 1) % n])
            for i in range(n)
        ]

        start_fragment = []
        end_fragment = []
        if spam:
            support = sorted(int(i) for i in nodes[0].indices)
            z_pauli = QubitSparsePauli.from_sparse_label(
                ("Z" * len(support), support), num_qubits=2
            )
            start_fragment = [
                FidelityIndex.from_transition(
                    gate=gate_set_cz["P"],
                    in_pauli=QubitSparsePauli("II"),
                    out_pauli=z_pauli,
                )
            ]
            end_fragment = [
                FidelityIndex.from_transition(
                    gate=gate_set_cz["M"],
                    in_pauli=z_pauli,
                    out_pauli=QubitSparsePauli("II"),
                )
            ]
        return Path(
            start_fragment=start_fragment,
            repeatable_fragment=repeatable_fragment,
            end_fragment=end_fragment,
        )

    return _make


@pytest.fixture()
def make_instruction_sequence():
    """Return a builder ``(name="CZ", depth=1) -> InstructionSequence``.

    Builds a minimal real instruction sequence whose repeatable fragment is a single
    :class:`~.ApplyGate`. Vary ``name`` to obtain distinct sequences.
    """

    def _make(name="CZ", depth=1):
        return InstructionSequence(
            start_fragment=[],
            repeatable_fragment=[ApplyGate(name)],
            end_fragment=[],
            depth=depth,
        )

    return _make
