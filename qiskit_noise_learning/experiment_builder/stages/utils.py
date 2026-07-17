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

"""Utility functions."""

from qiskit.quantum_info import QubitSparsePauliList

from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.math import LinearMap
from qiskit_noise_learning.models import (
    contains_pauli_lindblad_model,
    split_pauli_lindblad_model,
)


def default_prep_gate(gate_set: ModelGateSet) -> ModelGate:
    """Get the default preparation gate (named ``"P"``)."""
    prep_gate = gate_set.get("P")
    if prep_gate is None:
        raise ValueError("Gate set does not contain a preparation gate named 'P'.")
    return prep_gate


def default_meas_gate(gate_set: ModelGateSet) -> ModelGate:
    """Get the default measurement gate (named ``"M"``)."""
    meas_gate = gate_set.get("M")
    if meas_gate is None:
        raise ValueError("Gate set does not contain a measurement gate named 'M'.")
    return meas_gate


def default_unitary_gates(gate_set: ModelGateSet) -> list[ModelGate]:
    """Get all unitary gates."""
    return [gate for _, gate in gate_set.items() if not gate.prep_idxs and not gate.meas_idxs]


def default_input_paulis(model: LinearMap) -> dict[str, QubitSparsePauliList]:
    """A default set of Paulis for path generators.

    Currently requires ``model`` to contain a :class:`~.PauliLindbladModel`, and will return the
    generators.

    Args:
        model: The fidelity model.

    Returns:
        A mapping from gate name to that gate's Pauli-Lindblad generators.

    Raises:
        ValueError: If ``model`` does not contain a :class:`~.PauliLindbladModel`.
    """
    if not contains_pauli_lindblad_model(model):
        raise ValueError(
            "Cannot determine default input Paulis: the fidelity model does not contain a "
            "PauliLindbladModel. Provide input_paulis explicitly."
        )
    return split_pauli_lindblad_model(model).model.generators
