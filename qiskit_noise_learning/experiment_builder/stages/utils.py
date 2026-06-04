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

from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet


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


def default_gates(gate_set: ModelGateSet) -> list[ModelGate]:
    """Get all non-SPAM gates."""
    return [gate for _, gate in gate_set.items() if not gate.prep_idxs and not gate.meas_idxs]
