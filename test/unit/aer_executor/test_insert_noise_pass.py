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

"""Unit tests for InsertNoisePass."""

import warnings

import numpy as np
import pytest
from qiskit.circuit import Barrier, QuantumCircuit
from qiskit.quantum_info import PauliLindbladMap
from qiskit.transpiler import PassManager

from qiskit_noise_learning.aer_executor.insert_noise_pass import InsertNoisePass


def _circuit_with_barrier(n_qubits: int, label: str) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    qc.append(Barrier(n_qubits, label=label), list(range(n_qubits)))
    return qc


def _noise_error_ops(circuit: QuantumCircuit) -> list:
    # PauliLindbladError is wrapped in QuantumChannelInstruction when going through the DAG.
    return [instr.operation for instr in circuit.data if instr.operation.name == "quantum_channel"]


_NOISE_DICT = {"r0": PauliLindbladMap.from_list([("XI", 0.1), ("IX", 0.2)])}


def test_noise_after_true_injects_at_r_barriers():
    qc = _circuit_with_barrier(2, "R0@tag=r0")
    result = PassManager([InsertNoisePass(noise_dict=_NOISE_DICT, noise_after=True)]).run(qc)
    assert len(_noise_error_ops(result)) == 1


def test_noise_after_false_injects_at_m_barriers():
    qc = _circuit_with_barrier(2, "M0@tag=r0")
    result = PassManager([InsertNoisePass(noise_dict=_NOISE_DICT, noise_after=False)]).run(qc)
    assert len(_noise_error_ops(result)) == 1


def test_noise_after_true_ignores_m_barriers():
    qc = _circuit_with_barrier(2, "M0@tag=r0")
    result = PassManager([InsertNoisePass(noise_dict=_NOISE_DICT, noise_after=True)]).run(qc)
    assert len(_noise_error_ops(result)) == 0


def test_noise_after_false_ignores_r_barriers():
    qc = _circuit_with_barrier(2, "R0@tag=r0")
    result = PassManager([InsertNoisePass(noise_dict=_NOISE_DICT, noise_after=False)]).run(qc)
    assert len(_noise_error_ops(result)) == 0


def test_noise_scale_multiplies_rates():
    qc = _circuit_with_barrier(2, "R0@tag=r0")
    noise_dict = {"r0": PauliLindbladMap.from_list([("XI", 0.1)])}

    result_1x = PassManager([InsertNoisePass(noise_dict=noise_dict, noise_scale=1.0)]).run(qc)
    result_3x = PassManager([InsertNoisePass(noise_dict=noise_dict, noise_scale=3.0)]).run(qc)

    # PauliLindbladError is stored as ._quantum_error inside QuantumChannelInstruction.
    np.testing.assert_allclose(_noise_error_ops(result_1x)[0]._quantum_error.rates, [0.1])  # noqa: SLF001
    np.testing.assert_allclose(_noise_error_ops(result_3x)[0]._quantum_error.rates, [0.3])  # noqa: SLF001


def test_warn_absent_true_emits_warning():
    qc = _circuit_with_barrier(2, "R0@tag=unknown")
    noise_dict = {"r0": PauliLindbladMap.from_list([("XI", 0.1)])}

    with pytest.warns(UserWarning, match="unknown"):
        PassManager([InsertNoisePass(noise_dict=noise_dict, warn_absent=True)]).run(qc)


def test_warn_absent_false_suppresses_warning():
    qc = _circuit_with_barrier(2, "R0@tag=unknown")
    noise_dict = {"r0": PauliLindbladMap.from_list([("XI", 0.1)])}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PassManager([InsertNoisePass(noise_dict=noise_dict, warn_absent=False)]).run(qc)


def test_none_noise_dict_is_noop():
    qc = _circuit_with_barrier(2, "R0@tag=r0")
    result = PassManager([InsertNoisePass(noise_dict=None)]).run(qc)
    assert len(_noise_error_ops(result)) == 0


def test_missing_tag_leaves_barrier_intact():
    qc = _circuit_with_barrier(2, "R0@tag=unknown")
    noise_dict = {"r0": PauliLindbladMap.from_list([("XI", 0.1)])}

    result = PassManager([InsertNoisePass(noise_dict=noise_dict, warn_absent=False)]).run(qc)

    assert len(_noise_error_ops(result)) == 0
    assert result.count_ops().get("barrier", 0) == 1
