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
from qiskit.quantum_info import DensityMatrix, PauliLindbladMap
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator

from qiskit_noise_learning.aer_executor.insert_noise_pass import InsertNoisePass


def _circuit_with_barrier(n_qubits: int, label: str) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    qc.append(Barrier(n_qubits, label=label), list(range(n_qubits)))
    return qc


def _noise_error_ops(circuit: QuantumCircuit) -> list:
    # PauliLindbladError is wrapped in QuantumChannelInstruction when going through the DAG.
    return [instr.operation for instr in circuit.data if instr.operation.name == "quantum_channel"]


@pytest.fixture
def noise_dict():
    return {"r0": PauliLindbladMap.from_list([("XI", 0.1), ("IX", 0.2)])}


def test_noise_after_true_injects_at_r_barriers(noise_dict):
    qc = _circuit_with_barrier(2, "R0@tag=r0")
    result = PassManager([InsertNoisePass(noise_dict=noise_dict, noise_after=True)]).run(qc)
    assert len(_noise_error_ops(result)) == 1


def test_noise_after_false_injects_at_m_barriers(noise_dict):
    qc = _circuit_with_barrier(2, "M0@tag=r0")
    result = PassManager([InsertNoisePass(noise_dict=noise_dict, noise_after=False)]).run(qc)
    assert len(_noise_error_ops(result)) == 1


def test_noise_after_true_ignores_m_barriers(noise_dict):
    qc = _circuit_with_barrier(2, "M0@tag=r0")
    result = PassManager([InsertNoisePass(noise_dict=noise_dict, noise_after=True)]).run(qc)
    assert len(_noise_error_ops(result)) == 0


def test_noise_after_false_ignores_r_barriers(noise_dict):
    qc = _circuit_with_barrier(2, "R0@tag=r0")
    result = PassManager([InsertNoisePass(noise_dict=noise_dict, noise_after=False)]).run(qc)
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


def test_noise_qubits_ordered_by_physical_index(noise_dict):
    # Barrier on a non-canonical qubit subset/order: qargs = [2, 0].  After substitution the
    # PauliLindbladError must land on physical qubits [0, 2] (ascending), not [2, 0].
    qc = QuantumCircuit(3)
    qc.append(Barrier(2, label="R0@tag=r0"), [2, 0])

    result = PassManager([InsertNoisePass(noise_dict=noise_dict, noise_after=True)]).run(qc)

    noise_instrs = [instr for instr in result.data if instr.operation.name == "quantum_channel"]
    assert len(noise_instrs) == 1
    assert [result.find_bit(q).index for q in noise_instrs[0].qubits] == [0, 2]
    # The original barrier should appear exactly once (regression: an earlier fix duplicated it).
    assert result.count_ops().get("barrier", 0) == 1


def test_noise_simulation_applies_rates_to_correct_physical_qubits(noise_dict):
    # with asymmetric rates ("XI", 0.1) and ("IX", 0.2) and the barrier placed on qargs [2, 0],
    # the higher rate (0.2, from "IX" on local qubit 0) should land on physical qubit 0, and the
    # lower rate (0.1, from "XI" on local qubit 1) on physical qubit 2.
    qc = QuantumCircuit(3)
    qc.append(Barrier(2, label="R0@tag=r0"), [2, 0])

    noisy = PassManager([InsertNoisePass(noise_dict=noise_dict, noise_after=True)]).run(qc)
    noisy.save_density_matrix()

    result = AerSimulator(method="density_matrix").run(noisy).result()
    dm = DensityMatrix(result.data(0)["density_matrix"])

    p_q0_expected = (1 - np.exp(-2 * 0.2)) / 2
    p_q2_expected = (1 - np.exp(-2 * 0.1)) / 2
    np.testing.assert_allclose(dm.probabilities([0])[1], p_q0_expected, atol=1e-10)
    np.testing.assert_allclose(dm.probabilities([1])[1], 0.0, atol=1e-10)
    np.testing.assert_allclose(dm.probabilities([2])[1], p_q2_expected, atol=1e-10)
