# This code is a Qiskit project.
#
# (C) Copyright IBM 2025, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test AerExecutor with a Clifford circuit on the stabilizer simulator."""

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.quantum_program import QuantumProgram
from samplomatic.builders.build import build
from samplomatic.transpiler import generate_boxing_pass_manager

from qiskit_noise_learning.aer_executor import AerExecutor


def assert_correct(expected: dict[str, np.ndarray], executor_results: dict[str, np.ndarray]):
    """Assert that executor results match the expected bit arrays after flip correction.

    Bit arrays use LSB-first ordering: ``data[..., 0]`` corresponds to classical
    register bit 0.

    Args:
        expected: A map from classical register names to boolean arrays of shape
            ``(num_bits,)`` giving the expected deterministic outcome.
        executor_results: A map from register names to boolean arrays of shape
            ``(num_randomizations, num_shots, num_bits)``, and optionally
            ``measurement_flips.<name>`` arrays that broadcast over the shots axis.

    Raises:
        AssertionError: If not every expected key is present in the results.
        AssertionError: If any result array is not 3-d bool with the correct last axis.
        AssertionError: If corrected data does not exactly match the expected outcome.
    """
    for name, expected_bits in expected.items():
        assert (
            name in executor_results
        ), f"Classical register '{name}' not found in executor results"
        arr = executor_results[name]
        assert arr.dtype == np.bool_, f"Expected '{name}' to have dtype bool, got '{arr.dtype}'."
        assert arr.ndim == 3, f"Expected 3-d array for '{name}', got shape {arr.shape}"
        assert arr.shape[2] == len(
            expected_bits
        ), f"Expected {len(expected_bits)} bits for '{name}', got {arr.shape[2]}"

        corrected = arr
        if (flips := executor_results.get(f"measurement_flips.{name}")) is not None:
            corrected = arr ^ flips

        assert (
            corrected == expected_bits
        ).all(), f"Corrected data for '{name}' does not match expected outcome {expected_bits}"


def test_clifford_circuit_item(fez_backend, stabilizer_simulator):
    """Run a simple GHZ-like Clifford circuit on qubits [17, 18, 19] of FakeFez
    using the stabilizer simulation method via a CircuitItem."""

    # Build a simple 3-qubit Clifford circuit (GHZ state preparation + measurement)
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])

    # Transpile to FakeFez using a non-trivial layout: physical qubits [17, 18, 19]
    pm = generate_preset_pass_manager(
        backend=fez_backend,
        initial_layout=[17, 18, 19],
        optimization_level=0,
    )
    transpiled = pm.run(qc)

    # Build a QuantumProgram with the transpiled circuit (no free parameters)
    program = QuantumProgram(shots=1024)
    program.append_circuit_item(transpiled)

    # Run via AerExecutor
    executor = AerExecutor(stabilizer_simulator)
    job = executor.run(program)
    result = job.result()

    # The result should have one item
    assert len(result) == 1

    item_data = result[0]

    # There should be a classical register key in the result data
    assert len(item_data) > 0

    # Each measurement outcome array should have shape (shots, num_clbits)
    for key, arr in item_data.items():
        assert arr.shape[0] == 1024, f"Expected 1024 shots, got {arr.shape[0]} for register '{key}'"

    # For a GHZ state, only '000' and '111' outcomes are possible.
    # Verify that all shots are either all-zeros or all-ones across the 3 bits.
    for key, arr in item_data.items():
        for shot in arr:
            assert all(shot == 0) or all(
                shot == 1
            ), f"Unexpected measurement outcome {shot} — GHZ state should only yield 000 or 111"


def test_clifford_samplex_item(fez_backend, stabilizer_simulator):
    """Run a Pauli-twirled CX circuit on connected qubits [17, 27] of FakeFez
    using the stabilizer simulation method via a SamplexItem."""
    num_randomizations = 8
    shots = 256

    # Build a simple 2-qubit CX circuit (|00⟩ → |00⟩ deterministically)
    qc = QuantumCircuit(2, 2)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    # Transpile and box in one pass manager
    pm = generate_preset_pass_manager(
        backend=fez_backend,
        initial_layout=[17, 27],
        optimization_level=0,
    )
    pm.post_scheduling = generate_boxing_pass_manager()
    transpiled = pm.run(qc)
    template_circuit, samplex = build(transpiled)

    # Build a QuantumProgram using a SamplexItem
    program = QuantumProgram(shots=shots)
    program.append_samplex_item(
        template_circuit,
        samplex=samplex,
        shape=(num_randomizations,),
    )

    # Run via AerExecutor
    executor = AerExecutor(stabilizer_simulator)
    job = executor.run(program)
    result = job.result()

    assert len(result) == 1
    item_data = result[0]

    # CX|00⟩ = |00⟩
    assert_correct({"c": np.array([False, False])}, item_data)


def test_circuit_item_with_circuit_arguments(fez_backend, stabilizer_simulator):
    """Run a parameterized CircuitItem by supplying circuit_arguments directly.

    Uses RX(theta) bitflips (theta ∈ {0, π}) on two qubits to produce four
    deterministic outcomes, verifying that circuit_arguments are bound correctly
    in the CircuitItem branch of run_quantum_program.
    """
    shots = 128

    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(2, 2)
    qc.rx(theta, 0)
    qc.rx(phi, 1)
    qc.measure([0, 1], [0, 1])

    pm = generate_preset_pass_manager(
        backend=fez_backend,
        initial_layout=[17, 27],
        optimization_level=0,
    )
    transpiled = pm.run(qc)

    # circuit.parameters sorts alphabetically: [phi, theta]
    # shape (4, 2): 4 sweep configurations, 2 parameters each
    circuit_arguments = np.array(
        [[0.0, 0.0], [np.pi, 0.0], [0.0, np.pi], [np.pi, np.pi]], dtype=float
    )
    program = QuantumProgram(shots=shots)
    program.append_circuit_item(transpiled, circuit_arguments=circuit_arguments)

    result = AerExecutor(stabilizer_simulator).run(program).result()

    assert len(result) == 1
    item_data = result[0]

    # Result shape: (4, shots, 2)
    assert item_data["c"].shape == (4, shots, 2)

    # phi=0, theta=0 → |00⟩
    assert (item_data["c"][0] == [False, False]).all()
    # phi=π, theta=0 → |01⟩ (phi acts on q1, LSB-first: bit1=True)
    assert (item_data["c"][1] == [False, True]).all()
    # phi=0, theta=π → |10⟩ (theta acts on q0, LSB-first: bit0=True)
    assert (item_data["c"][2] == [True, False]).all()
    # phi=π, theta=π → |11⟩
    assert (item_data["c"][3] == [True, True]).all()


@pytest.mark.parametrize("angle_decimals", [5])
def test_samplex_item_with_parameter_sweep(fez_backend, stabilizer_simulator, angle_decimals: int):
    """Run a parameterized CircuitItem by supplying circuit_arguments directly.

    Uses RX(theta) bitflips (theta ∈ {0, π}) on two qubits to produce four
    deterministic outcomes, verifying that circuit_arguments are bound correctly
    in the CircuitItem branch of run_quantum_program.
    """
    shots = 128

    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(2, 2)
    qc.rx(theta, 0)
    qc.rx(phi, 1)
    qc.measure([0, 1], [0, 1])

    pm = generate_preset_pass_manager(
        backend=fez_backend,
        initial_layout=[17, 27],
        optimization_level=0,
    )
    transpiled = pm.run(qc)

    # circuit.parameters sorts alphabetically: [phi, theta]
    # shape (4, 2): 4 sweep configurations, 2 parameters each
    circuit_arguments = np.array(
        [[0.0, 0.0], [np.pi, 0.0], [0.0, np.pi], [np.pi, np.pi]], dtype=float
    )
    program = QuantumProgram(shots=shots)
    program.append_circuit_item(transpiled, circuit_arguments=circuit_arguments)

    result = AerExecutor(stabilizer_simulator).run(program).result()

    assert len(result) == 1
    item_data = result[0]

    # Result shape: (4, shots, 2)
    assert item_data["c"].shape == (4, shots, 2)

    # phi=0, theta=0 → |00⟩
    assert (item_data["c"][0] == [False, False]).all()
    # phi=π, theta=0 → |01⟩ (phi acts on q1, LSB-first: bit1=True)
    assert (item_data["c"][1] == [False, True]).all()
    # phi=0, theta=π → |10⟩ (theta acts on q0, LSB-first: bit0=True)
    assert (item_data["c"][2] == [True, False]).all()
    # phi=π, theta=π → |11⟩
    assert (item_data["c"][3] == [True, True]).all()


@pytest.mark.parametrize("angle_decimals", [5])
def test_samplex_item_with_broadcast_sweep(fez_backend, stabilizer_simulator, angle_decimals: int):
    """Run a Pauli-twirled circuit with a parameter sweep over input bitflips,
    verifying that broadcast dimensions in samplex_arguments are handled correctly.

    The circuit applies RX(theta) on qubit 0 before CX and RX(phi) on qubit 1
    after CX, where theta, phi ∈ {0, π} act as bitflips. This keeps the circuit
    Clifford for the stabilizer simulator and gives four distinct, deterministic
    outcomes.

    The shape ``(r0, 2, 2, r1)`` places the two broadcast axes (theta sweep,
    phi sweep) between two randomization axes, exercising non-adjacent
    randomization dimensions in ``_broadcast_sample``.
    """
    r0 = 3
    r1 = 4
    shots = 64

    # Build a 2-qubit circuit: RX(theta) on q0, CX, RX(phi) on q1, then measure
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(2, 2)
    qc.rx(theta, 0)
    qc.cx(0, 1)
    qc.rx(phi, 1)
    qc.measure([0, 1], [0, 1])

    # Transpile and box in one pass manager
    pm = generate_preset_pass_manager(
        backend=fez_backend,
        initial_layout=[17, 27],
        optimization_level=0,
    )
    pm.post_scheduling = generate_boxing_pass_manager()
    transpiled = pm.run(qc)
    template_circuit, samplex = build(transpiled)

    # Sweep over bitflip values: shape (2, 2, 1, 2) = (theta, phi, r1_placeholder, params)
    # Extrinsic shape (2, 2, 1) right-aligns with (r0, 2, 2, r1) as padded (1, 2, 2, 1)
    # circuit.parameters sorts alphabetically: [phi, theta]
    param_values = np.array(
        [
            [[0.0, 0.0], [np.pi, 0.0]],
            [[0.0, np.pi], [np.pi, np.pi]],
        ]
    ).reshape((2, 2, 1, 2))
    program = QuantumProgram(shots=shots)
    program.append_samplex_item(
        template_circuit,
        samplex=samplex,
        samplex_arguments={"parameter_values": param_values},
        shape=(r0, 2, 2, r1),
    )

    executor = AerExecutor(stabilizer_simulator, angle_decimals=angle_decimals)
    job = executor.run(program)
    result = job.result()

    assert len(result) == 1
    item_data = result[0]

    # Verify broadcast produced the expected extrinsic shape (r0, 2, 2, r1, ...)
    for key, arr in item_data.items():
        assert arr.shape[:4] == (
            r0,
            2,
            2,
            r1,
        ), f"Expected leading shape {(r0, 2, 2, r1)}, got {arr.shape[:4]} for '{key}'"

    # Check correctness per sweep value.
    # Collapse the two randomization axes (r0, r1) into one for assert_correct.
    def sweep_slice(i, j):
        return {
            key: arr[:, i, j, :].reshape(r0 * r1, *arr.shape[4:]) for key, arr in item_data.items()
        }

    # theta=0, phi=0: CX|00⟩ = |00⟩, no flip on q1 → |00⟩
    assert_correct({"c": np.array([False, False])}, sweep_slice(0, 0))

    # theta=0, phi=π: CX|00⟩ = |00⟩, X on q1 → |01⟩
    assert_correct({"c": np.array([False, True])}, sweep_slice(0, 1))

    # theta=π, phi=0: CX|10⟩ = |11⟩, no flip on q1 → |11⟩
    assert_correct({"c": np.array([True, True])}, sweep_slice(1, 0))

    # theta=π, phi=π: CX|10⟩ = |11⟩, X on q1 → |10⟩
    assert_correct({"c": np.array([True, False])}, sweep_slice(1, 1))


def test_rx_decomposition_without_transpile(stabilizer_simulator):
    """RX gates are automatically decomposed to the stabilizer basis.

    Submits a circuit with raw RX gates directly to the stabilizer simulator
    without pre-transpiling to a backend basis, exercising the internal
    BasisTranslator pass that converts RX(θ) to H-RZ(θ)-H.
    """
    shots = 128

    qc = QuantumCircuit(2, 2)
    qc.rx(np.pi, 0)  # flips qubit 0
    qc.rx(0, 1)  # identity on qubit 1
    qc.measure([0, 1], [0, 1])

    program = QuantumProgram(shots=shots)
    program.append_circuit_item(qc)

    result = AerExecutor(stabilizer_simulator, decompose_rx=True).run(program).result()

    assert len(result) == 1
    item_data = result[0]

    # RX(π)|0⟩ → |1⟩, RX(0)|0⟩ → |0⟩; LSB-first: bit0=True, bit1=False
    assert item_data["c"].shape == (shots, 2)
    assert (item_data["c"][:, 0] == True).all()  # noqa: E712
    assert (item_data["c"][:, 1] == False).all()  # noqa: E712


def test_rx_parameterized_decomposition_without_transpile(stabilizer_simulator):
    """Parameterized RX gates are decomposed correctly without pre-transpilation.

    Submits a circuit with RX(theta) where theta is a Parameter, bound via
    circuit_arguments. Verifies parameter binding works after BasisTranslator
    replaces RX with H-RZ-H.
    """
    shots = 128

    theta = Parameter("theta")
    qc = QuantumCircuit(1, 1)
    qc.rx(theta, 0)
    qc.measure(0, 0)

    circuit_arguments = np.array([[0.0], [np.pi]], dtype=float)

    program = QuantumProgram(shots=shots)
    program.append_circuit_item(qc, circuit_arguments=circuit_arguments)

    result = AerExecutor(stabilizer_simulator, decompose_rx=True).run(program).result()

    assert len(result) == 1
    item_data = result[0]

    # Shape: (2, shots, 1) — two parameter configurations
    assert item_data["c"].shape == (2, shots, 1)
    # theta=0: |0⟩ → measures 0
    assert (item_data["c"][0] == False).all()  # noqa: E712
    # theta=π: |1⟩ → measures 1
    assert (item_data["c"][1] == True).all()  # noqa: E712
