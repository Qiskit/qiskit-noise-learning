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

"""Test AerExecutor."""

from itertools import islice
from typing import Literal

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives.containers.bit_array import BitArray
from qiskit.quantum_info import PauliLindbladMap
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider.backends.fez import FakeFez
from qiskit_ibm_runtime.quantum_program import QuantumProgram
from samplomatic import Tag, Twirl
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


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def circ_a():
    num_qubits = 2
    active_qubits = list(range(num_qubits))

    qc_boxed = QuantumCircuit(num_qubits, num_qubits)
    with qc_boxed.box(
        annotations=[
            Twirl(dressing="left"),
            Tag(ref="r0"),
        ]
    ):  # pyright: ignore[reportGeneralTypeIssues]
        for edge in batched(active_qubits, 2):
            if len(edge) == 2:
                qc_boxed.cz(*edge)

    with qc_boxed.box(annotations=[Twirl(dressing="right")]):
        qc_boxed.noop([0, 1])
    return qc_boxed, active_qubits


def circ_b():
    fez_backend = FakeFez()
    coupling_map = fez_backend.coupling_map
    active_qubits = list(range(fez_backend.num_qubits))

    qc_boxed = QuantumCircuit(fez_backend.num_qubits, fez_backend.num_qubits)
    with qc_boxed.box(
        annotations=[
            Twirl(dressing="left"),
            Tag(ref="r0"),
        ]
    ):  # pyright: ignore[reportGeneralTypeIssues]
        for edge in batched(active_qubits, 2):
            if edge in coupling_map:
                qc_boxed.cz(*edge)
            else:
                qc_boxed.z(edge)

    with qc_boxed.box(annotations=[Twirl(dressing="right")]):
        qc_boxed.noop(active_qubits)

    return qc_boxed, active_qubits


@pytest.mark.parametrize("case", ["a", "b"])
@pytest.mark.parametrize("noise", [True, False])
def test_small_noisy(stabilizer_simulator, noise: bool, case: Literal["a", "b"]):
    if case == "a":
        qc_boxed, active_qubits = circ_a()
    elif case == "b":
        qc_boxed, active_qubits = circ_b()
    else:
        raise ValueError("...")

    qc_boxed.measure(active_qubits, active_qubits)

    template_circuit, samplex = build(qc_boxed)

    assert template_circuit.count_ops().get("rz", 0)

    shots_per_twirl = 1024
    num_twirls = 1
    num_shots_tot = shots_per_twirl * num_twirls

    # Build a QuantumProgram using a SamplexItem
    program = QuantumProgram(shots=shots_per_twirl)
    program.append_samplex_item(
        template_circuit,
        samplex=samplex,
        shape=(num_twirls,),
    )

    def _xi(i: int, n: int = len(active_qubits)) -> str:
        ll = ["I"] * n
        ll[i] = "X"
        return "".join(reversed(ll))

    if noise:
        noise_dict = {
            "r0": PauliLindbladMap.from_list([(_xi(i), 1e-1) for i in range(len(active_qubits))]),
        }
    else:
        noise_dict = None

    # Run via AerExecutor
    executor = AerExecutor(
        stabilizer_simulator,
        noise_dict=noise_dict,
    )
    job = executor.run(program)
    result = job.result()

    assert len(result) == 1

    ba_c = BitArray.from_bool_array(result[0]["c"])
    cts = ba_c.get_counts()
    if noise:
        assert cts.get("0" * len(active_qubits), 0) < num_shots_tot
    else:
        assert cts.get("0" * len(active_qubits), 0) == num_shots_tot
