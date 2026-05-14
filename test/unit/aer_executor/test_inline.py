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

"""Tests for qiskit_noise_learning.aer_executor.inline.resolve_samplex_item()."""

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.quantum_program import QuantumProgram
from qiskit_ibm_runtime.quantum_program.quantum_program import CircuitItem
from samplomatic.builders.build import build
from samplomatic.transpiler import generate_boxing_pass_manager

from qiskit_noise_learning.aer_executor import AerExecutor
from qiskit_noise_learning.aer_executor.inline import (
    inline_samplex_item,
    inline_samplexes,
)


@pytest.fixture
def cx_samplex_item(fez_backend):
    """SamplexItem for a Pauli-twirled CX circuit with shape (8,)."""
    qc = QuantumCircuit(2, 2)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    pm = generate_preset_pass_manager(
        backend=fez_backend, initial_layout=[17, 27], optimization_level=0
    )
    pm.post_scheduling = generate_boxing_pass_manager()
    transpiled = pm.run(qc)
    template_circuit, samplex = build(transpiled)
    program = QuantumProgram(shots=64)
    program.append_samplex_item(template_circuit, samplex=samplex, shape=(8,))
    return program.items[0]


@pytest.fixture
def param_samplex_item(fez_backend):
    """SamplexItem for a parameterized circuit with shape (r0=3, 2, 2, r1=4)."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(2, 2)
    qc.rx(theta, 0)
    qc.cx(0, 1)
    qc.rx(phi, 1)
    qc.measure([0, 1], [0, 1])
    pm = generate_preset_pass_manager(
        backend=fez_backend, initial_layout=[17, 27], optimization_level=0
    )
    pm.post_scheduling = generate_boxing_pass_manager()
    transpiled = pm.run(qc)
    template_circuit, samplex = build(transpiled)
    # parameters sorted alphabetically: [phi, theta]
    param_values = np.array([[[0.0, 0.0], [np.pi, 0.0]], [[0.0, np.pi], [np.pi, np.pi]]]).reshape(
        (2, 2, 1, 2)
    )
    program = QuantumProgram(shots=64)
    program.append_samplex_item(
        template_circuit,
        samplex=samplex,
        samplex_arguments={"parameter_values": param_values},
        shape=(3, 2, 2, 4),
    )
    return program.items[0]


def test_resolve_returns_circuit_item(cx_samplex_item):
    """resolve_samplex_item returns a (CircuitItem, dict) tuple."""
    circuit_item, passthrough = inline_samplex_item(cx_samplex_item)

    assert isinstance(circuit_item, CircuitItem)
    assert isinstance(passthrough, dict)


def test_resolve_parameter_values_not_in_passthrough(cx_samplex_item):
    """'parameter_values' must be consumed into circuit_item, not left in passthrough."""
    _, passthrough = inline_samplex_item(cx_samplex_item)

    assert "parameter_values" not in passthrough


def test_resolve_circuit_arguments_shape(cx_samplex_item):
    """circuit_arguments leading shape matches item.shape."""
    item = cx_samplex_item
    circuit_item, _ = inline_samplex_item(item)

    assert circuit_item.circuit_arguments is not None
    assert circuit_item.circuit_arguments.shape[: len(item.shape)] == item.shape


def test_resolve_passthrough_shapes(cx_samplex_item):
    """Passthrough arrays have leading shape matching item.shape."""
    item = cx_samplex_item
    _, passthrough = inline_samplex_item(item)

    assert len(passthrough) > 0
    for key, val in passthrough.items():
        assert (
            val.shape[: len(item.shape)] == item.shape
        ), f"Expected leading shape {item.shape} for passthrough '{key}', got {val.shape}"


def test_resolve_measurement_flips_in_passthrough(cx_samplex_item):
    """Passthrough contains 'measurement_flips.*' keys from the samplex."""
    _, passthrough = inline_samplex_item(cx_samplex_item)

    flip_keys = [k for k in passthrough if k.startswith("measurement_flips.")]
    assert len(flip_keys) > 0


def test_resolve_uses_same_circuit(cx_samplex_item):
    """The resolved CircuitItem uses the same circuit object as the SamplexItem."""
    item = cx_samplex_item
    circuit_item, _ = inline_samplex_item(item)

    assert circuit_item.circuit is item.circuit


def test_resolve_default_rng(cx_samplex_item):
    """resolve_samplex_item works when rng=None (uses default_rng internally)."""
    circuit_item, passthrough = inline_samplex_item(cx_samplex_item, rng=None)

    assert circuit_item.circuit_arguments is not None
    assert len(passthrough) > 0


def test_resolve_roundtrip_cx_correct(cx_samplex_item, stabilizer_simulator):
    """CX|00⟩ = |00⟩: after flip correction the inlined simulation yields all-zero bits.

    This validates the full pipeline: inline SamplexItem → run via AerExecutor →
    apply measurement_flips from passthrough.
    """
    item = cx_samplex_item
    rng = np.random.default_rng(42)

    circuit_item, passthrough = inline_samplex_item(item, rng=rng)

    program = QuantumProgram(shots=64)
    program.append_circuit_item(
        circuit_item.circuit, circuit_arguments=circuit_item.circuit_arguments
    )
    result = AerExecutor(stabilizer_simulator).run(program).result()

    for name, arr in result[0].items():
        flips = passthrough.get(f"measurement_flips.{name}")
        corrected = arr ^ flips if flips is not None else arr
        assert (corrected == False).all(), (  # noqa: E712
            f"CX|00⟩ should yield all-zero bits for '{name}' after flip correction"
        )


def test_resolve_param_sweep_shapes(param_samplex_item):
    """With a parameter sweep, circuit_arguments and passthrough have shape (r0, 2, 2, r1, ...)."""
    item = param_samplex_item
    shape = item.shape  # (3, 2, 2, 4)

    circuit_item, passthrough = inline_samplex_item(item, rng=np.random.default_rng(0))

    assert circuit_item.circuit_arguments.shape[: len(shape)] == shape
    for key, val in passthrough.items():
        assert (
            val.shape[: len(shape)] == shape
        ), f"Expected leading shape {shape} for '{key}', got {val.shape}"


# ---------------------------------------------------------------------------
# inline_samplexes tests
# ---------------------------------------------------------------------------


@pytest.fixture
def cx_program(fez_backend):
    """QuantumProgram with a single Pauli-twirled CX SamplexItem."""
    qc = QuantumCircuit(2, 2)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    pm = generate_preset_pass_manager(
        backend=fez_backend, initial_layout=[17, 27], optimization_level=0
    )
    pm.post_scheduling = generate_boxing_pass_manager()
    transpiled = pm.run(qc)
    template_circuit, samplex = build(transpiled)
    program = QuantumProgram(shots=64)
    program.append_samplex_item(template_circuit, samplex=samplex, shape=(4,))
    return program


def test_inline_samplexes_converts_samplex_to_circuit(cx_program):
    """A single SamplexItem is replaced by a CircuitItem in the returned program."""
    result = inline_samplexes(cx_program, rng=np.random.default_rng(0))

    assert len(result.items) == 1
    assert isinstance(result.items[0], CircuitItem)


def test_inline_samplexes_passthrough_data_stored(cx_program):
    """Passthrough data from the inlined item is stored under 'inlined'."""
    result = inline_samplexes(cx_program, rng=np.random.default_rng(0))

    assert isinstance(result.passthrough_data, dict)
    assert "inlined" in result.passthrough_data
    inlined = result.passthrough_data["inlined"]
    assert "0" in inlined  # item index 0
    flip_keys = [k for k in inlined["0"] if k.startswith("measurement_flips.")]
    assert len(flip_keys) > 0


def test_inline_samplexes_omit(fez_backend):
    """Items at omit indices are left as SamplexItems."""
    qc = QuantumCircuit(2, 2)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    pm = generate_preset_pass_manager(
        backend=fez_backend, initial_layout=[17, 27], optimization_level=0
    )
    pm.post_scheduling = generate_boxing_pass_manager()
    transpiled = pm.run(qc)
    template_circuit, samplex = build(transpiled)

    from qiskit_ibm_runtime.quantum_program.quantum_program import SamplexItem

    program = QuantumProgram(shots=64)
    program.append_samplex_item(template_circuit, samplex=samplex, shape=(2,))
    program.append_samplex_item(template_circuit, samplex=samplex, shape=(3,))

    result = inline_samplexes(program, omit=[1], rng=np.random.default_rng(0))

    assert isinstance(result.items[0], CircuitItem)
    assert isinstance(result.items[1], SamplexItem)
    assert "0" in result.passthrough_data["inlined"]
    assert "1" not in result.passthrough_data["inlined"]


def test_inline_samplexes_mixed_items(fez_backend):
    """CircuitItems are left untouched; only SamplexItems are inlined."""
    qc_bare = QuantumCircuit(2, 2)
    qc_bare.h(0)
    qc_bare.cx(0, 1)
    qc_bare.measure([0, 1], [0, 1])
    pm_bare = generate_preset_pass_manager(
        backend=fez_backend, initial_layout=[17, 27], optimization_level=0
    )
    transpiled_bare = pm_bare.run(qc_bare)

    qc_cx = QuantumCircuit(2, 2)
    qc_cx.cx(0, 1)
    qc_cx.measure([0, 1], [0, 1])
    pm_cx = generate_preset_pass_manager(
        backend=fez_backend, initial_layout=[17, 27], optimization_level=0
    )
    pm_cx.post_scheduling = generate_boxing_pass_manager()
    transpiled_cx = pm_cx.run(qc_cx)
    template_circuit, samplex = build(transpiled_cx)

    program = QuantumProgram(shots=64)
    program.append_circuit_item(transpiled_bare)
    program.append_samplex_item(template_circuit, samplex=samplex, shape=(2,))

    result = inline_samplexes(program, rng=np.random.default_rng(0))

    assert len(result.items) == 2
    assert isinstance(result.items[0], CircuitItem)
    assert isinstance(result.items[1], CircuitItem)
    # Item 0 was already a CircuitItem — not inlined, so not in passthrough
    assert "0" not in result.passthrough_data["inlined"]
    assert "1" in result.passthrough_data["inlined"]


def test_inline_samplexes_preserves_program_fields(cx_program):
    """shots, noise_maps, and meas_level are preserved."""
    result = inline_samplexes(cx_program, rng=np.random.default_rng(0))

    assert result.shots == cx_program.shots
    assert result.noise_maps == cx_program.noise_maps
    assert result.meas_level == cx_program.meas_level


def test_inline_samplexes_merges_existing_passthrough_data(fez_backend):
    """When ```program.passthrough_data``` is already a dict, inlined data is merged into it."""
    qc = QuantumCircuit(2, 2)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    pm = generate_preset_pass_manager(
        backend=fez_backend, initial_layout=[17, 27], optimization_level=0
    )
    pm.post_scheduling = generate_boxing_pass_manager()
    transpiled = pm.run(qc)
    template_circuit, samplex = build(transpiled)

    pre_existing = {"my_data": np.array([1, 2, 3])}
    program = QuantumProgram(shots=64, passthrough_data=pre_existing)
    program.append_samplex_item(template_circuit, samplex=samplex, shape=(2,))

    result = inline_samplexes(program, rng=np.random.default_rng(0))

    assert "inlined" in result.passthrough_data
    assert "my_data" in result.passthrough_data
    np.testing.assert_array_equal(result.passthrough_data["my_data"], pre_existing["my_data"])


def test_inline_samplexes_roundtrip(cx_program, stabilizer_simulator):
    """Inline a CX SamplexItem, run via AerExecutor, apply flips → all-zero bits."""
    inlined = inline_samplexes(cx_program, rng=np.random.default_rng(42))
    passthrough = inlined.passthrough_data["inlined"]["0"]

    result = AerExecutor(stabilizer_simulator).run(inlined).result()

    for name, arr in result[0].items():
        flips = passthrough.get(f"measurement_flips.{name}")
        corrected = arr ^ flips if flips is not None else arr
        assert (corrected == False).all(), (  # noqa: E712
            f"CX|00⟩ should yield all-zero bits for '{name}' after flip correction"
        )
