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

from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.quantum_program import QuantumProgramResult
from samplomatic import Twirl

from qiskit_noise_learning.circuit_generator import ExecutorCircuitGenerator, ExecutorDataMapper
from qiskit_noise_learning.experiment_builder import ExperimentBuilder
from qiskit_noise_learning.gate_sets import QiskitGateSet
from qiskit_noise_learning.sequences import (
    ApplyGate,
    InstructionPattern,
    InstructionSequence,
    PartialPauliPermutation,
)


def make_result(items, chunk_timing=None):
    """Create a stub object mimicking QuantumProgramResult for use with
    ExecutorCircuitGenerator.collect.

    Args:
        items: A list of dicts mapping creg names (and optionally "measurement_flips.<creg>")
            to 4D arrays with shape (num_d_idxs, num_randomizations, num_shots, num_qubits).
            The first axis is indexed by d_idx in collect.
        chunk_timing: Optional list of (start, stop, parts) tuples, where parts is a list of
            (idx_item, size) tuples. If None, a single chunk is generated with dummy timestamps
            that produces the correct number of time entries per item.

    Returns:
        A stub result object with the interface expected by ExecutorCircuitGenerator.collect.
    """
    # make a default chunk timing
    if chunk_timing is None:
        parts = []
        for item_idx, item in enumerate(items):
            first_key = next(k for k in item if not k.startswith("measurement_flips."))
            arr = item[first_key]
            num_d_idxs = arr.shape[0]
            num_randomizations = arr.shape[1]
            parts.append(SimpleNamespace(idx_item=item_idx, size=num_d_idxs * num_randomizations))
        chunk_timing = [
            SimpleNamespace(
                start="2026-01-01T00:00:00",
                stop="2026-01-01T00:01:00",
                parts=parts,
            )
        ]
    else:
        chunk_timing = [
            SimpleNamespace(
                start=start,
                stop=stop,
                parts=[SimpleNamespace(idx_item=idx, size=sz) for idx, sz in parts],
            )
            for start, stop, parts in chunk_timing
        ]

    class _Result:
        def __init__(self):
            self.metadata = SimpleNamespace(chunk_timing=chunk_timing)

        def __len__(self):
            return len(items)

        def __getitem__(self, idx):
            return items[idx]

    return _Result()


def gateset_full():
    gateset = QiskitGateSet(10)

    with gateset.build_new_gate() as builder:
        builder.circuit.cz(0, 1)
        builder.circuit.cz(2, 3)
        builder.circuit.noop(range(10))

    with gateset.build_new_gate() as builder:
        builder.circuit.cz(1, 2)
        builder.circuit.cz(3, 4)
        builder.circuit.noop(range(10))

    return gateset


def gateset_subset():
    gateset = QiskitGateSet(15, qubit_subset=range(2, 12))

    with gateset.build_new_gate() as builder:
        builder.circuit.cz(2, 3)
        builder.circuit.cz(4, 5)
        builder.circuit.noop(range(2, 12))

    with gateset.build_new_gate() as builder:
        builder.circuit.cz(3, 4)
        builder.circuit.cz(5, 6)
        builder.circuit.noop(range(2, 12))

    return gateset


@pytest.mark.parametrize("gateset", [gateset_full(), gateset_subset()])
def test_generate_samplex_item(gateset):
    """Test `ExecutorCircuitGenerator.generate_samplex_item()` works as expected."""
    circuit_generator = ExecutorCircuitGenerator(gateset)
    model_gateset = gateset.model_gate_set
    pattern0 = InstructionPattern(
        [ApplyGate(model_gateset["P"])],
        [ApplyGate(model_gateset["L0"]), ApplyGate(model_gateset["L1"])],
        [ApplyGate(model_gateset["M"])],
    )
    seq0 = InstructionSequence(pattern0, 5)
    samplex_item, creg_names, measurement_map = circuit_generator.generate_samplex_item([seq0])

    assert len(samplex_item.samplex_arguments) == 12
    assert creg_names == ["meas0"]
    assert "meas0" in measurement_map

    expected = np.zeros((1, 1, 10), np.uint8)
    for value in samplex_item.samplex_arguments.values():
        assert np.array_equal(value, expected)

    gateset_idxs = [idx for idx in gateset.qubit_subset]
    gateset_idxs.sort()
    array = np.empty((gateset.num_qubits,), dtype=np.uint8)
    array[gateset_idxs] = 1

    perm = PartialPauliPermutation(array)
    pattern1 = InstructionPattern(
        [ApplyGate(model_gateset["P"])],
        [ApplyGate(model_gateset["L0"]), ApplyGate(model_gateset["L1"])],
        [perm, ApplyGate(model_gateset["M"])],
    )
    seq1 = InstructionSequence(pattern1, 5)

    pattern2 = InstructionPattern(
        [ApplyGate(model_gateset["P"]), perm],
        [ApplyGate(model_gateset["L0"]), perm, ApplyGate(model_gateset["L1"]), perm],
        [ApplyGate(model_gateset["M"])],
    )
    seq2 = InstructionSequence(pattern2, 5)

    other_samplex_item, other_creg_names, other_meas_map = circuit_generator.generate_samplex_item(
        [seq0, seq1, seq2]
    )

    assert samplex_item.samplex == other_samplex_item.samplex
    assert len(other_samplex_item.samplex_arguments) == 12
    assert other_creg_names == creg_names
    assert other_meas_map.keys() == measurement_map.keys()

    expected = np.zeros((3, 1, 10), np.uint8)
    values = list(other_samplex_item.samplex_arguments.values())
    assert np.array_equal(values[0], expected)

    expected[2, 0] = 7
    for value in values[1:-1]:
        assert np.array_equal(value, expected)

    expected[1, 0] = 7
    assert np.array_equal(values[-1], expected)


@pytest.mark.parametrize("gateset", [gateset_full(), gateset_subset()])
def test_generate_samplex_item_permutation_composition(gateset):
    """Test `ExecutorCircuitGenerator.generate_samplex_item()` works as expected with multiple
    permutations in a row.
    """

    circuit_generator = ExecutorCircuitGenerator(gateset)
    model_gateset = gateset.model_gate_set

    gateset_idxs = [idx for idx in gateset.qubit_subset]
    gateset_idxs.sort()
    array = np.empty((gateset.num_qubits,), dtype=np.uint8)
    array[gateset_idxs] = 1

    perm0 = PartialPauliPermutation(array)
    pattern0 = InstructionPattern(
        [ApplyGate(model_gateset["P"]), perm0],
        [ApplyGate(model_gateset["L0"]), perm0, ApplyGate(model_gateset["L1"]), perm0],
        [ApplyGate(model_gateset["M"])],
    )
    seq0 = InstructionSequence(pattern0, 5)

    array = np.empty((gateset.num_qubits,), dtype=np.uint8)
    array[gateset_idxs] = 2
    perm1 = PartialPauliPermutation(array)
    pattern1 = InstructionPattern(
        [ApplyGate(model_gateset["P"]), perm1, perm0],
        [ApplyGate(model_gateset["L0"]), perm0, ApplyGate(model_gateset["L1"]), perm0],
        [ApplyGate(model_gateset["M"])],
    )
    seq1 = InstructionSequence(pattern1, 5)

    samplex_item0, creg_names0, meas_map0 = circuit_generator.generate_samplex_item([seq0])
    samplex_item1, creg_names1, meas_map1 = circuit_generator.generate_samplex_item([seq1])

    assert creg_names0 == creg_names1
    assert meas_map0.keys() == meas_map1.keys()
    for key in meas_map0:
        np.testing.assert_array_equal(meas_map0[key], meas_map1[key])

    # should have same structure but different values
    assert not all(
        (a == b).all()
        for a, b in zip(
            samplex_item0.samplex_arguments.values(), samplex_item1.samplex_arguments.values()
        )
    )

    pattern2 = InstructionPattern(
        [ApplyGate(model_gateset["P"]), perm0.compose(perm1)],
        [ApplyGate(model_gateset["L0"]), perm0, ApplyGate(model_gateset["L1"]), perm0],
        [ApplyGate(model_gateset["M"])],
    )
    seq2 = InstructionSequence(pattern2, 5)
    samplex_item2, creg_names2, meas_map2 = circuit_generator.generate_samplex_item([seq2])

    assert creg_names2 == creg_names1
    assert meas_map2.keys() == meas_map1.keys()
    for key in meas_map2:
        np.testing.assert_array_equal(meas_map2[key], meas_map1[key])

    # should have same structure and same values
    assert all(
        (a == b).all()
        for a, b in zip(
            samplex_item2.samplex_arguments.values(), samplex_item1.samplex_arguments.values()
        )
    )

    pattern3 = InstructionPattern(
        [ApplyGate(model_gateset["P"]), perm1.compose(perm0)],
        [ApplyGate(model_gateset["L0"]), perm0, ApplyGate(model_gateset["L1"]), perm0],
        [ApplyGate(model_gateset["M"])],
    )
    seq3 = InstructionSequence(pattern3, 5)
    samplex_item3, creg_names3, meas_map3 = circuit_generator.generate_samplex_item([seq3])

    assert creg_names3 == creg_names1
    for key in meas_map3:
        np.testing.assert_array_equal(meas_map3[key], meas_map1[key])

    # should have same structure different values
    assert not all(
        (a == b).all()
        for a, b in zip(
            samplex_item3.samplex_arguments.values(), samplex_item1.samplex_arguments.values()
        )
    )

    # no repeatable fragment
    pattern4 = InstructionPattern(
        [ApplyGate(model_gateset["P"]), perm0],
        [],
        [perm1, ApplyGate(model_gateset["M"])],
    )
    seq4 = InstructionSequence(pattern4, 5)
    samplex_item4, creg_names4, meas_map4 = circuit_generator.generate_samplex_item([seq4])

    pattern5 = InstructionPattern(
        [ApplyGate(model_gateset["P"])],
        [],
        [perm1.compose(perm0), ApplyGate(model_gateset["M"])],
    )
    seq5 = InstructionSequence(pattern5, 5)
    samplex_item5, creg_names5, meas_map5 = circuit_generator.generate_samplex_item([seq5])

    assert creg_names5 == creg_names4
    for key in meas_map5:
        np.testing.assert_array_equal(meas_map5[key], meas_map4[key])

    # should have same structure same values
    assert all(
        (a == b).all()
        for a, b in zip(
            samplex_item5.samplex_arguments.values(), samplex_item4.samplex_arguments.values()
        )
    )


def test_generate_samplex_item_raises():
    """Test `ExecutorCircuitGenerator.generate_samplex_item()` raises errors as expected."""
    gateset = gateset_full()
    circuit_generator = ExecutorCircuitGenerator(gateset)

    with pytest.raises(ValueError, match="At least one instruction"):
        circuit_generator.generate_samplex_item([])

    model_gateset = gateset.model_gate_set

    pattern0 = InstructionPattern(
        [ApplyGate(model_gateset["P"])],
        [ApplyGate(model_gateset["L0"])],
        [ApplyGate(model_gateset["M"])],
    )
    pattern1 = InstructionPattern(
        [ApplyGate(model_gateset["P"])],
        [ApplyGate(model_gateset["L1"])],
        [ApplyGate(model_gateset["M"])],
    )
    seq0 = InstructionSequence(pattern0, 3)
    seq1 = InstructionSequence(pattern1, 3)
    seq2 = InstructionSequence(pattern1, 4)

    with pytest.raises(ValueError, match="require the same structure"):
        circuit_generator.generate_samplex_item([seq0, seq1])

    with pytest.raises(ValueError, match="require the same structure"):
        circuit_generator.generate_samplex_item([seq1, seq2])

    perm = PartialPauliPermutation.from_sets([{("X", "Y")}])
    pattern2 = InstructionPattern(
        [ApplyGate(model_gateset["P"])], [perm], [ApplyGate(model_gateset["M"])]
    )
    seq3 = InstructionSequence(pattern2, 1)

    with pytest.raises(ValueError, match="incomplete Pauli"):
        circuit_generator.generate_samplex_item([seq3])


@pytest.mark.parametrize("gateset", [gateset_full(), gateset_subset()])
def test_generate_samplex_items(gateset):
    """Test `ExecutorCircuitGenerator.generate_samplex_items()` works as expected."""
    circuit_generator = ExecutorCircuitGenerator(gateset)
    model_gateset = gateset.model_gate_set
    pattern0 = InstructionPattern(
        [ApplyGate(model_gateset["P"])],
        [ApplyGate(model_gateset["L0"]), ApplyGate(model_gateset["L1"])],
        [ApplyGate(model_gateset["M"])],
    )

    # Single pattern with 3 depths => 3 sequences, each in its own partition
    builder = ExperimentBuilder(model_gateset)
    builder._instruction_patterns = [pattern0]  # noqa: SLF001
    builder.complete()

    depths = [1, 3, 10]
    samplex_items, data_mapper = circuit_generator.generate_samplex_items(builder, depths)
    assert len(samplex_items) == 3
    assert data_mapper.creg_names == [["meas0"], ["meas0"], ["meas0"]]
    assert data_mapper.item_sequence_indices == [[0], [1], [2]]

    # Two patterns with the same gate structure => same-depth sequences are grouped together
    gateset_idxs = sorted(gateset.qubit_subset)
    array = np.empty((gateset.num_qubits,), dtype=np.uint8)
    array[gateset_idxs] = 1
    perm = PartialPauliPermutation(array)

    pattern1 = InstructionPattern(
        [ApplyGate(model_gateset["P"]), perm],
        [perm, ApplyGate(model_gateset["L0"]), ApplyGate(model_gateset["L1"]), perm],
        [ApplyGate(model_gateset["M"]), perm],
    )

    builder = ExperimentBuilder(model_gateset)
    builder._instruction_patterns = [pattern0, pattern1]  # noqa: SLF001
    builder.complete()

    samplex_items, data_mapper = circuit_generator.generate_samplex_items(builder, depths)
    # 2 patterns × 3 depths = 6 sequences; same-structure pairs grouped => 3 partitions
    assert len(samplex_items) == 3
    assert all(len(indices) == 2 for indices in data_mapper.item_sequence_indices)

    # Third pattern with different gate structure => creates additional partitions
    pattern2 = InstructionPattern(
        [ApplyGate(model_gateset["P"])],
        [ApplyGate(model_gateset["L1"]), ApplyGate(model_gateset["L0"])],
        [ApplyGate(model_gateset["M"])],
    )

    builder = ExperimentBuilder(model_gateset)
    builder._instruction_patterns = [pattern0, pattern1, pattern2]  # noqa: SLF001
    builder.complete()

    depths = [1, 3]
    samplex_items, data_mapper = circuit_generator.generate_samplex_items(builder, depths)
    # 3 patterns × 2 depths = 6 sequences
    # pattern0 and pattern1 share gate structure, pattern2 is different
    # => 4 partitions: (p0@1,p1@1), (p2@1), (p0@3,p1@3), (p2@3)
    assert len(samplex_items) == 4


def test_generate_different_decomposition_mode():
    """Test `ExecutorCircuitGenerator.generate_samplex_items()` works with different decomposition
    modes.
    """
    gateset = QiskitGateSet(5)

    box_circuit = QuantumCircuit(5)
    with box_circuit.box([Twirl(decomposition="rzrx")]):
        box_circuit.ecr(0, 2)
        box_circuit.ecr(1, 3)
        box_circuit.noop(4)

    gateset.add_box_as_gate(box_circuit[0], name="my_gate")

    model_gateset = gateset.model_gate_set
    pattern = InstructionPattern(
        [ApplyGate(model_gateset["P"])],
        [ApplyGate(model_gateset["my_gate"])],
        [ApplyGate(model_gateset["M"])],
    )

    builder = ExperimentBuilder(model_gateset)
    builder._instruction_patterns = [pattern]  # noqa: SLF001
    builder.complete()

    samplex_items, _ = ExecutorCircuitGenerator(gateset, 10).generate_samplex_items(builder, [5])

    ops = samplex_items[0].circuit.count_ops()
    assert ops["sx"] == 20  # 10 in the prepare, 10 in the measure
    assert ops["rx"] == 25  # 5 in each of the 5 layers


def test_collect_empty():
    """Test `ExecutorCircuitGenerator.collect()` with no sequences."""
    data_mapper = ExecutorDataMapper(
        item_sequence_indices=[],
        creg_names=[],
        measurement_maps=[],
        instruction_sequences=[],
        num_randomizations=0,
    )
    result = QuantumProgramResult([])
    fit = ExecutorCircuitGenerator.collect(result, data_mapper)
    assert len(fit.raw_data.datatree) == 0


def test_collect_single_sequence_no_measurement_flips():
    """Test `ExecutorCircuitGenerator.collect()` with a single sequence and no measurement flips."""
    creg_data = np.array([[[[1, 0, 1]]]], dtype=np.uint8)
    result = make_result([{"meas0": creg_data}])
    data_mapper = ExecutorDataMapper(
        item_sequence_indices=[[0]],
        creg_names=[["meas0"]],
        measurement_maps=[{"meas0": np.array([0, 1, 2])}],
        instruction_sequences=[
            InstructionSequence(pattern=InstructionPattern([], [], []), depth=0)
        ],
        num_randomizations=1,
    )

    fit = ExecutorCircuitGenerator.collect(result, data_mapper)
    raw_data = fit.raw_data
    dataset = raw_data.datatree["0"]
    np.testing.assert_array_equal(
        dataset["instruction_pattern"].data, [InstructionPattern([], [], [])]
    )
    np.testing.assert_array_equal(dataset["depth"].data, [0])
    np.testing.assert_array_equal(dataset["data"].data, creg_data.reshape(1, 1, 3))
    np.testing.assert_array_equal(dataset["measurement_flips"].data, np.array([[False] * 3]))
    assert dataset.dataset.attrs["creg_bit_boundaries"] == {"meas0": (0, 3)}


def test_collect_single_sequence_with_measurement_flips():
    """Test `ExecutorCircuitGenerator.collect()` with measurement flips present."""
    creg_data = np.array([[[[1, 0, 1]]]], dtype=np.uint8)
    flip_data = np.array([[[[1, 1, 0]]]], dtype=np.uint8)
    result = make_result([{"meas0": creg_data, "measurement_flips.meas0": flip_data}])
    data_mapper = ExecutorDataMapper(
        item_sequence_indices=[[0]],
        creg_names=[["meas0"]],
        measurement_maps=[{"meas0": np.array([0, 1, 2])}],
        instruction_sequences=[
            InstructionSequence(pattern=InstructionPattern([], [], []), depth=0)
        ],
        num_randomizations=1,
    )

    fit = ExecutorCircuitGenerator.collect(result, data_mapper)
    dataset = fit.raw_data.datatree["0"].dataset
    np.testing.assert_array_equal(dataset["data"].values, creg_data.reshape(1, 1, 3))
    np.testing.assert_array_equal(dataset["measurement_flips"].values, flip_data.reshape(1, 3))
    assert dataset["depth"].values == [0]
    assert dataset.attrs["creg_bit_boundaries"] == {"meas0": (0, 3)}


def test_collect_multiple_sequences_same_item():
    """Test `ExecutorCircuitGenerator.collect()` with multiple sequences in the same item."""
    creg_data = np.array([[[[1, 0]]], [[[0, 1]]]], dtype=np.uint8)
    result = make_result([{"meas0": creg_data}])
    data_mapper = ExecutorDataMapper(
        item_sequence_indices=[[0, 1]],
        creg_names=[["meas0"]],
        measurement_maps=[{"meas0": np.array([0, 1])}],
        instruction_sequences=[
            InstructionSequence(pattern=InstructionPattern([], [], []), depth=0),
            InstructionSequence(pattern=InstructionPattern([], [], []), depth=1),
        ],
        num_randomizations=1,
    )

    fit = ExecutorCircuitGenerator.collect(result, data_mapper)
    dataset = fit.raw_data.datatree["0"].dataset
    np.testing.assert_array_equal(
        dataset["instruction_pattern"].data, [InstructionPattern([], [], [])] * 2
    )
    np.testing.assert_array_equal(dataset["depth"].data, [0, 1])
    np.testing.assert_array_equal(dataset["data"].values, creg_data.reshape(2, 1, 2))
    np.testing.assert_array_equal(dataset["measurement_flips"].data, np.array([[False] * 2] * 2))
    assert dataset.attrs["creg_bit_boundaries"] == {"meas0": (0, 2)}


def test_collect_multiple_sequences_different_items():
    """Test `ExecutorCircuitGenerator.collect()` with sequences across different items."""
    data0 = np.array([[[[1, 1]]]], dtype=np.uint8)
    data1 = np.array([[[[0, 0]]]], dtype=np.uint8)
    flips1 = np.array([[[[1, 0]]]], dtype=bool)
    result = make_result(
        [
            {"meas0": data0},
            {"meas0": data1, "measurement_flips.meas0": flips1},
        ]
    )
    data_mapper = ExecutorDataMapper(
        item_sequence_indices=[[0], [1]],
        creg_names=[["meas0"], ["meas0"]],
        measurement_maps=[{"meas0": np.array([0, 1])}, {"meas0": np.array([0, 1])}],
        instruction_sequences=[
            InstructionSequence(pattern=InstructionPattern([], [], []), depth=0),
            InstructionSequence(pattern=InstructionPattern([], [], []), depth=1),
        ],
        num_randomizations=1,
    )

    fit = ExecutorCircuitGenerator.collect(result, data_mapper)
    dataset = fit.raw_data.datatree["0"].dataset
    np.testing.assert_array_equal(
        dataset["instruction_pattern"].data, [InstructionPattern([], [], [])] * 2
    )
    np.testing.assert_array_equal(dataset["depth"].data, [0, 1])
    np.testing.assert_array_equal(
        dataset["data"].values, np.append(data0, data1, axis=0).reshape(2, 1, 2)
    )
    np.testing.assert_array_equal(
        dataset["measurement_flips"].values, np.array([[False, False], [True, False]])
    )
    assert dataset.attrs["creg_bit_boundaries"] == {"meas0": (0, 2)}


def test_collect_multiple_cregs():
    """Test `ExecutorCircuitGenerator.collect()` with multiple classical registers per item."""
    creg0_data = np.array([[[[1, 0]]]], dtype=np.uint8)
    creg1_data = np.array([[[[0, 1, 1]]]], dtype=np.uint8)
    creg0_flips = np.array([[[[1, 1]]]], dtype=bool)
    result = make_result(
        [
            {
                "meas0": creg0_data,
                "meas1": creg1_data,
                "measurement_flips.meas0": creg0_flips,
            }
        ]
    )
    data_mapper = ExecutorDataMapper(
        item_sequence_indices=[[0]],
        creg_names=[["meas0", "meas1"]],
        measurement_maps=[{"meas0": np.array([0, 1]), "meas1": np.array([2, 3, 4])}],
        instruction_sequences=[
            InstructionSequence(pattern=InstructionPattern([], [], []), depth=0)
        ],
        num_randomizations=1,
    )

    fit = ExecutorCircuitGenerator.collect(result, data_mapper)
    dataset = fit.raw_data.datatree["0"].dataset
    np.testing.assert_array_equal(
        dataset["instruction_pattern"].data, [InstructionPattern([], [], [])]
    )
    np.testing.assert_array_equal(dataset["depth"].data, [0])
    np.testing.assert_array_equal(
        dataset["data"].values, np.append(creg0_data, creg1_data, axis=-1).reshape(1, 1, 5)
    )
    np.testing.assert_array_equal(
        dataset["measurement_flips"].values, np.array([[True, True, False, False, False]])
    )
    assert dataset.attrs["creg_bit_boundaries"] == {"meas0": (0, 2), "meas1": (2, 5)}


def test_collect_complex_mapping():
    """Test `ExecutorCircuitGenerator.collect()` with a mapping similar to what
    `generate_samplex_items` produces.

    Simulates three items where some sequences share an item (different d_idx) and others
    are in separate items, with a mix of measurement flips present and absent.
    Items 0 and 1 share the same creg structure (["meas0"] only) so they merge into one leaf.
    Item 2 has a different structure (["meas0", "meas1"]) so it gets its own leaf.
    """
    result = make_result(
        [
            {
                "meas0": np.array([[[[1, 0]]], [[[0, 1]]]], dtype=np.uint8),
                "measurement_flips.meas0": np.array([[[[1, 1]]], [[[0, 0]]]], dtype=bool),
            },
            {
                "meas0": np.array([[[[1, 1, 1]]]], dtype=np.uint8),
            },
            {
                "meas0": np.array([[[[0, 0, 0]]]], dtype=np.uint8),
                "meas1": np.array([[[[1]]]], dtype=np.uint8),
            },
        ]
    )
    data_mapper = ExecutorDataMapper(
        item_sequence_indices=[[0, 2], [1], [3]],
        creg_names=[["meas0"], ["meas0"], ["meas0", "meas1"]],
        measurement_maps=[
            {"meas0": np.array([0, 1])},
            {"meas0": np.array([0, 1, 2])},
            {"meas0": np.array([0, 1, 2]), "meas1": np.array([3])},
        ],
        instruction_sequences=[
            InstructionSequence(pattern=InstructionPattern([], [], []), depth=depth)
            for depth in range(4)
        ],
        num_randomizations=1,
    )

    fit = ExecutorCircuitGenerator.collect(result, data_mapper)
    raw_data = fit.raw_data

    # Item 0 has meas0 with 2 bits — leaf "0"
    dataset = raw_data.datatree["0"].dataset
    np.testing.assert_array_equal(
        dataset["instruction_pattern"].data, [InstructionPattern([], [], [])] * 2
    )
    np.testing.assert_array_equal(dataset["depth"].data, [0, 2])
    np.testing.assert_array_equal(dataset["data"].values, result[0]["meas0"].reshape(2, 1, 2))
    np.testing.assert_array_equal(
        dataset["measurement_flips"].values, result[0]["measurement_flips.meas0"].reshape(2, 2)
    )
    assert dataset.attrs["creg_bit_boundaries"] == {"meas0": (0, 2)}

    # Item 1 has meas0 with 3 bits — different measurement_map, so new leaf "1"
    dataset = raw_data.datatree["1"].dataset
    np.testing.assert_array_equal(
        dataset["instruction_pattern"].data, [InstructionPattern([], [], [])]
    )
    np.testing.assert_array_equal(dataset["depth"].data, [1])
    np.testing.assert_array_equal(dataset["data"].values, result[1]["meas0"].reshape(1, 1, 3))
    np.testing.assert_array_equal(
        dataset["measurement_flips"].values, np.array([[False, False, False]])
    )
    assert dataset.attrs["creg_bit_boundaries"] == {"meas0": (0, 3)}

    # Item 2 has meas0 + meas1 — leaf "2"
    dataset = raw_data.datatree["2"].dataset
    np.testing.assert_array_equal(
        dataset["instruction_pattern"].data, [InstructionPattern([], [], [])]
    )
    np.testing.assert_array_equal(dataset["depth"].data, [3])
    np.testing.assert_array_equal(
        dataset["data"].values,
        np.append(result[2]["meas0"], result[2]["meas1"], axis=-1).reshape(1, 1, 4),
    )
    np.testing.assert_array_equal(
        dataset["measurement_flips"].values, np.array([[False, False, False, False]])
    )
    assert dataset.attrs["creg_bit_boundaries"] == {"meas0": (0, 3), "meas1": (3, 4)}


def test_generate_and_collect_with_pass_manager():
    """Test generate_samplex_items and collect with a pass manager that adds an extra
    measurement.
    """
    from qiskit.circuit import ClassicalRegister
    from qiskit.circuit.library import Measure
    from qiskit.transpiler import PassManager, TransformationPass

    class AddMeasPass(TransformationPass):
        def run(self, dag):
            creg = ClassicalRegister(1, "pass_meas")
            dag.add_creg(creg)
            dag.apply_operation_back(Measure(), qargs=[dag.qubits[0]], cargs=[creg[0]])
            return dag

    gateset = QiskitGateSet(2)
    with gateset.build_new_gate() as gate_builder:
        gate_builder.circuit.cz(0, 1)
        gate_builder.circuit.noop(range(2))

    model_gateset = gateset.model_gate_set
    pattern = InstructionPattern(
        [ApplyGate(model_gateset["P"])],
        [ApplyGate(model_gateset["L0"])],
        [ApplyGate(model_gateset["M"])],
    )

    num_randomizations = 2
    cg = ExecutorCircuitGenerator(
        gateset, num_randomizations=num_randomizations, pass_manager=PassManager([AddMeasPass()])
    )

    builder = ExperimentBuilder(model_gateset)
    builder._instruction_patterns = [pattern]  # noqa: SLF001
    builder.complete()

    samplex_items, data_mapper = cg.generate_samplex_items(builder, [1])

    # Verify generate_samplex_items produces the expected data mapper
    assert data_mapper.item_sequence_indices == [[0]]
    assert data_mapper.creg_names == [["meas0", "pass_meas"]]
    assert list(data_mapper.measurement_maps[0].keys()) == ["meas0", "pass_meas"]
    np.testing.assert_array_equal(data_mapper.measurement_maps[0]["meas0"], [0, 1])
    np.testing.assert_array_equal(data_mapper.measurement_maps[0]["pass_meas"], [0])

    # Verify the template circuit has both cregs
    assert "pass_meas" in [c.name for c in samplex_items[0].circuit.cregs]

    # Now test collect with spoofed data
    num_shots = 3
    meas0_data = np.ones((1, num_randomizations, num_shots, 2), dtype=np.uint8)
    pass_meas_data = np.zeros((1, num_randomizations, num_shots, 1), dtype=np.uint8)
    result = make_result([{"meas0": meas0_data, "pass_meas": pass_meas_data}])
    fit = ExecutorCircuitGenerator.collect(result, data_mapper)

    dataset = fit.raw_data.datatree["0"].dataset
    assert dataset.attrs["creg_names"] == ["meas0", "pass_meas"]
    assert dataset.attrs["creg_bit_boundaries"] == {"meas0": (0, 2), "pass_meas": (2, 3)}
    np.testing.assert_array_equal(
        dataset["data"].values,
        np.concatenate([meas0_data, pass_meas_data], axis=-1).reshape(
            num_randomizations, num_shots, 3
        ),
    )
    np.testing.assert_array_equal(
        dataset["measurement_flips"].values, np.zeros((num_randomizations, 3), dtype=bool)
    )


def test_generate_with_pass_manager_multi_qubit_creg():
    """Test that the measurement map correctly captures all qubits when the pass manager adds a
    creg with multiple measurements."""
    from qiskit.circuit import ClassicalRegister
    from qiskit.circuit.library import Measure
    from qiskit.transpiler import PassManager, TransformationPass

    class AddMultiMeasPass(TransformationPass):
        def run(self, dag):
            creg = ClassicalRegister(2, "extra")
            dag.add_creg(creg)
            dag.apply_operation_back(Measure(), qargs=[dag.qubits[0]], cargs=[creg[0]])
            dag.apply_operation_back(Measure(), qargs=[dag.qubits[1]], cargs=[creg[1]])
            return dag

    gateset = QiskitGateSet(2)
    with gateset.build_new_gate() as gate_builder:
        gate_builder.circuit.cz(0, 1)
        gate_builder.circuit.noop(range(2))

    model_gateset = gateset.model_gate_set
    pattern = InstructionPattern(
        [ApplyGate(model_gateset["P"])],
        [ApplyGate(model_gateset["L0"])],
        [ApplyGate(model_gateset["M"])],
    )

    num_randomizations = 2
    cg = ExecutorCircuitGenerator(
        gateset,
        num_randomizations=num_randomizations,
        pass_manager=PassManager([AddMultiMeasPass()]),
    )

    builder = ExperimentBuilder(model_gateset)
    builder._instruction_patterns = [pattern]  # noqa: SLF001
    builder.complete()

    _, data_mapper = cg.generate_samplex_items(builder, [1])

    assert data_mapper.creg_names == [["meas0", "extra"]]
    assert list(data_mapper.measurement_maps[0].keys()) == ["meas0", "extra"]
    np.testing.assert_array_equal(data_mapper.measurement_maps[0]["meas0"], [0, 1])
    np.testing.assert_array_equal(data_mapper.measurement_maps[0]["extra"], [0, 1])
