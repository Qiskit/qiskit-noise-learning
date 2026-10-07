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

"""Tests for prepare_learning_program."""

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import QubitSparsePauli
from qiskit_ibm_runtime.fake_provider.backends.fez import FakeFez
from qiskit_ibm_runtime.quantum_program import QuantumProgram
from samplomatic import InjectNoise, Twirl

from qiskit_noise_learning.circuit_generator.executor.data_mapper_serialization import load
from qiskit_noise_learning.experiment_builder import Experiment
from qiskit_noise_learning.protocols import prepare_learning_program

PAIR = (17, 27)


@pytest.fixture(scope="module")
def backend():
    return FakeFez()


def _boxed_cz(backend, pair=PAIR, name=None):
    """One twirled box holding a single CZ, optionally named by an InjectNoise annotation."""
    annotations = [Twirl()] if name is None else [Twirl(), InjectNoise(name)]
    circuit = QuantumCircuit(backend.num_qubits)
    with circuit.box(annotations):
        circuit.cz(*pair)
    return circuit[0]


def _boxed_measure(backend, qubits=PAIR, name=None):
    """One twirled box measuring the given qubits, optionally named by an InjectNoise annotation."""
    annotations = [Twirl()] if name is None else [Twirl(), InjectNoise(name)]
    circuit = QuantumCircuit(backend.num_qubits, len(qubits))
    with circuit.box(annotations):
        for clbit, qubit in enumerate(qubits):
            circuit.measure(qubit, clbit)
    return circuit[0]


def _boxed_cz_and_measure(backend, pair=PAIR):
    """One twirled box holding a CZ and then a measurement of each of its qubits."""
    circuit = QuantumCircuit(backend.num_qubits, len(pair))
    with circuit.box([Twirl()]):
        circuit.cz(*pair)
        for clbit, qubit in enumerate(pair):
            circuit.measure(qubit, clbit)
    return circuit[0]


def _split_paths(paths):
    """Partition paths into the layer decays and the depth-0 SPAM paths."""
    spam = [path for path in paths if len(path.repeatable_fragment) == 0]
    layer = [path for path in paths if len(path.repeatable_fragment) > 0]
    return layer, spam


def test_rejects_non_box_instruction(backend):
    """Only BoxOp instructions describe a layer whose noise can be learned."""
    circuit = QuantumCircuit(backend.num_qubits)
    circuit.cz(*PAIR)

    with pytest.raises(ValueError, match="BoxOps"):
        prepare_learning_program(backend, [circuit[0]])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_randomizations": 0}, "num_randomizations must be at least 1"),
        ({"shots_per_randomization": 0}, "shots_per_randomization must be at least 1"),
        ({"fragment_depths": [0, -1]}, "fragment_depths must all be non-negative"),
    ],
)
def test_rejects_invalid_quantities(backend, kwargs, match):
    """A negative depth in particular would collide with the decay-row sentinel downstream."""
    with pytest.raises(ValueError, match=match):
        prepare_learning_program(backend, [_boxed_cz(backend, name="layer")], **kwargs)


def test_program_shape(backend):
    """Shots are per randomization, and each fragment depth becomes its own program item.

    The depth-0 SPAM sequence has no repeatable fragment to bind, so it contributes the one
    remaining item.
    """
    depths = [2, 8, 32]
    program = prepare_learning_program(
        backend,
        [_boxed_cz(backend, name="layer")],
        num_randomizations=4,
        shots_per_randomization=16,
        fragment_depths=depths,
    )

    assert isinstance(program, QuantumProgram)
    assert program.shots == 16
    assert len(program.items) == len(depths) + 1


def test_defaults(backend):
    """The documented defaults, pinned so a change to them is deliberate."""
    program = prepare_learning_program(backend, [_boxed_cz(backend, name="layer")])

    assert program.shots == 128
    assert load(program.passthrough_data).num_randomizations == 32
    assert len(program.items) == 7


def test_passthrough_carries_the_experiment(backend):
    """Everything process_learning_result needs travels in the program itself."""
    program = prepare_learning_program(
        backend,
        [_boxed_cz(backend, name="layer")],
        num_randomizations=4,
        shots_per_randomization=16,
        fragment_depths=[2, 8],
    )
    mapper = load(program.passthrough_data)

    assert mapper.num_randomizations == 4
    assert mapper.paths
    assert mapper.instruction_sequences
    assert mapper.relations

    layer_paths, spam_paths = _split_paths(mapper.paths)
    assert all(len(path.repeatable_fragment) == 2 for path in layer_paths)
    assert len(spam_paths) == len(PAIR)


@pytest.mark.parametrize(("name", "expected"), [("layer", "layer"), (None, "L0")])
def test_gate_naming(backend, name, expected):
    """An annotated box is keyed by its reference; an unannotated one is named automatically."""
    program = prepare_learning_program(
        backend,
        [_boxed_cz(backend, name=name)],
        num_randomizations=2,
        shots_per_randomization=8,
        fragment_depths=[2],
    )
    gate_names = set(load(program.passthrough_data).fidelity_model.gate_set)

    assert gate_names == {expected, "P", "M"}


def test_paths_are_rank_reduced(backend):
    """The path set is linearly independent."""
    program = prepare_learning_program(
        backend,
        [_boxed_cz(backend, name="layer")],
        num_randomizations=2,
        shots_per_randomization=8,
        fragment_depths=[2],
    )
    mapper = load(program.passthrough_data)
    experiment = Experiment(fidelity_model=mapper.fidelity_model, paths=mapper.paths)

    assert len(mapper.paths) == experiment.design_matrix.rank


def test_measurement_carries_all_spam_noise(backend):
    """Depth-0 data cannot separate preparation from measurement, so only ``M`` is modelled."""
    program = prepare_learning_program(
        backend,
        [_boxed_cz(backend, name="layer")],
        num_randomizations=2,
        shots_per_randomization=8,
        fragment_depths=[2],
    )
    generators = load(program.passthrough_data).fidelity_model.generators

    assert len(generators["P"]) == 0
    # One single-qubit X generator per qubit of the subset, as LegacySolve requires.
    assert {
        (tuple(generator.indices), tuple(generator.paulis)) for generator in generators["M"]
    } == {((qubit,), (QubitSparsePauli.Pauli.X,)) for qubit in PAIR}


def test_supplied_measurement_layer_replaces_the_default(backend):
    """A box that measures becomes the measurement gate, rather than duplicating ``"M"``."""
    program = prepare_learning_program(
        backend,
        [_boxed_cz(backend), _boxed_measure(backend)],
        num_randomizations=2,
        shots_per_randomization=8,
        fragment_depths=[2],
    )
    model = load(program.passthrough_data).fidelity_model

    assert set(model.gate_set) == {"L0", "L1", "P"}
    assert model.gate_set["L1"].meas_idxs == frozenset(PAIR)
    # The 1-local measurement model applies to the supplied box, not to a synthesized "M".
    assert {
        (tuple(generator.indices), tuple(generator.paulis)) for generator in model.generators["L1"]
    } == {((qubit,), (QubitSparsePauli.Pauli.X,)) for qubit in PAIR}
    assert len(program.items) == 2


def test_supplied_measurement_layer_is_named_by_its_annotation(backend):
    """It is named like any other box, so its noise can be keyed by a reference of your choosing."""
    program = prepare_learning_program(
        backend,
        [_boxed_cz(backend, name="layer"), _boxed_measure(backend, name="readout")],
        num_randomizations=2,
        shots_per_randomization=8,
        fragment_depths=[2],
    )
    gate_names = set(load(program.passthrough_data).fidelity_model.gate_set)

    assert gate_names == {"layer", "readout", "P"}


@pytest.mark.parametrize(
    ("build_instructions", "match"),
    [
        (
            lambda backend: [
                _boxed_cz(backend),
                _boxed_measure(backend),
                _boxed_measure(backend, PAIR[::-1]),
            ],
            "At most one instruction may measure",
        ),
        (lambda backend: [_boxed_cz_and_measure(backend)], "non-trivial unitary part"),
        (
            lambda backend: [_boxed_cz(backend), _boxed_measure(backend, PAIR[:1])],
            "does not measure all qubits",
        ),
    ],
    ids=["two-measuring-boxes", "measurement-mixed-with-a-gate", "measures-part-of-the-subset"],
)
def test_rejects_measurement_layers_that_cannot_be_the_measurement_gate(
    backend, build_instructions, match
):
    """Rejected outright, rather than quietly added as a layer that no path probes."""
    with pytest.raises(ValueError, match=match):
        prepare_learning_program(backend, build_instructions(backend))
