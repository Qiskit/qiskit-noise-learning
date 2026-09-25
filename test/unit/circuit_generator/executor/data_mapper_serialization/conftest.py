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
"""Helpers shared by the serialization tests."""

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from samplomatic.annotations import Twirl

from qiskit_noise_learning.circuit_generator import ExecutorDataMapper
from qiskit_noise_learning.circuit_generator.executor import ExecutorCircuitGenerator
from qiskit_noise_learning.experiment_builder import (
    BindFragmentDepths,
    CompleteSequences,
    EvenDepthVanillaPaths,
    Experiment,
    GenerateInstructionSequences,
    IdentifyRelations,
    MergeInstructionSequences,
)
from qiskit_noise_learning.gate_sets import QiskitGateSet
from qiskit_noise_learning.models import IdentityFidelityModel, PauliLindbladModel
from qiskit_noise_learning.sequences import FidelityIndex, PartialPauliPermutation

# --------------------------------------------------------------------------------------------------
# Example data mappers
# --------------------------------------------------------------------------------------------------

MAPPER_SHAPES = [
    "built_experiment",
    "full",
    "identity_model",
    "no_optionals",
    "unbound_sequence",
    "empty_not_absent",
    "two_registers_in_one_item",
]
"""The keys of :func:`example_mappers`, for parametrizing a test over every shape."""


@pytest.fixture(scope="session")
def built_experiment_mapper():
    """A data mapper from a real experiment, built the way a learning run builds one.

    Every other example is assembled by hand, which is fine for covering a shape but says nothing
    about what the package actually produces. This one runs the experiment builder over a
    four-qubit gate set with a two-gate CZ layer, and takes the mapper from the circuit generator,
    so it holds merged sequences, the Pauli permutations that completing them introduces, and
    relations and paths in the quantities a real run has. It builds in well under a second, and is
    shared across the session because nothing here modifies a mapper.
    """
    gate_set = QiskitGateSet(4)
    layer = QuantumCircuit(4)
    with layer.box([Twirl()]):
        layer.cz(0, 1)
        layer.cz(2, 3)
    gate_set.add_box_as_gate(layer[0], name="cz_layer")

    pipeline = (
        EvenDepthVanillaPaths()
        + GenerateInstructionSequences()
        + MergeInstructionSequences()
        + IdentifyRelations()
        + CompleteSequences()
        + BindFragmentDepths([0, 1, 2])
    )
    experiment = pipeline.run(
        Experiment(
            fidelity_model=PauliLindbladModel.k_local(gate_set, k=2), shots=64, randomizations=4
        )
    )
    _, mapper = ExecutorCircuitGenerator(gate_set).generate(experiment)
    return mapper


@pytest.fixture()
def example_mappers(gate_set_cz, make_cz_path, built_experiment_mapper):
    """A data mapper of every shape the format has to handle, keyed by :data:`MAPPER_SHAPES`."""
    path = make_cz_path("IX")
    sequence = path.to_instruction_sequence().complete().bind_at(1)
    layout = {
        "item_sequence_indices": [[0]],
        "item_creg_names": [["meas"]],
        "item_clbit_qubit_idxs": [{"meas": np.arange(2)}],
        "num_randomizations": 4,
    }
    full = {
        **layout,
        "instruction_sequences": [sequence],
        "fidelity_model": PauliLindbladModel.k_local(gate_set_cz, k=2),
        "paths": [path.bind_at(1)],
        "relations": {(0, 0)},
    }
    mappers = {
        "built_experiment": built_experiment_mapper,
        "full": ExecutorDataMapper(**full),
        "identity_model": ExecutorDataMapper(
            **{**full, "fidelity_model": IdentityFidelityModel(gate_set_cz)}
        ),
        "no_optionals": ExecutorDataMapper(**layout, instruction_sequences=[sequence]),
        "unbound_sequence": ExecutorDataMapper(**layout, instruction_sequences=[sequence.unbind()]),
        "empty_not_absent": ExecutorDataMapper(**{**full, "paths": [], "relations": set()}),
        "two_registers_in_one_item": ExecutorDataMapper(
            item_sequence_indices=[[0]],
            item_creg_names=[["meas_0", "meas_1"]],
            item_clbit_qubit_idxs=[{"meas_0": np.array([1]), "meas_1": np.array([0, 1])}],
            num_randomizations=4,
            instruction_sequences=[sequence],
        ),
    }
    assert sorted(mappers) == sorted(MAPPER_SHAPES), "MAPPER_SHAPES is out of step with the fixture"
    return mappers


# --------------------------------------------------------------------------------------------------
# Comparing data mappers
#
# Rather than walk two mappers together, each is turned into plain data and the two compared with
# ``==``. That keeps the comparison in one readable place, and makes what is compared explicit.
# --------------------------------------------------------------------------------------------------


def as_comparable(mapper):
    """Everything a data mapper carries, as plain data.

    Equality of two of these is the promise the format makes: every field agrees, with set-valued
    fields compared as sets, so a round trip that reorders the relations has still preserved
    everything.
    """
    return {
        "num_randomizations": mapper.num_randomizations,
        "item_sequence_indices": [[int(i) for i in row] for row in mapper.item_sequence_indices],
        "item_creg_names": [list(names) for names in mapper.item_creg_names],
        "item_clbit_qubit_idxs": [
            {name: list(idxs) for name, idxs in item.items()}
            for item in mapper.item_clbit_qubit_idxs
        ],
        "relations": mapper.relations,
        "instruction_sequences": [_as_fragments(s) for s in mapper.instruction_sequences],
        "paths": None if mapper.paths is None else [_as_fragments(p) for p in mapper.paths],
        "model": _model_as_comparable(mapper.fidelity_model),
    }


def _as_fragments(sequence):
    """An instruction sequence or a path, as its three fragments and its depth."""
    return (
        [_element_as_comparable(e) for e in sequence.start_fragment],
        [_element_as_comparable(e) for e in sequence.repeatable_fragment],
        [_element_as_comparable(e) for e in sequence.end_fragment],
        sequence.fragment_depth,
    )


def _element_as_comparable(element):
    """One instruction or fidelity index, as plain data.

    A fidelity index is spelled out field by field on purpose: its own equality compares only four
    of its eight fields, so comparing with ``==`` would pass while the other four were corrupted.
    """
    if isinstance(element, FidelityIndex):
        input_pauli, output_pauli = element.transition
        return (
            element.gate_name,
            element.pauli,
            input_pauli,
            output_pauli,
            element.sign_flip,
            sorted(element.in_z_idxs),
            sorted(element.out_z_idxs),
            sorted(element.meas_idxs),
        )
    if isinstance(element, PartialPauliPermutation):
        return ("permutation", list(element.partial_permutation_indices))
    return ("gate", element.gate_name)


def _model_as_comparable(model):
    """A fidelity model and its gate set, as plain data, or ``None``."""
    if model is None:
        return None
    gate_set = model.gate_set.model_gate_set
    return {
        "type": type(model).__name__,
        "num_qubits": gate_set.num_qubits,
        "qubit_subset": sorted(gate_set.qubit_subset),
        "coupling_map": sorted(map(tuple, gate_set.coupling_map.get_edges())),
        "gates": {name: _gate_as_comparable(gate_set[name]) for name in gate_set},
        # Only a Pauli-Lindblad model has these; an identity model has no parameters of its own.
        "generators": dict(model.generators) if hasattr(model, "generators") else None,
        "noise_site": dict(model.noise_site) if hasattr(model, "noise_site") else None,
    }


def _gate_as_comparable(gate):
    """One model gate, as plain data."""
    return (
        sorted(gate.qubit_idxs),
        sorted(gate.meas_idxs),
        sorted(gate.prep_idxs),
        [(tuple(idxs), clifford) for idxs, clifford in gate.cliffords],
    )


# --------------------------------------------------------------------------------------------------
# Checking a payload, whichever version wrote it
# --------------------------------------------------------------------------------------------------

WIRE_DTYPES = frozenset(
    ["float16", "float32", "float64", "int8", "int16", "int32", "int64"]
    + ["uint8", "uint16", "uint32", "uint64", "bool", "complex64", "complex128"]
)
"""The array dtypes the transport can encode.

Mirrors ``SupportedDtypesExtended`` in ``ibm_quantum_schemas``, listed here rather than imported so
that these tests do not depend on a package this one does not declare.
"""


def passthrough_problems(node, path="payload"):
    """Return every reason a payload would not survive a program's passthrough data.

    Catches the three traps the format has to avoid: a tuple, which returns as a list; a numpy
    scalar, which is neither an array nor a Python number and so reaches the transport unconverted;
    and a dictionary key that is not a string.
    """
    if isinstance(node, dict):
        problems = [
            f"{path}: key {key!r} is a {type(key).__name__}, not a str"
            for key in node
            if not isinstance(key, str)
        ]
        for key, value in node.items():
            problems += passthrough_problems(value, f"{path}[{key!r}]")
        return problems
    if isinstance(node, tuple):
        return [f"{path}: a tuple, which would return as a list"]
    if isinstance(node, list):
        return [p for n, v in enumerate(node) for p in passthrough_problems(v, f"{path}[{n}]")]
    if isinstance(node, np.ndarray):
        if node.dtype.name not in WIRE_DTYPES:
            return [f"{path}: dtype {node.dtype.name} is not one the transport encodes"]
        return []
    if isinstance(node, np.generic):
        return [f"{path}: numpy scalar {type(node).__name__}, not a Python scalar"]
    if not isinstance(node, str | int | bool | type(None)):
        return [f"{path}: {type(node).__name__} is not a passthrough data type"]
    return []


def assert_same_payload(actual, expected, path="payload"):
    """Assert two payloads hold the same entries, arrays compared by value and by dtype."""
    if isinstance(expected, dict):
        assert set(actual) == set(
            expected
        ), f"{path}: {sorted(set(actual) ^ set(expected))} is in one and not the other"
        for key in expected:
            assert_same_payload(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(expected, np.ndarray):
        assert (
            actual.dtype == expected.dtype
        ), f"{path}: dtype {actual.dtype}, want {expected.dtype}"
        assert np.array_equal(actual, expected), f"{path}: {list(actual)}, want {list(expected)}"
    else:
        assert actual == expected, f"{path}: {actual!r}, want {expected!r}"
