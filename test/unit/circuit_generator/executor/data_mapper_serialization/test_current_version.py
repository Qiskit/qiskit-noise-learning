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
"""The ``read`` and ``write`` functions of the current version.

Round trip tests for the current version.
"""

import typing

import pytest

from qiskit_noise_learning.circuit_generator import ExecutorDataMapper
from qiskit_noise_learning.circuit_generator.executor.data_mapper_serialization import v1

from .conftest import MAPPER_SHAPES, as_comparable, assert_same_payload, passthrough_problems
from .test_read_v1 import FROZEN as FROZEN_V1

SECTIONS = {
    "": v1.Payload,
    "layout": v1.LayoutPayload,
    "sequences": v1.SequencesPayload,
    "sequences.structure": v1.SequenceStructurePayload,
    "sequences.instructions": v1.InstructionsPayload,
    "paths": v1.PathsPayload,
    "paths.fidelity_indices": v1.FidelityIndicesPayload,
    "relations": v1.RelationsPayload,
    "model": v1.ModelPayload,
}
"""Where each documented section sits in a payload."""


def section_at(payload, path):
    """The part of ``payload`` that ``path`` names, ``""`` being the whole payload."""
    for key in filter(None, path.split(".")):
        payload = payload[key]
    return payload


@pytest.mark.parametrize("shape", MAPPER_SHAPES)
def test_writing_and_reading_preserves_a_mapper(example_mappers, shape):
    """A mapper written and read back carries everything it started with."""
    mapper = example_mappers[shape]
    assert as_comparable(v1.read(v1.write(mapper))) == as_comparable(mapper)


@pytest.mark.parametrize("shape", MAPPER_SHAPES)
def test_a_payload_is_passthrough_data(example_mappers, shape):
    """Every payload is something a program's passthrough data can carry."""
    problems = passthrough_problems(v1.write(example_mappers[shape]))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", SECTIONS, ids=lambda path: path or "payload")
def test_documented_entries_are_the_ones_written(example_mappers, path):
    """Each documented section lists exactly the entries the writer produces for it."""
    declared = SECTIONS[path]
    written = set(section_at(v1.write(example_mappers["built_experiment"]), path))
    assert not set(declared.__required_keys__) - written, "declares entries that are not written"
    undeclared = written - set(declared.__required_keys__) - set(declared.__optional_keys__)
    assert not undeclared, f"writes undeclared entries {sorted(undeclared)}"


def test_the_payload_resolves_its_own_sections():
    """The payload's forward references to the sections below it resolve."""
    assert set(typing.get_type_hints(v1.Payload)) == set(v1.Payload.__required_keys__)


def test_an_identity_model_writes_no_generators(example_mappers):
    """An identity model has no parameters of its own, so it omits the optional entries."""
    written = set(v1.write(example_mappers["identity_model"])["model"])
    assert written == set(v1.ModelPayload.__required_keys__)


def test_an_unwritable_model_is_refused(example_mappers):
    """A model this version cannot write is refused, naming the version that cannot write it."""

    class UnknownModel:
        pass

    mapper = example_mappers["no_optionals"]
    with pytest.raises(TypeError, match="UnknownModel"):
        v1.write(
            ExecutorDataMapper(
                item_sequence_indices=mapper.item_sequence_indices,
                item_creg_names=mapper.item_creg_names,
                item_clbit_qubit_idxs=mapper.item_clbit_qubit_idxs,
                instruction_sequences=mapper.instruction_sequences,
                num_randomizations=mapper.num_randomizations,
                fidelity_model=UnknownModel(),
            )
        )


@pytest.mark.parametrize("name", sorted(FROZEN_V1))
def test_writing_reproduces_the_frozen_payloads(name):
    """What the writer produces is what the frozen version 1 payloads hold.

    Change this to the latest version.
    """
    frozen, expected = FROZEN_V1[name]
    assert_same_payload(v1.write(expected()), frozen.payload)
