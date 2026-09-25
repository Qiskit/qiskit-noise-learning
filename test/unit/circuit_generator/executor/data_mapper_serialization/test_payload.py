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
"""The ``dump`` and ``load`` functions.

These test the version handling and error raising - tests for correctness of the supported payload
version is not tested here.
"""

import pytest

from qiskit_noise_learning.circuit_generator.executor.data_mapper_serialization import (
    PayloadVersionError,
    dump,
    load,
    payload,
    v1,
)

from .conftest import as_comparable


def test_dump_occupies_one_entry(example_mappers):
    """A dumped mapper takes up a single, named entry, leaving the rest to whoever submits it."""
    assert list(dump(example_mappers["full"])) == [payload.PAYLOAD_KEY]


def test_dump_writes_the_current_version(example_mappers):
    """What is dumped is written at the version this release writes."""
    assert dump(example_mappers["full"])[payload.PAYLOAD_KEY]["version"] == payload.CURRENT_VERSION


def test_a_dumped_mapper_loads_alongside_other_entries(example_mappers):
    """A submitter's own passthrough entries survive, and do not disturb the reading."""
    mapper = example_mappers["full"]
    merged = {"submitted_by": "someone else", **dump(mapper)}
    assert set(merged) == {"submitted_by", payload.PAYLOAD_KEY}
    assert as_comparable(load(merged)) == as_comparable(mapper)


def test_the_current_version_can_be_read():
    """Whatever version is written has a reader, which is what makes a dump readable at all."""
    assert payload.CURRENT_VERSION in payload.READERS


def test_no_version_is_both_readable_and_retired():
    """A version is either readable or retired, never recorded as both."""
    assert not set(payload.READERS) & set(payload.RETIRED)


@pytest.mark.parametrize(
    "passthrough_data",
    [
        pytest.param(None, id="no passthrough data"),
        pytest.param({}, id="empty passthrough data"),
        pytest.param({"someone_elses": 1}, id="only another entry"),
        pytest.param("not a mapping", id="not a mapping"),
    ],
)
def test_load_refuses_data_carrying_no_mapper(passthrough_data):
    """Passthrough data with no serialized mapper is refused, saying why it may be missing."""
    with pytest.raises(PayloadVersionError, match="did not generate"):
        load(passthrough_data)


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({}, id="version absent"),
        pytest.param({"version": "one"}, id="version not a number"),
        pytest.param({"version": None}, id="version is None"),
    ],
)
def test_load_refuses_an_entry_without_a_version(entry):
    """An entry that does not say which format it is in cannot be read."""
    with pytest.raises(PayloadVersionError, match="integer 'version'"):
        load({payload.PAYLOAD_KEY: entry})


def test_load_refuses_a_newer_version_and_says_to_upgrade():
    """A payload from a later release names the action that would let it be read."""
    with pytest.raises(PayloadVersionError, match="Upgrade qiskit-noise-learning"):
        load({payload.PAYLOAD_KEY: {"version": max(payload.READERS) + 1}})


def test_load_refuses_a_retired_version_and_names_a_release(monkeypatch):
    """A retired payload names a release to install, rather than only refusing."""
    monkeypatch.setitem(payload.RETIRED, 0, "0.4.2")
    with pytest.raises(PayloadVersionError, match="Install qiskit-noise-learning 0.4.2"):
        load({payload.PAYLOAD_KEY: {"version": 0}})


def test_load_refuses_a_version_missing_between_two_readable_ones(monkeypatch):
    """A gap in the readable versions is refused as unrecognized, not as newer or as retired."""
    monkeypatch.setitem(payload.READERS, max(payload.READERS) + 2, v1.read)
    with pytest.raises(PayloadVersionError, match="does not recognize"):
        load({payload.PAYLOAD_KEY: {"version": max(payload.READERS) - 1}})
