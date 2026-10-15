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
    PAYLOAD_KEY,
    PROTOCOL_KEY,
    PayloadVersionError,
)
from qiskit_noise_learning.protocols.vanilla.vanilla_data import VanillaData
from qiskit_noise_learning.protocols.vanilla.vanilla_data_serialization import (
    dump,
    load,
    payload,
    v1,
)

VANILLA_KEY = payload.VANILLA_KEY


def protocol_entry(entry):
    """Passthrough data whose protocol data holds ``entry`` as the vanilla protocol's."""
    return {PAYLOAD_KEY: {PROTOCOL_KEY: {VANILLA_KEY: entry}}}


@pytest.fixture()
def passthrough_data():
    """Passthrough data holding a serialized data mapper, as a dumped program's does."""
    return {PAYLOAD_KEY: {"version": 1, "library_version": "0.0.0"}}


@pytest.fixture()
def data():
    """Protocol data whose gate names are deliberately not in sorted order."""
    return VanillaData(gate_names=["zebra", "apple", "readout"])


def test_dump_occupies_one_entry_of_the_protocol_data(passthrough_data, data):
    """A dump takes up the one entry this protocol owns, leaving every other entry alone."""
    dump(passthrough_data, data)

    assert list(passthrough_data) == [PAYLOAD_KEY]
    assert set(passthrough_data[PAYLOAD_KEY]) == {"version", "library_version", PROTOCOL_KEY}
    assert list(passthrough_data[PAYLOAD_KEY][PROTOCOL_KEY]) == [VANILLA_KEY]


def test_dump_leaves_another_protocols_entry_alone(passthrough_data, data):
    """The protocol data entry is shared, so writing into it is not writing over it."""
    passthrough_data[PAYLOAD_KEY][PROTOCOL_KEY] = {"another_protocol": {"version": 7}}
    dump(passthrough_data, data)

    protocols = passthrough_data[PAYLOAD_KEY][PROTOCOL_KEY]
    assert protocols["another_protocol"] == {"version": 7}
    assert load(passthrough_data).gate_names == data.gate_names


def test_dump_writes_the_current_version(passthrough_data, data):
    """What is dumped is written at the version this release writes."""
    dump(passthrough_data, data)

    entry = passthrough_data[PAYLOAD_KEY][PROTOCOL_KEY][VANILLA_KEY]

    assert entry["version"] == payload.CURRENT_VERSION


def test_dumping_and_loading_preserves_the_submitted_order(passthrough_data, data):
    """The order is the whole point of recording the names, so it survives a round trip."""
    dump(passthrough_data, data)

    assert load(passthrough_data).gate_names == ["zebra", "apple", "readout"]


def test_a_dump_loads_alongside_other_entries(passthrough_data, data):
    """A submitter's own passthrough entries survive, and do not disturb the reading."""
    passthrough_data["submitted_by"] = "someone else"
    dump(passthrough_data, data)

    assert load(passthrough_data).gate_names == data.gate_names


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
        pytest.param({PAYLOAD_KEY: "not a mapping"}, id="payload not a mapping"),
        pytest.param({PAYLOAD_KEY: {"version": 1}}, id="a mapper but no protocol data"),
        pytest.param(
            {PAYLOAD_KEY: {PROTOCOL_KEY: "not a mapping"}}, id="protocol data not a mapping"
        ),
    ],
)
def test_load_refuses_data_carrying_no_protocol_data(passthrough_data):
    """Passthrough data with no protocol data is refused, saying why it may be missing."""
    with pytest.raises(PayloadVersionError, match="did not build"):
        load(passthrough_data)


@pytest.mark.parametrize(
    "protocols",
    [
        pytest.param({}, id="no protocol at all"),
        pytest.param({"gauge_aware": {"version": 1}}, id="another protocol"),
        pytest.param({VANILLA_KEY: "not a mapping"}, id="entry not a mapping"),
    ],
)
def test_load_refuses_a_program_built_by_another_protocol(protocols):
    """Processing is paired with building, so another protocol's program is refused by name."""
    with pytest.raises(PayloadVersionError, match="built by a different protocol"):
        load({PAYLOAD_KEY: {PROTOCOL_KEY: protocols}})


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
        load(protocol_entry(entry))


def test_load_refuses_a_newer_version_and_says_to_upgrade():
    """A payload from a later release names the action that would let it be read."""
    entry = {"version": max(payload.READERS) + 1}
    with pytest.raises(PayloadVersionError, match="Upgrade qiskit-noise-learning"):
        load(protocol_entry(entry))


def test_load_refuses_a_retired_version_and_names_a_release(monkeypatch):
    """A retired payload names a release to install, rather than only refusing."""
    monkeypatch.setitem(payload.RETIRED, 0, "0.4.2")
    with pytest.raises(PayloadVersionError, match="Install qiskit-noise-learning 0.4.2"):
        load(protocol_entry({"version": 0}))


def test_load_refuses_a_version_missing_between_two_readable_ones(monkeypatch):
    """A gap in the readable versions is refused as unrecognized, not as newer or as retired."""
    monkeypatch.setitem(payload.READERS, max(payload.READERS) + 2, v1.read)
    entry = {"version": max(payload.READERS) - 1}
    with pytest.raises(PayloadVersionError, match="does not recognize"):
        load(protocol_entry(entry))
