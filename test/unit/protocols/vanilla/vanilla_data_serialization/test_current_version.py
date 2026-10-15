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

import pytest

from qiskit_noise_learning.protocols.vanilla.vanilla_data import VanillaData
from qiskit_noise_learning.protocols.vanilla.vanilla_data_serialization import v1

from .test_read_v1 import FROZEN as FROZEN_V1

GATE_NAMES = [
    pytest.param([], id="no gates"),
    pytest.param(["layer"], id="one gate"),
    pytest.param(["zebra", "apple", "readout"], id="not in sorted order"),
]
"""The submissions the format has to handle."""


@pytest.mark.parametrize("gate_names", GATE_NAMES)
def test_writing_and_reading_preserves_protocol_data(gate_names):
    """Protocol data written and read back carries everything it started with."""
    data = VanillaData(gate_names=gate_names)

    assert v1.read(v1.write(data)).gate_names == data.gate_names


@pytest.mark.parametrize("gate_names", GATE_NAMES)
def test_a_payload_is_passthrough_data(gate_names):
    """Every payload is something a program's passthrough data can carry.

    The traps are a tuple, which returns as a list, a numpy scalar, which reaches the transport
    unconverted, and a dictionary key that is not a string.
    """
    written = v1.write(VanillaData(gate_names=gate_names))

    assert all(type(key) is str for key in written)
    assert type(written["version"]) is int
    assert type(written["gate_names"]) is list
    assert all(type(name) is str for name in written["gate_names"])


def test_documented_entries_are_the_ones_written():
    """The documented payload lists exactly the entries the writer produces."""
    written = set(v1.write(VanillaData(gate_names=["layer"])))

    assert written == set(v1.Payload.__required_keys__)


@pytest.mark.parametrize("name", sorted(FROZEN_V1))
def test_writing_reproduces_the_frozen_payloads(name):
    """What the writer produces is what the frozen version 1 payloads hold.

    Change this to the latest version.
    """
    frozen, expected = FROZEN_V1[name]

    assert v1.write(expected()) == frozen.payload
