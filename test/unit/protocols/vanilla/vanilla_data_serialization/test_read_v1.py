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

"""Version 1 payloads that must always deserialize to the same submission.

**Every payload in this module was written by hand and must never be regenerated.** Their whole
purpose is to be independent of the writer: a payload produced by the current writer can only show
that the reader and writer still agree with each other, whereas one written by hand also fails when
the writer is wrong.

Each is paired with a description of the submission it holds, written in terms of the instructions
given to a protocol function rather than of the format. **That description, not the payload, is what
a future reader should work from.** If the package's classes change so that the expected protocol
data below no longer expresses the same submission, translate the description into the new classes
and leave the payload untouched.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from qiskit_noise_learning.protocols.vanilla.vanilla_data import VanillaData
from qiskit_noise_learning.protocols.vanilla.vanilla_data_serialization import v1


@dataclass(frozen=True)
class FrozenPayload:
    """A hand-written payload, kept together with the submission it describes."""

    description: str
    """What submission this payload holds, in terms independent of the format and of the classes."""

    payload: dict[str, Any] = field(default_factory=dict)
    """The payload itself, exactly as version 1 writes it."""


# --------------------------------------------------------------------------------------------------

ONE_GATE = FrozenPayload(
    description="""
    A single submitted instruction, which became the gate named "layer".
    """,
    payload={"version": 1, "gate_names": ["layer"]},
)

THREE_GATES_OUT_OF_ORDER = FrozenPayload(
    description="""
    Three submitted instructions, which became the gates named "zebra", "apple" and "readout", in
    that order. The names are deliberately not in sorted order, because the order they were
    submitted in is what this format exists to carry: the gate names of the payload the protocol
    data sits inside are interned sorted, so the submitted order is recoverable from nowhere else.
    """,
    payload={"version": 1, "gate_names": ["zebra", "apple", "readout"]},
)


FROZEN = {
    "one_gate": (ONE_GATE, lambda: VanillaData(gate_names=["layer"])),
    "three_gates_out_of_order": (
        THREE_GATES_OUT_OF_ORDER,
        lambda: VanillaData(gate_names=["zebra", "apple", "readout"]),
    ),
}
"""Each frozen payload, against the protocol data it must read into."""


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_read_gives_the_described_submission(name):
    """Each hand-written payload reads into the protocol data its description names."""
    frozen, expected = FROZEN[name]

    assert v1.read(frozen.payload).gate_names == expected().gate_names
