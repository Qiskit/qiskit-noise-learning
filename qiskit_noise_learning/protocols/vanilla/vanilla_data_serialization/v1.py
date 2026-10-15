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

"""Version 1 of the vanilla protocol's serialized data."""

from typing import TypedDict

from ..vanilla_data import VanillaData

VERSION = 1
"""The version number written into payloads by this module."""


class Payload(TypedDict):
    """Serialized vanilla protocol data."""

    version: int
    """The format version."""

    gate_names: list[str]
    """The gate name of every submitted instruction, in submission order."""


def write(data: VanillaData) -> Payload:
    """Serialize the vanilla protocol's data.

    Args:
        data: The data to serialize.

    Returns:
        The payload, at :data:`VERSION`.
    """
    return {"version": VERSION, "gate_names": list(data.gate_names)}


def read(payload: Payload) -> VanillaData:
    """Rebuild the data that :func:`write` was given.

    Args:
        payload: A payload written by :func:`write`.

    Returns:
        The data.
    """
    return VanillaData(gate_names=[str(name) for name in payload["gate_names"]])
