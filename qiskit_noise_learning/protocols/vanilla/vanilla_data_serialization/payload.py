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

"""Writing and reading the vanilla protocol's data through a program's passthrough data."""

from collections.abc import Callable
from typing import Any

from ....circuit_generator.executor.data_mapper_serialization import (
    PAYLOAD_KEY,
    PROTOCOL_KEY,
    PayloadVersionError,
)
from ..vanilla_data import VanillaData
from . import v1

VANILLA_KEY = "vanilla"
"""The entry of the protocol data that the vanilla protocol owns."""

CURRENT_VERSION = v1.VERSION
"""The format version :func:`dump` writes."""

READERS: dict[int, Callable[[dict[str, Any]], VanillaData]] = {v1.VERSION: v1.read}
"""Every format version this release can read, and the function that reads it."""

RETIRED: dict[int, str] = {}
"""Format versions no longer readable, against the last release that could read them."""


def dump(passthrough_data: dict[str, Any], data: VanillaData) -> None:
    """Write vanilla protocol data into a program's passthrough data, where :func:`load` reads it.

    This format does not occupy a passthrough entry of its own: it is written inside the entry the
    serialized data mapper occupies, which must already be there.

    Args:
        passthrough_data: The passthrough data of a program the data mapper has been dumped into.
            Modified in place.
        data: The data to serialize.
    """
    passthrough_data[PAYLOAD_KEY].setdefault(PROTOCOL_KEY, {})[VANILLA_KEY] = v1.write(data)


def load(passthrough_data: dict[str, Any]) -> VanillaData:
    """Rebuild the vanilla protocol data carried by a program result's passthrough data.

    Args:
        passthrough_data: The passthrough data from the result of a program that :func:`dump` wrote
            data into.

    Returns:
        The data.

    Raises:
        PayloadVersionError: If the passthrough data carries no protocol data, if the protocol data
            it carries belongs to a different protocol, or if the vanilla data it carries was
            written at a version this release cannot read.
    """
    # Checked rather than assumed: this data has been through execution, and whoever submitted the
    # program chose what else went into it.
    payload = passthrough_data.get(PAYLOAD_KEY) if isinstance(passthrough_data, dict) else None
    protocols = payload.get(PROTOCOL_KEY) if isinstance(payload, dict) else None
    if not isinstance(protocols, dict):
        raise PayloadVersionError(
            f"Found no protocol data to read: the passthrough data has no '{PROTOCOL_KEY}' entry "
            f"under '{PAYLOAD_KEY}'. This result may come from a program that "
            "prepare_vanilla_program did not build."
        )

    entry = protocols.get(VANILLA_KEY)
    if not isinstance(entry, dict):
        raise PayloadVersionError(
            f"This program was built by a different protocol: its protocol data holds "
            f"{sorted(protocols)} rather than a '{VANILLA_KEY}' entry, so it has to be processed "
            "by the function pair that built it."
        )

    try:
        version = int(entry["version"])
    except (TypeError, KeyError, ValueError):
        raise PayloadVersionError(
            f"The '{VANILLA_KEY}' protocol data entry has no integer 'version', so there is no "
            "way to tell how to read it."
        ) from None

    if (reader := READERS.get(version)) is not None:
        return reader(entry)

    if version in RETIRED:
        raise PayloadVersionError(
            f"This protocol data was written in format version {version}, which this release of "
            f"qiskit-noise-learning no longer reads. Install qiskit-noise-learning "
            f"{RETIRED[version]} to read it."
        )
    if version > max(READERS):
        raise PayloadVersionError(
            f"This protocol data was written in format version {version}, which is newer than "
            "this release of qiskit-noise-learning reads. Upgrade qiskit-noise-learning to read it."
        )
    raise PayloadVersionError(
        f"This protocol data was written in format version {version}, which this release of "
        f"qiskit-noise-learning does not recognize. It reads version(s) {sorted(READERS)}."
    )
