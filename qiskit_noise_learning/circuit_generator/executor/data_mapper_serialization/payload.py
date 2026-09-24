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

"""Writing and reading a data mapper through a program's passthrough data."""

from collections.abc import Callable
from typing import Any

from ..executor_data_mapper import ExecutorDataMapper
from . import v1

PAYLOAD_KEY = "qiskit_noise_learning"
"""The entry under which a program's passthrough data carries the serialized data mapper."""

CURRENT_VERSION = v1.VERSION
"""The format version :func:`dump` writes."""

READERS: dict[int, Callable[[dict[str, Any]], ExecutorDataMapper]] = {v1.VERSION: v1.read}
"""Every format version this release can read, and the function that reads it."""

RETIRED: dict[int, str] = {}
"""Format versions no longer readable, against the last release that could read them."""


class PayloadVersionError(ValueError):
    """Raised when serialized data cannot be read by this release."""


def dump(mapper: ExecutorDataMapper) -> dict[str, Any]:
    """Serialize a data mapper for a program's passthrough data.

    Its values are only the types passthrough data accepts: arrays, strings, whole numbers,
    booleans, ``None``, and lists and dictionaries of those.

    Args:
        mapper: The data mapper to serialize.

    Returns:
        The passthrough data entry carrying the mapper, at :data:`CURRENT_VERSION`.
    """
    return {PAYLOAD_KEY: v1.write(mapper)}


def load(passthrough_data: dict[str, Any]) -> ExecutorDataMapper:
    """Rebuild the data mapper carried by a program result's passthrough data.

    Args:
        passthrough_data: The passthrough data from the result of a program that :func:`dump` wrote
            a mapper into.

    Returns:
        The data mapper. Fields whose order carries no meaning, such as the relations, may come back
        ordered differently to the mapper that was written.

    Raises:
        PayloadVersionError: If the passthrough data carries no serialized mapper, or carries one
            written at a version this release cannot read.
    """
    # Checked rather than assumed: this data has been through execution, and whoever submitted
    # the program chose what else went into it.
    if not isinstance(passthrough_data, dict) or PAYLOAD_KEY not in passthrough_data:
        raise PayloadVersionError(
            f"Found no noise learning data to read: the passthrough data has no "
            f"'{PAYLOAD_KEY}' entry. This result may come from a program that this package did "
            "not generate."
        )
    payload = passthrough_data[PAYLOAD_KEY]

    try:
        version = int(payload["version"])
    except (TypeError, KeyError, ValueError):
        raise PayloadVersionError(
            f"The '{PAYLOAD_KEY}' passthrough data entry has no whole-number 'version', so there "
            "is no way to tell how to read it."
        ) from None

    if (reader := READERS.get(version)) is not None:
        return reader(payload)

    if version in RETIRED:
        raise PayloadVersionError(
            f"This data was written in format version {version}, which this release of "
            f"qiskit-noise-learning no longer reads. Install qiskit-noise-learning "
            f"{RETIRED[version]} to read it."
        )
    if version > max(READERS):
        raise PayloadVersionError(
            f"This data was written in format version {version}, which is newer than this release "
            f"of qiskit-noise-learning reads. Upgrade qiskit-noise-learning to read it."
        )
    raise PayloadVersionError(
        f"This data was written in format version {version}, which this release of "
        f"qiskit-noise-learning does not recognize. It reads version(s) {sorted(READERS)}."
    )
