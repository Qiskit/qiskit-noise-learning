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


"""Serialization of the executor data mapper.

The main functions are :func:`dump` and :func:`load`, for serialization and deserialization
respectively. The serialization format is a dictionary of the form
``{"qiskit_noise_learning": {"version": k, ...}}``, where ``k`` is an integer specifying the
serialization version, and ``...`` represents other ``str`` to object mappings, which will be
arrays, strings, integers, booleans, ``None``, or nested containers of lists or dictionaries of
these. The serialization is intended to be used as executor passthrough data.

The versioning system policy is as follows:
* A version is specified in a file ``v*.py``, containing a ``read`` function for deserializing the
  inner dictionary above.
* The latest version file also contains a ``write`` function for serialization: as such only the
  latest version ever writes.
* The current version number, supported previous version numbers for reading, and the retired
  version numbers, are specified in ``payload.py``. The application of the correct read version, and
  error raising for unsupported version numbers, is managed by :func:`load`.

A new version should be defined when the :class:`ExecutorDataMapper` is modified in a way that
fundamentally changes the payload format. For example:
* Adding a new optional argument to the data mapper will not require a new version: the current
  ``read`` function can be updated to assign the value ``None`` if the field is not present.

A version should be retired when it cannot be reconciled with the current version of
:class:`ExecutorDataMapper`. For example:
* If the quantum program layout information is changed, but in a recoverable way, it is not
  necessary to retire the version.
* However, if the latest version of the layout becomes incomparable to the payload version, it must
  be retired.

If a new version is specified, if possible, the ``read`` function for previous versions should be
updated to output the latest data mapper format. If this is not possible, the entire version file
should be deleted. If the ``read`` of the superseded version is still supported, its ``write``
function should be deleted. The version flags in ``payload.py`` should be updated to reflect any
changes.

Lastly, any supported version should have unit tests verifying instances of correct payload
deserialization, where correctness is defined in terms of an expected deserialization.
"""

from .payload import PayloadVersionError, dump, load
