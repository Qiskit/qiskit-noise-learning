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

"""Serialization of the vanilla protocol's data.

The main functions are :func:`dump` and :func:`load`, which write a :class:`~.VanillaData` into a
program's passthrough data and read it back. The serialization occupies the ``"vanilla"`` entry of
the ``"protocol"`` entry of the reserved ``"qiskit_noise_learning"`` key that the data mapper's
payload owns, giving a passthrough entry of the form ``{"qiskit_noise_learning": {"protocol":
{"vanilla": {"version": k, ...}}}}``, where ``k`` is an integer specifying the serialization version
of *this* format, independent of the version of the payload it sits inside.

The ``"protocol"`` entry is shared with every other protocol, each of which owns one entry in it and
versions it independently: this format is the vanilla protocol's alone, and nothing about it is
promised to any other. Which entry a program carries is therefore what identifies the protocol that
built it, and :func:`load` refuses a program built by a different one.

The versioning system policy, and the rules for defining and retiring a version, are the same as
that payload's and are documented in
:mod:`~qiskit_noise_learning.circuit_generator.executor.data_mapper_serialization`: a ``read``
function per version in a ``v*.py`` file, only the latest version writing, and the current,
readable and retired version numbers declared in ``payload.py``. A new version is defined when
:class:`~.VanillaData` is modified in a way that requires a change to the payload format.

Version problems raise that module's ``PayloadVersionError``, so a single ``except`` clause catches
an unreadable payload whichever format carried it.
"""

from .payload import dump, load
