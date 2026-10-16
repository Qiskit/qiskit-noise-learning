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

"""Serialization of the vanilla protocol's additional passthrough data.

The versioning system policy, and the rules for defining and retiring a version, are the same as
are documented in
:mod:`~qiskit_noise_learning.circuit_generator.executor.data_mapper_serialization`: a ``read``
function per version in a ``v*.py`` file, only the latest version writing, and the current,
readable and retired version numbers declared in ``payload.py``.
"""

from .payload import dump, load
