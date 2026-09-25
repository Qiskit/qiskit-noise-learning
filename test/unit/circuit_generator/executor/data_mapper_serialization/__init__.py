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

"""Tests for the ExecutorDataMapper serialization routine.

These tests are structured as follows:
* test_payload.py tests the ``dump``/``load`` functions - centered around correct versioning
  and errors.
* test_current_version.py tests the ``read``/``write`` functions of the current version, e.g.
  preservation of the data mapper under a roundtrip through the functions.
* test_read_v*.py, for each supported version, contains frozen payloads in the format of the given
  version, and tests that ``read`` successfully deserializes.
"""
