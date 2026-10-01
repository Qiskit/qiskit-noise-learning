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

from qiskit_noise_learning.sequences import ApplyGate


def test_equal_gates_hash_equally():
    """Two instructions applying the same gate are interchangeable as mapping keys."""
    assert hash(ApplyGate("CZ")) == hash(ApplyGate("CZ"))
    assert {ApplyGate("CZ"): "first", ApplyGate("CZ"): "second"} == {ApplyGate("CZ"): "second"}


def test_distinct_gates_are_distinct_keys():
    """Instructions applying different gates do not collapse into one key."""
    assert len({ApplyGate("L0"), ApplyGate("L1"), ApplyGate("L0")}) == 2
