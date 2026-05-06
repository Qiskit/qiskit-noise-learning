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

import pytest
from pydantic import ValidationError

from qiskit_noise_learning.noise_learner import LearningOptions


def test_default_options():
    """Test default values of LearningOptions."""
    assert LearningOptions().num_randomizations == 32
    assert LearningOptions().shots_per_randomizations == 128
    assert LearningOptions().depths == [0, 1, 2, 4, 16, 32]
    assert LearningOptions().k_locality == 2
    assert LearningOptions().path_generator == "even_depth"
    assert LearningOptions().analyzer == "standard"


def test_custom_values():
    """Test learning options with custom values."""
    opts = LearningOptions(
        num_randomizations=10,
        shots_per_randomizations=50,
        depths=[0, 2, 4],
        k_locality=3,
    )
    assert opts.num_randomizations == 10
    assert opts.shots_per_randomizations == 50
    assert opts.depths == [0, 2, 4]
    assert opts.k_locality == 3


def test_valid_num_randomizations():
    """Test edge cases for valid number of randomizations."""
    assert LearningOptions(num_randomizations=1).num_randomizations == 1


def test_invalid_num_randomizations_zero_rejected():
    """Test non-positive number of randomizations are rejected."""
    with pytest.raises(ValidationError):
        LearningOptions(num_randomizations=0)


def test_valid_shots_per_randomizations():
    """Test edge cases for valid shots per randomization."""
    assert LearningOptions(shots_per_randomizations=1).shots_per_randomizations == 1


def test_invalid_shots_per_randomizations():
    """Test non-positive shots per randomization is rejected."""
    with pytest.raises(ValidationError):
        LearningOptions(shots_per_randomizations=0)


def test_k_locality_zero_allowed():
    """Test edge cases for k-locality."""
    assert LearningOptions(k_locality=0).k_locality == 0


def test_k_locality_negative_rejected():
    """Test negative k-locality is rejected."""
    with pytest.raises(ValidationError):
        LearningOptions(k_locality=-1)


def test_valid_depths():
    """Test edge cases for valid depths."""
    assert LearningOptions(depths=[0]).depths == [0]
    assert LearningOptions(depths=[]).depths == []
    assert LearningOptions(depths=[0, 100, 1000]).depths == [0, 100, 1000]


def test_invalid_depths():
    """Test negative depths are rejected."""
    with pytest.raises(ValidationError, match=">= 0"):
        LearningOptions(depths=[-1])

    with pytest.raises(ValidationError, match=">= 0"):
        LearningOptions(depths=[0, 1, -2, 4])


def test_path_generator_invalid_rejected():
    """Test invalid path generator names are rejected."""
    with pytest.raises(ValidationError):
        LearningOptions(path_generator="odd_depth")


def test_analyzer_invalid_rejected():
    """Test invalid analyzer names are rejected."""
    with pytest.raises(ValidationError):
        LearningOptions(analyzer="fancy")
