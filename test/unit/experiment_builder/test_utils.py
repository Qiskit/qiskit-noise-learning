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
from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.experiment_builder.utils import generate_bases


@pytest.mark.parametrize(
    "graph", [CouplingMap.from_heavy_hex(11).graph, CouplingMap.from_heavy_square(11).graph]
)
def test_generate_bases(graph):
    """Test that generate_bases finds 9 basis strings for standard layouts."""
    assert len(generate_bases(graph)) == 9
