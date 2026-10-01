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

from qiskit_noise_learning.experiment_builder.experiment import Experiment
from qiskit_noise_learning.experiment_builder.stages import AddPaths


def test_adds_paths_from_iterators(gate_set_cz, make_cz_path):
    path_ix = make_cz_path("IX")
    path_xi = make_cz_path("XI")
    exp = Experiment(fidelity_model=gate_set_cz)

    result = AddPaths([path_ix, path_xi]).run(exp)

    assert result.paths == [path_ix, path_xi]


def test_drops_duplicate_paths(gate_set_cz, make_cz_path):
    path_ix = make_cz_path("IX")
    path_xi = make_cz_path("XI")
    path_yy = make_cz_path("YY")
    exp = Experiment(fidelity_model=gate_set_cz)

    # The second stage repeats path_xi, which the first stage already added, and the first stage
    # repeats path_ix as a distinct but equal object.
    pipeline = AddPaths([path_ix, make_cz_path("IX"), path_xi]) + AddPaths([path_xi, path_yy])
    result = pipeline.run(exp)

    assert result.paths == [path_ix, path_xi, path_yy]
