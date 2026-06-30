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

from qiskit_noise_learning.experiment_builder.experiment import Experiment


@pytest.fixture()
def simple_experiment_with_paths(gate_set_cz, make_cz_path):
    """An experiment with a fidelity model and two unbound paths."""
    return Experiment(
        fidelity_model=gate_set_cz,
        paths=[make_cz_path("IX"), make_cz_path("XI")],
    )


@pytest.fixture()
def experiment_with_sequences(gate_set_cz, make_cz_path):
    """An experiment with paths, instruction sequences, relations, and multipliers."""
    unbound_path_ix = make_cz_path("IX")
    unbound_path_xi = make_cz_path("XI")
    seq_ix = unbound_path_ix.to_instruction_sequence()
    seq_xi = unbound_path_xi.to_instruction_sequence()
    return Experiment(
        fidelity_model=gate_set_cz,
        paths=[unbound_path_ix, unbound_path_xi],
        instruction_sequences=[seq_ix, seq_xi],
        relations={(0, 0), (1, 1)},
        randomization_multipliers=[1, 1],
    )
