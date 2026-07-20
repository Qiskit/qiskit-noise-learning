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

import numpy as np
import pytest

from qiskit_noise_learning.analysis.utils import predicted_path_decays
from qiskit_noise_learning.sequences import Path


def test_predicted_path_decays_base_and_intercept(make_cz_path, make_fidelity_model_data):
    p = make_cz_path("XI")
    rate = 0.05
    model, model_data = make_fidelity_model_data([p], rate_default=rate)
    bases, intercepts = predicted_path_decays(model, model_data, [p])
    # base = product of repeatable-fragment fidelities = exp(-rate * n_repeatable).
    assert bases[p] == pytest.approx(np.exp(-rate * len(p.repeatable_fragment)))
    # intercept = product over start+end fragments = exp(-rate * (n_start + n_end)).
    n_spam = len(p.start_fragment) + len(p.end_fragment)
    assert intercepts[p] == pytest.approx(np.exp(-rate * n_spam))


def test_predicted_path_decays_raises_on_bound_path(make_cz_path, make_fidelity_model_data):
    p = make_cz_path("XI")
    model, model_data = make_fidelity_model_data([p])
    with pytest.raises(ValueError, match="unbound decay paths"):
        predicted_path_decays(model, model_data, [p.bind_at(3)])


def test_predicted_path_decays_raises_on_empty_repeatable(make_cz_path, make_fidelity_model_data):
    p = make_cz_path("XI")
    model, model_data = make_fidelity_model_data([p])
    non_decay = Path(
        start_fragment=p.start_fragment, repeatable_fragment=[], end_fragment=p.end_fragment
    )
    with pytest.raises(ValueError, match="unbound decay paths"):
        predicted_path_decays(model, model_data, [non_decay])


def test_predicted_path_decays_raises_on_non_fidelity_model(make_cz_path, make_fidelity_model_data):
    p = make_cz_path("XI")
    _, model_data = make_fidelity_model_data([p])

    class _NotFidelity:
        output_space = object()

    with pytest.raises(ValueError, match="fidelity model"):
        predicted_path_decays(_NotFidelity(), model_data, [p])
