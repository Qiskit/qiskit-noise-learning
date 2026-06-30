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

"""Shared helpers for analysis tests."""

import numpy as np
import pytest

from qiskit_noise_learning.analysis import Fit
from qiskit_noise_learning.data import AveragedData, ObservableData, RawData
from qiskit_noise_learning.gate_sets import ModelGateSet
from qiskit_noise_learning.models import CompleteFidelityModel


@pytest.fixture()
def make_averaged_data():
    """Return a builder ``(entries, std_default=0.001) -> AveragedData``.

    Each entry is ``(unbound_path, depth, fidelity)`` or ``(unbound_path, depth, fidelity, std)``.
    Use ``depth=-1`` for an unbound (decay) entry.
    """

    def _make(entries, std_default=0.001):
        unbound_paths = [e[0] for e in entries]
        depths = [e[1] for e in entries]
        observables = np.array([e[2] for e in entries], dtype=float)
        std = np.array([e[3] if len(e) > 3 else std_default for e in entries], dtype=float)
        n = len(entries)
        return AveragedData.from_arrays(
            unbound_paths=unbound_paths,
            depths=depths,
            observables=observables,
            std=std,
            time_lbs=np.empty(n, dtype="datetime64[us]"),
            time_ubs=np.empty(n, dtype="datetime64[us]"),
        )

    return _make


@pytest.fixture()
def make_observable_data():
    """Return a builder for synthetic :class:`~.ObservableData` decay curves.

    ``make(entries, ...)`` takes a list of ``(unbound_path, spam_amplitude, fidelity, depths)``
    tuples and emits ``observable = spam_amplitude * fidelity**depth`` plus Gaussian noise across
    ``n_rand`` randomizations for each depth.
    """

    def _make(entries, n_rand=20, noise_std=0.005, seed=42):
        rng = np.random.default_rng(seed)
        all_observables = []
        all_unbound_paths = []
        all_depths = []
        for path, amplitude, fidelity, depths in entries:
            for depth in depths:
                true_val = amplitude * fidelity**depth
                all_observables.append(true_val + rng.normal(0, noise_std, size=n_rand))
                all_unbound_paths.append(path)
                all_depths.append(depth)
        n = len(all_observables)
        return ObservableData.from_arrays(
            unbound_paths=all_unbound_paths,
            depths=all_depths,
            observables=np.stack(all_observables),
            time_lbs=np.empty((n, n_rand), dtype="datetime64[us]"),
            time_ubs=np.empty((n, n_rand), dtype="datetime64[us]"),
        )

    return _make


@pytest.fixture()
def make_fit():
    """Return a builder ``(raw_data, coupling_map) -> Fit``.

    The fit wraps a real :class:`~.CompleteFidelityModel` over a real
    :class:`~.ModelGateSet` carrying the requested coupling map, since the
    post-select stages read only ``fit.model.gate_set.coupling_map``.
    """

    def _make(raw_data, coupling_map):
        gate_set = ModelGateSet(coupling_map.size(), coupling_map=coupling_map)
        fit = Fit(model=CompleteFidelityModel(gate_set))
        fit[RawData] = raw_data
        return fit

    return _make


@pytest.fixture()
def make_raw_data(make_instruction_sequence):
    """Return a builder ``(creg_names, measurement_map, data) -> RawData`` (1 sequence)."""

    def _make(creg_names, measurement_map, data):
        seq = make_instruction_sequence(name="p0", depth=1)
        num_rand = data.shape[0]
        num_bits = data.shape[2]
        return RawData.from_arrays(
            creg_names=creg_names,
            measurement_map=measurement_map,
            instruction_sequences=[seq],
            data=[data],
            measurement_flips=[np.zeros((num_rand, num_bits), dtype=bool)],
            time_lbs=[np.array(["2026-01-01"] * num_rand, dtype="datetime64[us]")],
            time_ubs=[np.array(["2026-01-02"] * num_rand, dtype="datetime64[us]")],
        )

    return _make
