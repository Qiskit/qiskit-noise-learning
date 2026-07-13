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

"""Pytest Configuration"""

from itertools import chain

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, XGate
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.data import AveragedData, ModelData, ObservableData
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.models import IdentityFidelityModel
from qiskit_noise_learning.sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionSequence,
    Path,
)

# enable scipy-style doctests
pytest_plugins = "scipy_doctest"


# --------------------------------------------------------------------------------------------------
# Gate sets
# --------------------------------------------------------------------------------------------------


@pytest.fixture()
def gate_set_1q():
    """A 1-qubit gate set with prep ``P``, measurement ``M``, and unitaries ``L0``/``L1``."""
    model_gate_set = ModelGateSet(1)
    ident = Clifford(QuantumCircuit(1))
    model_gate_set.add_gate(ModelGate("P", [((0,), ident)], prep_idxs=range(1)))
    model_gate_set.add_gate(ModelGate("M", [((0,), ident)], meas_idxs=range(1)))
    # Clifford maps X -> -Y, Y -> Z, Z -> -X
    model_gate_set.add_gate(
        ModelGate("L0", [((0,), Clifford([[True, True, True], [True, False, True]]))])
    )
    model_gate_set.add_gate(ModelGate("L1", [((0,), Clifford(XGate()))]))
    return model_gate_set


@pytest.fixture()
def gate_set_cz():
    """A 2-qubit gate set with a ``CZ`` gate, pure preparation ``P``, and measurement ``M``."""
    model_gate_set = ModelGateSet(2)
    model_gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
    return model_gate_set


# --------------------------------------------------------------------------------------------------
# Factory fixtures
# --------------------------------------------------------------------------------------------------


@pytest.fixture()
def make_cz_path(gate_set_cz):
    """Return a builder for real, traversable :class:`~.Path` objects over ``gate_set_cz``.

    ``make(repeatable)`` builds the closed ``CZ`` orbit through one or more Pauli labels. A single
    label ``P`` is expanded to the orbit ``[P, CZ(P)]``; a list of labels is used as the explicit
    orbit, with each consecutive pair representing a transition. The resulting repeatable fragment
    is a closed loop, so the path is traversable at any depth.

    With ``spam=True`` (the default) the path gains a ``P`` start fragment and an ``M`` end fragment
    whose transition Paulis are ``Z`` on the support of the loop's starting Pauli. The returned path
    is unbound; bind a depth with ``.bind_at(depth)``.
    """

    def _make(repeatable, spam=True):
        cz = gate_set_cz["CZ"]

        if isinstance(repeatable, str):
            primary = QubitSparsePauli(repeatable)
            nodes = [primary, cz.clifford_propagate(pauli=primary, inverse=False)]
        else:
            nodes = [QubitSparsePauli(label) for label in repeatable]

        n = len(nodes)
        repeatable_fragment = [
            FidelityIndex.from_transition(gate=cz, in_pauli=nodes[i], out_pauli=nodes[(i + 1) % n])
            for i in range(n)
        ]

        start_fragment = []
        end_fragment = []
        if spam:
            support = sorted(int(i) for i in nodes[0].indices)
            z_pauli = QubitSparsePauli.from_sparse_label(
                ("Z" * len(support), support), num_qubits=2
            )
            start_fragment = [
                FidelityIndex.from_transition(
                    gate=gate_set_cz["P"],
                    in_pauli=QubitSparsePauli("II"),
                    out_pauli=z_pauli,
                )
            ]
            end_fragment = [
                FidelityIndex.from_transition(
                    gate=gate_set_cz["M"],
                    in_pauli=z_pauli,
                    out_pauli=QubitSparsePauli("II"),
                )
            ]
        return Path(
            start_fragment=start_fragment,
            repeatable_fragment=repeatable_fragment,
            end_fragment=end_fragment,
        )

    return _make


@pytest.fixture()
def make_instruction_sequence():
    """Return a builder ``(name="CZ", depth=1) -> InstructionSequence``.

    Builds a minimal real instruction sequence whose repeatable fragment is a single
    :class:`~.ApplyGate`. Vary ``name`` to obtain distinct sequences.
    """

    def _make(name="CZ", depth=1):
        return InstructionSequence(
            start_fragment=[],
            repeatable_fragment=[ApplyGate(name)],
            end_fragment=[],
            depth=depth,
        )

    return _make


# --------------------------------------------------------------------------------------------------
# Data factory fixtures
# --------------------------------------------------------------------------------------------------


@pytest.fixture()
def make_averaged_data():
    """Return a builder ``(entries, std_default=0.001) -> AveragedData``.

    Each entry is ``(unbound_path, depth, value)`` with an optional trailing ``std`` (float) and/or
    ``meta`` (dict), e.g. ``(path, -1, 0.8, 0.01, {"spam_fidelity": 0.95})``. Use ``depth == -1``
    for the unbound exponential-fit (base) row and ``depth >= 0`` for empirical point rows.
    """

    def _make(entries, std_default=0.001):
        unbound_paths = [e[0] for e in entries]
        depths = [e[1] for e in entries]
        observables = np.array([e[2] for e in entries], dtype=float)
        std = np.array(
            [next((x for x in e[3:] if not isinstance(x, dict)), std_default) for e in entries],
            dtype=float,
        )
        metas = [next((x for x in e[3:] if isinstance(x, dict)), None) for e in entries]
        # ``from_arrays`` uses ``metadata or ...``, so only pass an array when something is present.
        metadata = np.array(metas, dtype=object) if any(m is not None for m in metas) else None
        n = len(entries)
        return AveragedData.from_arrays(
            unbound_paths=unbound_paths,
            depths=depths,
            observables=observables,
            std=std,
            time_lbs=np.empty(n, dtype="datetime64[us]"),
            time_ubs=np.empty(n, dtype="datetime64[us]"),
            metadata=metadata,
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
def make_fidelity_model_data(gate_set_cz):
    """Return a builder ``(paths, rates=None, rate_default=0.05) -> (model, model_data)``.

    The model is a real :class:`~.IdentityFidelityModel` over ``gate_set_cz``, and the
    :class:`~.ModelData` carries a parameter for every distinct :class:`~.FidelityIndex` in the
    given paths' repeatable, start, and end fragments.
    """

    def _make(paths, rates=None, rate_default=0.05):
        rates = rates or {}
        fidelities = list(
            dict.fromkeys(
                chain.from_iterable(
                    chain(p.repeatable_fragment, p.start_fragment, p.end_fragment) for p in paths
                )
            )
        )
        values = np.array([rates.get(f, rate_default) for f in fidelities], dtype=float)
        n = len(fidelities)
        model_data = ModelData.from_arrays(
            parameter_indices=fidelities,
            parameter_values=values,
            covariance=np.zeros((n, n)),
            time_lbs=np.empty(n, dtype="datetime64[us]"),
            time_ubs=np.empty(n, dtype="datetime64[us]"),
        )
        return IdentityFidelityModel(gate_set_cz), model_data

    return _make
