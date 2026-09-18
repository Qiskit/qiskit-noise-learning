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
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliLindbladMap
from samplomatic import build
from samplomatic.transpiler import generate_boxing_pass_manager

from qiskit_noise_learning.utils import build_with_prep_noise, inject_prep_noise

_PLM = PauliLindbladMap.from_list([("XX", 0.03), ("ZI", 0.05), ("IZ", 0.02), ("YY", 0.01)])
_REF = "prep"


def _map_inputs(samplex):
    """Supply ``_PLM`` for every Pauli-Lindblad-map input the samplex registered."""
    return {
        spec.name: _PLM
        for spec in samplex.inputs().specs
        if spec.name.startswith("pauli_lindblad_maps.")
    }


def _boxed(num_layers, site, leading_h=False):
    qc = QuantumCircuit(2)
    if leading_h:
        qc.h(0)
    for _ in range(num_layers):
        qc.cz(0, 1)
    kwargs = {"decomposition": "rzsx"}
    if site is not None:
        kwargs.update(
            inject_noise_targets="all",
            inject_noise_strategy="uniform_modification",
            inject_noise_site=site,
        )
    return generate_boxing_pass_manager(**kwargs).run(qc)


def test_matches_native_before_without_leading_gate():
    """With nothing absorbed ahead of the first dressing, front surgery == native "before"."""
    native = build(_boxed(1, "before"))[1]
    surgery = build(_boxed(1, None))[1]
    inject_prep_noise(surgery, [0, 1], _REF)

    native_out = native.sample(_map_inputs(native), num_randomizations=4000, rng=99)
    surgery_out = surgery.sample(_map_inputs(surgery), num_randomizations=4000, rng=99)

    assert np.array_equal(
        np.asarray(native_out["parameter_values"]), np.asarray(surgery_out["parameter_values"])
    )
    assert np.array_equal(
        np.asarray(native_out["pauli_signs"]), np.asarray(surgery_out["pauli_signs"])
    )


def test_precedes_leading_single_qubit_gate():
    """A leading gate absorbed into the first dressing must sit *after* the injected noise, so the
    front surgery must differ from the native "before" site (which places noise after it)."""
    native = build(_boxed(1, "before", leading_h=True))[1]
    surgery = build(_boxed(1, None, leading_h=True))[1]
    inject_prep_noise(surgery, [0, 1], _REF)

    native_out = native.sample(_map_inputs(native), num_randomizations=4000, rng=7)
    surgery_out = surgery.sample(_map_inputs(surgery), num_randomizations=4000, rng=7)

    assert not np.array_equal(
        np.asarray(native_out["parameter_values"]), np.asarray(surgery_out["parameter_values"])
    )


def test_registers_input_and_sign_output():
    surgery = build(_boxed(2, None))[1]
    inject_prep_noise(surgery, [0, 1], _REF)

    out = surgery.sample(_map_inputs(surgery), num_randomizations=1000, rng=1)
    assert np.asarray(out["parameter_values"]).shape[0] == 1000
    assert np.asarray(out["pauli_signs"]).shape == (1000, 1)


def test_grows_existing_pauli_signs_column():
    """A second injection appends a column rather than colliding with the first."""
    surgery = build(_boxed(2, None))[1]
    inject_prep_noise(surgery, [0, 1], "prep_a")
    inject_prep_noise(surgery, [0, 1], "prep_b")

    out = surgery.sample(
        {"pauli_lindblad_maps.prep_a": _PLM, "pauli_lindblad_maps.prep_b": _PLM},
        num_randomizations=500,
        rng=3,
    )
    assert np.asarray(out["pauli_signs"]).shape == (500, 2)


def test_build_with_prep_noise_builds_and_samples():
    template, samplex = build_with_prep_noise(_boxed(2, None), [0, 1], _REF)
    out = samplex.sample(_map_inputs(samplex), num_randomizations=200, rng=2)
    assert template.num_qubits == 2
    assert np.asarray(out["pauli_signs"]).shape == (200, 1)
