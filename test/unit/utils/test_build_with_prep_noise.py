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
from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp
from qiskit.quantum_info import PauliLindbladMap
from samplomatic import Twirl
from samplomatic.transpiler import generate_boxing_pass_manager

from qiskit_noise_learning.utils.build_with_prep_noise import build_with_prep_noise

PLM = PauliLindbladMap.from_list([("XX", 0.03), ("ZI", 0.05), ("IZ", 0.02), ("YY", 0.01)])


def map_inputs(samplex, plm=None):
    plm = plm or PLM
    return {
        spec.name: plm
        for spec in samplex.inputs().specs
        if spec.name.startswith("pauli_lindblad_maps.")
    }


def boxed(num_layers, site):
    qc = QuantumCircuit(2)
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


def test_raises():
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValueError, match="does not begin"):
        build_with_prep_noise(qc)

    qc = QuantumCircuit(1)
    with qc.box():
        qc.noop(0)
    with pytest.raises(ValueError, match="does not contain"):
        build_with_prep_noise(qc)

    qc = QuantumCircuit(1)
    with qc.box([Twirl(dressing="right")]):
        qc.noop(0)
    with pytest.raises(ValueError, match="Dressing of the initial"):
        build_with_prep_noise(qc)


def test_registers_input_and_sign_output():
    surgery = build_with_prep_noise(boxed(2, None))[1]

    out = surgery.sample(map_inputs(surgery), num_randomizations=1000, rng=1)
    assert np.asarray(out["parameter_values"]).shape[0] == 1000
    assert np.asarray(out["pauli_signs"]).shape == (1000, 1)


def test_build_with_prep_noise_builds_and_samples():
    template, samplex = build_with_prep_noise(boxed(2, None))
    out = samplex.sample(map_inputs(samplex), num_randomizations=200, rng=2)
    assert template.num_qubits == 2
    assert np.asarray(out["pauli_signs"]).shape == (200, 1)


def test_prep_noise_addressability():
    qc = QuantumCircuit(2)
    box_qc = QuantumCircuit(2)
    box_op = BoxOp(box_qc, annotations=[Twirl()])
    qc.append(box_op, qargs=[1, 0])

    with qc.box([Twirl(dressing="right")]):
        qc.noop(0, 1)

    plm = PauliLindbladMap.from_list([("XI", -100)])

    _, samplex = build_with_prep_noise(qc)
    out = samplex.sample(
        map_inputs(samplex, plm), num_randomizations=200, rng=2, keep_registers=True
    )

    signs = out.metadata["registers"]["inject_noise_0"]
    assert np.all(signs[[0]].virtual_gates == 0)
    assert np.any(signs[[1]].virtual_gates == 2)
