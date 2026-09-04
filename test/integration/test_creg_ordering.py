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

"""Tests for creg ordering conventions across multiple classes."""

import numpy as np
from qiskit.quantum_info import QubitSparsePauli

from qiskit_noise_learning.analysis import ComputeObservables, Fit
from qiskit_noise_learning.circuit_generator import ExecutorCircuitGenerator
from qiskit_noise_learning.data import RawData
from qiskit_noise_learning.gate_sets import QiskitGateSet
from qiskit_noise_learning.sequences import FidelityIndex, Path


def test_observable_of_crossed_measurement():
    """Test that an observable computed from raw data follows the classical bit ordering that the
    generated circuit measures into, rather than ascending physical qubit order.
    """
    # an explicitly crossed measurement gate: "M" has qubit_idxs == clbit_meas_idxs == (1, 0)
    gate_set = QiskitGateSet(2, add_default_spam=False)
    gate_set.add_measurement([1, 0], name="M")
    gate_set.add_preparation(name="P")
    with gate_set.build_new_gate() as builder:
        builder.circuit.cz(0, 1)

    # a path whose observable is <Z> on physical qubit 0 alone
    model_gate_set = gate_set.model_gate_set
    ident = QubitSparsePauli.identity(2)
    z0 = QubitSparsePauli.from_label("IZ")
    unbound_path = Path(
        start_fragment=[FidelityIndex.from_transition(model_gate_set["P"], ident, z0)],
        repeatable_fragment=[FidelityIndex.from_transition(model_gate_set["L0"], z0, z0)],
        end_fragment=[FidelityIndex.from_transition(model_gate_set["M"], z0, ident)],
    )
    assert unbound_path.end_fragment[-1].observable_idxs == [0]

    fragment_depth = 2
    unbound_seq = unbound_path.to_instruction_sequence().complete()
    seq = unbound_seq.bind_at(fragment_depth)
    path = unbound_path.bind_at(fragment_depth)

    item, _, clbit_qubit_idxs = ExecutorCircuitGenerator(gate_set).generate_samplex_item(
        [seq], num_randomizations=1
    )
    creg_names = [creg.name for creg in item.circuit.cregs]
    np.testing.assert_array_equal(clbit_qubit_idxs["meas0"], [1, 0])

    # a single shot outcome of 1 on physical qubit 0 and 0 on physical qubit 1
    data = np.zeros((1, 1, 2), dtype=bool)
    data[..., list(clbit_qubit_idxs["meas0"]).index(0)] = True

    fit = Fit(paths=[path])
    fit[RawData] = RawData.from_arrays(
        creg_names=creg_names,
        clbit_qubit_idxs=clbit_qubit_idxs,
        instruction_sequences=[seq],
        data=[data],
        measurement_flips=[np.zeros((1, 2), dtype=bool)],
        time_lbs=[np.empty(1, dtype="datetime64[us]")],
        time_ubs=[np.empty(1, dtype="datetime64[us]")],
    )
    observable_values = ComputeObservables().run(fit).observable_data.dataset["observable_values"]

    # a mask built by sorting the measured qubits would read the qubit 1 bit and flip the sign
    start_end_flip, repeatable_flip = unbound_path.fragment_sign_flips(unbound_seq)
    sign = (-1) ** (start_end_flip + fragment_depth * repeatable_flip)
    np.testing.assert_allclose(observable_values[0], -sign)
