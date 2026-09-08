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

"""Tests that stages processing RawData are immune to bit permutations."""

import numpy as np
import pytest
import xarray as xr
from qiskit.quantum_info import QubitSparsePauli
from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.analysis import ComputeObservables, Fit, FlipPostSelect
from qiskit_noise_learning.data import MeasurementRegister, RawData
from qiskit_noise_learning.gate_sets import ModelGateSet, QiskitGateSet
from qiskit_noise_learning.models import IdentityFidelityModel
from qiskit_noise_learning.sequences import ApplyGate, FidelityIndex, InstructionSequence, Path

# Every leaf below holds a 2-bit measuring register followed by a 2-bit post-select register, so a
# bit's position alone is ambiguous. ``interleaved`` splits each register's bits into a
# non-contiguous run; ``reversed_registers`` puts the register belonging to no measuring gate first.
# The natural contiguous layout is left to the two stages' own unit tests.
PERMUTED_BITS = pytest.mark.parametrize(
    "perm", [[0, 2, 1, 3], [2, 3, 0, 1]], ids=["interleaved", "reversed_registers"]
)


def _raw_data(ps_name, instruction_sequences, data, perm):
    """Build a single-leaf ``RawData`` over a measuring register and a post-select register."""
    raw_data = RawData.from_arrays(
        registers=[
            MeasurementRegister("meas0", (0, 1), measuring_gate_idx=0),
            MeasurementRegister(ps_name, (0, 1), measuring_gate_idx=-1),
        ],
        instruction_sequences=instruction_sequences,
        data=data,
        measurement_flips=[np.zeros((1, 4), dtype=bool) for _ in data],
        time_lbs=[np.empty(1, dtype="datetime64[us]") for _ in data],
        time_ubs=[np.empty(1, dtype="datetime64[us]") for _ in data],
    )
    assert list(raw_data.datatree) == ["0"]
    return RawData(xr.DataTree.from_dict({"0": raw_data.datatree["0"].dataset.isel(bit=perm)}))


def _observable_path():
    """Return an unbound path over a CZ layer whose observable is ``Z`` on physical qubit 0."""
    gate_set = QiskitGateSet(2, add_default_spam=False)
    gate_set.add_measurement([0, 1], name="M")
    gate_set.add_preparation(name="P")
    with gate_set.build_new_gate() as builder:
        builder.circuit.cz(0, 1)

    model_gate_set = gate_set.model_gate_set
    ident = QubitSparsePauli.identity(2)
    z0 = QubitSparsePauli.from_label("IZ")
    return Path(
        start_fragment=[FidelityIndex.from_transition(model_gate_set["P"], ident, z0)],
        repeatable_fragment=[FidelityIndex.from_transition(model_gate_set["L0"], z0, z0)],
        end_fragment=[FidelityIndex.from_transition(model_gate_set["M"], z0, ident)],
    )


def _observables(unbound_path, fragment_depths, data, perm):
    """Run ``ComputeObservables`` over one leaf, returning ``{fragment_depth: value}``."""
    unbound_seq = unbound_path.to_instruction_sequence().complete()
    raw_data = _raw_data("flag_ps", [unbound_seq.bind_at(d) for d in fragment_depths], data, perm)
    fit = Fit(paths=[unbound_path.bind_at(d) for d in fragment_depths])
    fit[RawData] = raw_data
    dataset = ComputeObservables().run(fit).observable_data.dataset
    return {
        int(fragment_depth): float(value)
        for fragment_depth, value in zip(
            dataset["fragment_depth"].values, dataset["observable_values"].values.ravel()
        )
    }


@PERMUTED_BITS
def test_compute_observables_is_invariant_to_bit_order(perm):
    """Test that the observable is read from the same qubits however the bit axis is ordered."""
    unbound_path = _observable_path()
    # meas0 reads 1 on qubit 0, so the observable is -1; flag_ps reads 1 on qubit 1, which is
    # neither the observed qubit nor part of a measuring register, so it must not contribute.
    data = [np.array([[[True, False, False, True]]])]

    assert _observables(unbound_path, [1], data, perm) == {1: -1.0}


@PERMUTED_BITS
def test_flip_post_select_is_invariant_to_bit_order(perm):
    """Test that a base/ps pair is located by creg name however the bit axis is ordered."""
    seq = InstructionSequence([], [ApplyGate("CZ")], [], fragment_depth=1)
    # contiguous bit axis: [meas0/q0, meas0/q1, meas0_ps/q0, meas0_ps/q1]
    # shot 0: both qubits flip between the registers -> keep
    # shot 1: qubit 0 holds False in both -> failed -> node mode masks the shot
    data = [np.array([[[False, False, True, True], [False, False, False, True]]])]
    fit = Fit(model=IdentityFidelityModel(ModelGateSet(2, coupling_map=CouplingMap.from_line(2))))
    fit[RawData] = _raw_data("meas0_ps", [seq], data, perm)

    result = FlipPostSelect(mode="node").run(fit)

    mask = result[RawData].datatree["0"].dataset["data_mask"].values
    np.testing.assert_array_equal(mask, [[False, True]])


def test_compute_observables_across_fragment_depths_in_one_leaf():
    """Test that one leaf serving several fragment depths gives the right observable at each."""
    unbound_path = _observable_path()
    data = [
        # qubit 0 reads 1 -> -1
        np.array([[[True, False, False, True]]]),
        # qubit 0 reads 0 -> +1
        np.array([[[False, True, True, True]]]),
        # qubit 0 and qubit 1 both read 1, but only qubit 0 is observed -> -1
        np.array([[[True, True, False, False]]]),
    ]

    observables = _observables(unbound_path, [1, 2, 3], data, perm=[0, 1, 2, 3])

    assert observables == {1: -1.0, 2: 1.0, 3: -1.0}
