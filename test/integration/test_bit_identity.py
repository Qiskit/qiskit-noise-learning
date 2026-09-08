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

"""Tests that each bit of a :class:`~.RawData` leaf carries its own identity.

The ``"bit"`` dimension is described by the ``creg_name``, ``qubit_idx`` and ``measuring_gate_idx``
coordinates, so an analysis stage reads a bit's meaning off the bit itself rather than from where it
sits along the dimension. These tests permute a leaf's bit axis and check the stages are unmoved.

Every test asserts the *absolute* expected result rather than only agreement between two layouts,
since two layouts read wrongly in the same way would agree with each other.
"""

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

# The bit axis holds two 2-bit registers, so a bit's position alone is ambiguous. The permutations
# below are applied with ``isel``, which carries the bit coordinates and permutes ``data`` and
# ``measurement_flips`` with them.
CONTIGUOUS = [0, 1, 2, 3]
INTERLEAVED = [0, 2, 1, 3]
REVERSED_REGISTERS = [2, 3, 0, 1]


def _permute_bits(raw_data, perm):
    """Return a copy of a single-leaf ``RawData`` whose ``"bit"`` dimension is permuted."""
    dataset = raw_data.datatree["0"].dataset.isel(bit=list(perm))
    return RawData(xr.DataTree.from_dict({"0": dataset}))


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
    unbound_path = Path(
        start_fragment=[FidelityIndex.from_transition(model_gate_set["P"], ident, z0)],
        repeatable_fragment=[FidelityIndex.from_transition(model_gate_set["L0"], z0, z0)],
        end_fragment=[FidelityIndex.from_transition(model_gate_set["M"], z0, ident)],
    )
    assert unbound_path.end_fragment[-1].observable_idxs == [0]
    return unbound_path


def _observable_raw_data(unbound_path, fragment_depths, data):
    """Build a single-leaf ``RawData`` over one measuring register and one post-select register.

    The post-select register measures the same qubits but belongs to no measuring gate, so
    ``ComputeObservables`` must leave its bits out of the observable however they are laid out.
    """
    unbound_seq = unbound_path.to_instruction_sequence().complete()
    return RawData.from_arrays(
        registers=[
            MeasurementRegister("meas0", (0, 1), measuring_gate_idx=0),
            MeasurementRegister("flag_ps", (0, 1), measuring_gate_idx=-1),
        ],
        instruction_sequences=[unbound_seq.bind_at(d) for d in fragment_depths],
        data=data,
        measurement_flips=[np.zeros((1, 4), dtype=bool) for _ in fragment_depths],
        time_lbs=[np.empty(1, dtype="datetime64[us]") for _ in fragment_depths],
        time_ubs=[np.empty(1, dtype="datetime64[us]") for _ in fragment_depths],
    )


def _observables_by_depth(raw_data, unbound_path, fragment_depths):
    """Run ``ComputeObservables`` and return ``{fragment_depth: observable_value}``."""
    fit = Fit(paths=[unbound_path.bind_at(d) for d in fragment_depths])
    fit[RawData] = raw_data
    dataset = ComputeObservables().run(fit).observable_data.dataset
    return {
        int(fragment_depth): float(value)
        for fragment_depth, value in zip(
            dataset["fragment_depth"].values, dataset["observable_values"].values.ravel()
        )
    }


def _post_select_raw_data():
    """Build a single-leaf ``RawData`` holding a paired ``meas0`` / ``meas0_ps`` register."""
    seq = InstructionSequence([], [ApplyGate("CZ")], [], fragment_depth=1)
    # contiguous bit axis: [meas0/q0, meas0/q1, meas0_ps/q0, meas0_ps/q1]
    # shot 0: both qubits flip between the registers -> keep
    # shot 1: qubit 0 holds False in both -> failed -> node mode masks the shot
    data = np.array([[[False, False, True, True], [False, False, False, True]]])
    return RawData.from_arrays(
        registers=[
            MeasurementRegister("meas0", (0, 1), measuring_gate_idx=0),
            MeasurementRegister("meas0_ps", (0, 1), measuring_gate_idx=-1),
        ],
        instruction_sequences=[seq],
        data=[data],
        measurement_flips=[np.zeros((1, 4), dtype=bool)],
        time_lbs=[np.empty(1, dtype="datetime64[us]")],
        time_ubs=[np.empty(1, dtype="datetime64[us]")],
    )


def _post_select_mask(raw_data):
    """Run ``FlipPostSelect`` in node mode and return the resulting ``data_mask``."""
    coupling_map = CouplingMap.from_line(2)
    gate_set = ModelGateSet(coupling_map.size(), coupling_map=coupling_map)
    fit = Fit(model=IdentityFidelityModel(gate_set))
    fit[RawData] = raw_data
    result = FlipPostSelect(mode="node").run(fit)
    return result[RawData].datatree["0"].dataset["data_mask"].values


@pytest.mark.parametrize(
    "perm",
    [CONTIGUOUS, INTERLEAVED, REVERSED_REGISTERS],
    ids=["contiguous", "interleaved", "reversed_registers"],
)
def test_compute_observables_is_invariant_to_bit_order(perm):
    """Test that the observable is read from the same qubits however the bit axis is ordered.

    ``REVERSED_REGISTERS`` puts the post-select register, which belongs to no measuring gate,
    first on the bit axis. Pairing the path's measuring gate with the register whose bits come
    first would read ``flag_ps`` instead of ``meas0`` and give ``+1``.
    """
    unbound_path = _observable_path()
    # meas0 reads 1 on qubit 0, so the observable is -1; flag_ps reads 1 on qubit 1, which is
    # neither the observed qubit nor part of a measuring register, so it must not contribute.
    data = [np.array([[[True, False, False, True]]])]
    raw_data = _permute_bits(_observable_raw_data(unbound_path, [1], data), perm)

    assert _observables_by_depth(raw_data, unbound_path, [1]) == {1: -1.0}


@pytest.mark.parametrize(
    "perm",
    [CONTIGUOUS, INTERLEAVED, REVERSED_REGISTERS],
    ids=["contiguous", "interleaved", "reversed_registers"],
)
def test_flip_post_select_is_invariant_to_bit_order(perm):
    """Test that a base/ps pair is located by creg name however the bit axis is ordered.

    Under ``REVERSED_REGISTERS`` the creg names reach the identifier as
    ``["meas0_ps", "meas0"]``, so this also covers the pairing being independent of the order the
    names are discovered in.
    """
    raw_data = _permute_bits(_post_select_raw_data(), perm)

    np.testing.assert_array_equal(_post_select_mask(raw_data), [[False, True]])


@pytest.mark.parametrize("perm", [CONTIGUOUS, INTERLEAVED], ids=["contiguous", "interleaved"])
def test_compute_observables_across_fragment_depths_in_one_leaf(perm):
    """Test that one leaf serving several fragment depths gives the right observable at each.

    This is why ``measuring_gate_idx`` counts measuring gates rather than all gates: the bit axis
    is shared across depths, so a position among all gates would not be a property of a bit.
    """
    unbound_path = _observable_path()
    fragment_depths = [1, 2, 3]
    data = [
        # qubit 0 reads 1 -> -1
        np.array([[[True, False, False, True]]]),
        # qubit 0 reads 0 -> +1
        np.array([[[False, True, True, True]]]),
        # qubit 0 and qubit 1 both read 1, but only qubit 0 is observed -> -1
        np.array([[[True, True, False, False]]]),
    ]
    raw_data = _observable_raw_data(unbound_path, fragment_depths, data)
    assert list(raw_data.datatree) == ["0"]

    observables = _observables_by_depth(
        _permute_bits(raw_data, perm), unbound_path, fragment_depths
    )

    assert observables == {1: -1.0, 2: 1.0, 3: -1.0}
