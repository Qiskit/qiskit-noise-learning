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
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CZGate, XGate
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.analysis import ComputeObservables, Fit
from qiskit_noise_learning.analysis.compute_observables import compute_expectation_value
from qiskit_noise_learning.data import ObservableData, RawData
from qiskit_noise_learning.experiment_builder.experiment_builder import ExperimentBuilder
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import (
    FidelityIndex,
    PartialPauliPermutation,
    Path,
)


@pytest.fixture()
def gate_set_1q():
    model_gate_set = ModelGateSet(1)
    ident = Clifford(QuantumCircuit(1))
    model_gate_set.add_gate(ModelGate("P", [((0,), ident)], prep_idxs=range(1)))
    model_gate_set.add_gate(ModelGate("M", [((0,), ident)], meas_idxs=range(1)))
    model_gate_set.add_gate(
        ModelGate("L0", [((0,), Clifford([[True, True, True], [True, False, True]]))])
    )
    model_gate_set.add_gate(ModelGate("L1", [((0,), Clifford(XGate()))]))
    return model_gate_set


@pytest.fixture()
def gate_set_cz():
    model_gate_set = ModelGateSet(2)
    model_gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    model_gate_set.add_gate(
        ModelGate("P", [((0, 1), Clifford(QuantumCircuit(2)))], prep_idxs=range(2))
    )
    model_gate_set.add_gate(
        ModelGate("M", [((0, 1), Clifford(QuantumCircuit(2)))], meas_idxs=range(2))
    )
    return model_gate_set


@pytest.fixture()
def unbound_path_ix(gate_set_cz):
    return Path(
        start_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["P"],
                in_pauli=QubitSparsePauli("II"),
                out_pauli=QubitSparsePauli("IZ"),
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("IX"),
                out_pauli=QubitSparsePauli("ZX"),
            ),
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("ZX"),
                out_pauli=QubitSparsePauli("IX"),
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["M"],
                in_pauli=QubitSparsePauli("IZ"),
                out_pauli=QubitSparsePauli("II"),
            )
        ],
    )


@pytest.fixture()
def unbound_path_xi(gate_set_cz):
    return Path(
        start_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["P"],
                in_pauli=QubitSparsePauli("II"),
                out_pauli=QubitSparsePauli("ZI"),
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("XI"),
                out_pauli=QubitSparsePauli("XZ"),
            ),
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("XZ"),
                out_pauli=QubitSparsePauli("XI"),
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["M"],
                in_pauli=QubitSparsePauli("ZI"),
                out_pauli=QubitSparsePauli("II"),
            )
        ],
    )


class TestEv:
    """Tests for the compute_expectation_value helper function."""

    def test_all_zeros(self):
        """All-zero bits and flips gives ev=1."""
        bits = np.zeros((2, 10, 2), dtype=bool)
        flips = np.zeros((2, 2), dtype=bool)
        shot_mask = np.zeros((2, 10), dtype=bool)
        bit_mask = np.array([True, True])
        signs = np.ones((2,), dtype=int)
        ev = compute_expectation_value(bits, flips, shot_mask, bit_mask, signs)
        assert np.array_equal(ev, np.ones((2,), dtype=np.float64))

    def test_all_ones(self):
        """All-one bits (no flips) with full mask gives ev=1 for even qubits."""
        bits = np.ones((3, 10, 2), dtype=bool)
        flips = np.zeros((3, 2), dtype=bool)
        shot_mask = np.zeros((3, 10), dtype=bool)
        bit_mask = np.array([True, True])
        signs = np.ones((3,), dtype=int)
        ev = compute_expectation_value(bits, flips, shot_mask, bit_mask, signs)
        assert np.array_equal(ev, np.ones((3,), dtype=np.float64))

    def test_all_ones_single_qubit(self):
        """All-one bits with 1 qubit gives ev=-1."""
        bits = np.ones((1, 10, 1), dtype=bool)
        flips = np.zeros((1, 1), dtype=bool)
        shot_mask = np.zeros((1, 10), dtype=bool)
        bit_mask = np.array([True])
        signs = np.ones((1,), dtype=int)
        ev = compute_expectation_value(bits, flips, shot_mask, bit_mask, signs)
        assert np.array_equal(ev, -np.ones((1,), dtype=np.float64))

    def test_flips_cancel_bits(self):
        """When flips match bits, XOR cancels and ev=1."""
        bits = np.ones((1, 10, 1), dtype=bool)
        flips = np.ones((1, 1), dtype=bool)
        shot_mask = np.zeros((1, 10), dtype=bool)
        bit_mask = np.array([True])
        signs = np.ones((1,), dtype=int)
        ev = compute_expectation_value(bits, flips, shot_mask, bit_mask, signs)
        assert np.array_equal(ev, np.ones((1,), dtype=np.float64))

    def test_mask_zeros_out_qubits(self):
        """Masked-out qubits don't affect the result."""
        bits = np.ones((5, 10, 2), dtype=bool)
        flips = np.zeros((5, 2), dtype=bool)
        shot_mask = np.zeros((5, 10), dtype=bool)
        bit_mask = np.array([False, True])
        signs = np.ones((5,), dtype=int)
        ev = compute_expectation_value(bits, flips, shot_mask, bit_mask, signs)
        assert np.array_equal(ev, -np.ones((5,), dtype=np.float64))

    def test_half_shots(self):
        """Half True/half False gives ev=0."""
        bits = np.zeros((1, 10, 1), dtype=bool)
        bits[:, 5:] = True
        flips = np.zeros((1, 1), dtype=bool)
        shot_mask = np.zeros((1, 10), dtype=bool)
        bit_mask = np.array([True])
        signs = np.ones((1,), dtype=int)
        ev = compute_expectation_value(bits, flips, shot_mask, bit_mask, signs)
        assert ev.mean() == pytest.approx(0.0)

    def test_sign_negative(self):
        """Negative sign flips the expectation value."""
        bits = np.zeros((1, 10, 1), dtype=bool)
        flips = np.zeros((1, 1), dtype=bool)
        shot_mask = np.zeros((1, 10), dtype=bool)
        bit_mask = np.array([True])
        signs = -np.ones((1,), dtype=int)
        ev = compute_expectation_value(bits, flips, shot_mask, bit_mask, signs)
        assert np.array_equal(ev, -np.ones((1,), dtype=np.float64))

    def test_multiple_bits(self):
        """A case where multiple bits in the creg contribute."""
        bits = np.array([[[False, True, False], [True, True, True]]])
        flips = np.zeros((1, 3), dtype=bool)
        shot_mask = np.zeros((1, 2), dtype=bool)
        bit_mask = np.array([True, False, True])
        signs = np.ones((1,), dtype=int)
        ev = compute_expectation_value(bits, flips, shot_mask, bit_mask, signs)
        assert np.array_equal(ev, np.array([1.0]))


def _run_compute_observables(paths, instruction_sequences, data, measurement_flips):
    """Helper: build a Fit with the given paths and RawData, run ComputeObservables."""
    num_bits = data[0].shape[-1] if data else 0
    raw_data = RawData.from_arrays(
        creg_names=["meas0"],
        measurement_map={"meas0": np.arange(num_bits)},
        instruction_sequences=instruction_sequences,
        data=data,
        measurement_flips=measurement_flips,
        time_lbs=[np.empty(len(x), dtype="datetime64[us]") for x in data],
        time_ubs=[np.empty(len(x), dtype="datetime64[us]") for x in data],
    )
    fit = Fit(paths=paths)
    fit[RawData] = raw_data
    return ComputeObservables().run(fit)


class TestComputeObservables:
    def test_unbound_path_observables_basic_1q(self, gate_set_1q):
        """Basic 1-qubit test verifying ev computation and sign correction across depths."""
        unbound_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("Z"), QubitSparsePauli("Z")
                )
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
                )
            ],
        )

        eb = ExperimentBuilder(gate_set_1q)
        eb.add_paths([(unbound_path, None)])
        eb.complete()

        ip = eb.instruction_sequences[0]
        se_flip, r_flip = unbound_path.fragment_sign_flips(ip)

        for depth in [1, 2, 3]:
            sign = (-1) ** (se_flip + depth * r_flip)
            inst_seqs = eb.generate_instruction_sequences([depth])
            path = unbound_path.bind_at(depth)
            # All-zero data -> compute_expectation_value = 1.0
            result = _run_compute_observables(
                paths=[path],
                instruction_sequences=inst_seqs,
                data=[np.zeros((1, 10, 1), dtype=bool)],
                measurement_flips=[np.zeros((1, 1), dtype=bool)],
            )

            obs = result.observable_data
            assert isinstance(obs, ObservableData)
            assert obs.dataset.sizes["observable"] == 1
            assert path.without_depth() in obs.dataset["unbound_path"]
            np.testing.assert_allclose(obs.dataset["observables"][0], sign * 1.0)

            # All-one data -> compute_expectation_value = -1.0
            result = _run_compute_observables(
                paths=[path],
                instruction_sequences=inst_seqs,
                data=[np.ones((1, 10, 1), dtype=bool)],
                measurement_flips=[np.zeros((1, 1), dtype=bool)],
            )
            obs = result.observable_data
            assert isinstance(obs, ObservableData)
            assert obs.dataset.sizes["observable"] == 1
            assert path.without_depth() in obs.dataset["unbound_path"]
            np.testing.assert_allclose(obs.dataset["observables"][0], sign * -1.0)

    def test_observables_basic_1q(self, gate_set_1q):
        """Basic 1-qubit test verifying ev computation for fixed depths."""
        unbound_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("Z"), QubitSparsePauli("Z")
                )
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
                )
            ],
        )

        eb = ExperimentBuilder(gate_set_1q)
        eb.add_paths([(unbound_path, None)])
        eb.complete()

        ip = unbound_path.to_instruction_sequence().complete()
        se_flip, r_flip = unbound_path.fragment_sign_flips(ip)

        depths = [1, 2]
        inst_seqs = eb.generate_instruction_sequences(depths)
        paths = [unbound_path.bind_at(d) for d in depths]
        signs = [(-1) ** (se_flip + d * r_flip) for d in depths]

        # All-zero data -> compute_expectation_value = 1.0
        result = _run_compute_observables(
            paths=paths,
            instruction_sequences=inst_seqs,
            data=[np.zeros((1, 10, 1), dtype=bool)] * 2,
            measurement_flips=[np.zeros((1, 1), dtype=bool)] * 2,
        )

        obs = result.observable_data
        assert isinstance(obs, ObservableData)
        ds = obs.dataset
        for path, expected_sign in zip(paths, signs):
            idx = int(
                np.argwhere(
                    (ds["unbound_path"].values == path.without_depth())
                    & (ds["depth"].values == path.depth)
                )[0, 0]
            )
            np.testing.assert_allclose(ds["observables"][idx], expected_sign * 1.0)

    def test_unbound_and_bound_observables_basic_1q(self, gate_set_1q):
        """Test with both variable-depth and fixed-depth paths."""
        unbound_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("Z"), QubitSparsePauli("Z")
                )
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
                )
            ],
        )

        eb = ExperimentBuilder(gate_set_1q)
        eb.add_paths([(unbound_path, None)])
        fixed_paths = [unbound_path.bind_at(x) for x in [1, 2]]
        eb.add_paths([(x, None) for x in fixed_paths], rank_reduce=False)
        eb.complete()

        ip = unbound_path.to_instruction_sequence().complete()
        se_flip, r_flip = unbound_path.fragment_sign_flips(ip)

        depths = [3, 4, 5, 8]
        inst_seqs = eb.generate_instruction_sequences(depths)

        # Variable-depth paths + fixed-depth paths
        variable_paths = [unbound_path.bind_at(d) for d in depths]
        all_paths = variable_paths + fixed_paths

        # 4 variable-depth sequences + 2 fixed-depth sequences = 6 total
        result = _run_compute_observables(
            paths=all_paths,
            instruction_sequences=inst_seqs,
            data=[np.zeros((1, 10, 1), dtype=bool)] * 6,
            measurement_flips=[np.zeros((1, 1), dtype=bool)] * 6,
        )

        obs = result.observable_data
        assert isinstance(obs, ObservableData)
        ds = obs.dataset

        # Check that all variable-depth paths are present
        for d in depths:
            sign = (-1) ** (se_flip + d * r_flip)
            idx = int(
                np.argwhere(
                    (ds["unbound_path"].values == unbound_path) & (ds["depth"].values == d)
                )[0, 0]
            )
            np.testing.assert_allclose(ds["observables"][idx], sign * 1.0)

        # Fixed-depth paths should also be matched (depths 1, 2 overlap)
        for fp in fixed_paths:
            assert np.any(
                (ds["unbound_path"].values == fp.without_depth()) & (ds["depth"].values == fp.depth)
            )

    def test_sign_alternates_with_depth(self, gate_set_1q):
        """Verify the sign alternates with depth when r_flip is True."""
        unbound_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("Z"), QubitSparsePauli("Z")
                )
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
                )
            ],
        )

        eb = ExperimentBuilder(gate_set_1q)
        eb.add_paths([(unbound_path, None)])
        eb.complete()

        ip = eb.instruction_sequences[0]
        _, r_flip = unbound_path.fragment_sign_flips(ip)
        assert r_flip is True

        inst_seqs = eb.generate_instruction_sequences([1, 2])
        paths = [unbound_path.bind_at(d) for d in [1, 2]]

        result = _run_compute_observables(
            paths=paths,
            instruction_sequences=inst_seqs,
            data=[np.zeros((1, 10, 1), dtype=bool)] * 2,
            measurement_flips=[np.zeros((1, 1), dtype=bool)] * 2,
        )

        obs = result.observable_data
        ds = obs.dataset
        idx1 = int(
            np.argwhere((ds["unbound_path"].values == unbound_path) & (ds["depth"].values == 1))[
                0, 0
            ]
        )
        idx2 = int(
            np.argwhere((ds["unbound_path"].values == unbound_path) & (ds["depth"].values == 2))[
                0, 0
            ]
        )
        ev1 = ds["observables"][idx1].values
        ev2 = ds["observables"][idx2].values
        assert ev1 == pytest.approx(-ev2)

    def test_unbound_path_observables_mask_2q(self, gate_set_cz, unbound_path_ix):
        """Verify the observable mask selects only the correct qubits."""
        assert unbound_path_ix.end_fragment[-1].observable_indices == [0]

        eb = ExperimentBuilder(gate_set_cz)
        eb.add_paths([(unbound_path_ix, None)])
        eb.complete()

        ip = eb.instruction_sequences[0]
        se_flip, r_flip = unbound_path_ix.fragment_sign_flips(ip)
        sign = (-1) ** (se_flip + 1 * r_flip)

        inst_seqs = eb.generate_instruction_sequences([1])
        path = unbound_path_ix.bind_at(1)

        # Data: qubit 0 = False (zero), qubit 1 = True (one)
        # Mask = [True, False] -> only qubit 0 observed -> ev = 1
        result = _run_compute_observables(
            paths=[path],
            instruction_sequences=inst_seqs,
            data=[np.tile([False, True], (1, 10, 1))],
            measurement_flips=[np.zeros((1, 2), dtype=bool)],
        )
        np.testing.assert_allclose(result.observable_data.dataset["observables"][0], sign * 1.0)

        # Data: qubit 0 = True (one), qubit 1 = False (zero)
        # Qubit 0: True -> ev = -1
        result = _run_compute_observables(
            paths=[path],
            instruction_sequences=inst_seqs,
            data=[np.tile([True, False], (1, 10, 1))],
            measurement_flips=[np.zeros((1, 2), dtype=bool)],
        )
        np.testing.assert_allclose(result.observable_data.dataset["observables"][0], sign * (-1.0))

    def test_unbound_path_observables_multiway(self, gate_set_cz, unbound_path_ix, unbound_path_xi):
        """Verify computation with multiple unbound paths per instruction sequence."""
        eb = ExperimentBuilder(gate_set_cz)
        eb.add_paths([(unbound_path_ix, None), (unbound_path_xi, None)])
        eb.merge_instruction_sequences()
        eb.complete()

        assert len(eb.instruction_sequences) == 1
        ip = eb.instruction_sequences[0]

        inst_seqs = eb.generate_instruction_sequences([1])
        paths = [
            unbound_path_ix.bind_at(1),
            unbound_path_xi.bind_at(1),
        ]

        # All-zero data -> ev = 1.0 for both paths
        result = _run_compute_observables(
            paths=paths,
            instruction_sequences=inst_seqs,
            data=[np.zeros((1, 10, 2), dtype=bool)],
            measurement_flips=[np.zeros((1, 2), dtype=bool)],
        )

        obs = result.observable_data
        ds = obs.dataset
        assert ds.sizes["observable"] == 2

        for path in paths:
            se_flip, r_flip = path.without_depth().fragment_sign_flips(ip)
            expected_sign = (-1) ** (se_flip + 1 * r_flip)
            idx = int(
                np.argwhere(
                    (ds["unbound_path"].values == path.without_depth())
                    & (ds["depth"].values == path.depth)
                )[0, 0]
            )
            np.testing.assert_allclose(ds["observables"][idx], expected_sign * 1.0)

    def test_unbound_path_observables_multiple_depths(self, gate_set_1q):
        """Verify computation handles multiple depths correctly."""
        unbound_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("Z"), QubitSparsePauli("Z")
                )
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
                )
            ],
        )

        eb = ExperimentBuilder(gate_set_1q)
        eb.add_paths([(unbound_path, None)])
        eb.complete()

        ip = eb.instruction_sequences[0]
        se_flip, r_flip = unbound_path.fragment_sign_flips(ip)

        inst_seqs = eb.generate_instruction_sequences([1, 2])
        paths = [unbound_path.bind_at(d) for d in [1, 2]]

        result = _run_compute_observables(
            paths=paths,
            instruction_sequences=inst_seqs,
            data=[np.zeros((2, 10, 1), dtype=bool)] * 2,
            measurement_flips=[np.zeros((2, 1), dtype=bool)] * 2,
        )

        obs = result.observable_data
        ds = obs.dataset
        assert ds.sizes["observable"] == 2

        sign1 = (-1) ** (se_flip + 1 * r_flip)
        sign2 = (-1) ** (se_flip + 2 * r_flip)

        idx1 = int(
            np.argwhere((ds["unbound_path"].values == unbound_path) & (ds["depth"].values == 1))[
                0, 0
            ]
        )
        idx2 = int(
            np.argwhere((ds["unbound_path"].values == unbound_path) & (ds["depth"].values == 2))[
                0, 0
            ]
        )
        np.testing.assert_allclose(ds["observables"][idx1], sign1 * 1.0)
        np.testing.assert_allclose(ds["observables"][idx2], sign2 * 1.0)

    def test_path_to_multiple_sequences(self, gate_set_1q):
        """Test computation of an observable for a single path measured by two different instruction
        sequences with different sign flips (so there are different sign corrections across raw data
        for a single observable).
        """
        # Clifford maps X -> -Y, Y -> Z, Z -> -X
        unbound_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_1q["P"],
                    in_pauli=QubitSparsePauli("I"),
                    out_pauli=QubitSparsePauli("Z"),
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_1q["L0"],
                    in_pauli=QubitSparsePauli("X"),
                    out_pauli=QubitSparsePauli("Y"),
                ),
                FidelityIndex.from_transition(
                    gate=gate_set_1q["L0"],
                    in_pauli=QubitSparsePauli("Z"),
                    out_pauli=QubitSparsePauli("X"),
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_1q["M"],
                    in_pauli=QubitSparsePauli("Z"),
                    out_pauli=QubitSparsePauli("I"),
                )
            ],
        )
        unbound_inst_seq0 = unbound_path.to_instruction_sequence().complete()

        # copy it again, but we want different signs
        unbound_inst_seq1 = unbound_path.to_instruction_sequence().complete()
        # the one in the standard construction is Y -> -Z
        unbound_inst_seq1.repeatable_fragment[1] = PartialPauliPermutation([2]).complete()

        # compute sign flips and validate convention that they are different
        sign_flips0 = unbound_path.fragment_sign_flips(unbound_inst_seq0)
        sign_flips1 = unbound_path.fragment_sign_flips(unbound_inst_seq1)
        assert sign_flips0 == (False, True)
        assert sign_flips1 == (False, False)

        result = _run_compute_observables(
            paths=[unbound_path.bind_at(1)],
            instruction_sequences=[
                unbound_inst_seq0.bind_at(1),
                unbound_inst_seq1.bind_at(1),
            ],
            data=[np.array([[[True]]])] * 2,
            measurement_flips=[np.zeros((1, 1), dtype=bool)] * 2,
        )

        obs = result.observable_data
        ds = obs.dataset

        # only one observable
        assert ds.sizes["observable"] == 1

        # value without sign flips is -1 for both, but unbound_inst_seq0 requires a sign flip
        np.testing.assert_allclose(ds["observables"], np.array([[1.0, -1.0]]))
