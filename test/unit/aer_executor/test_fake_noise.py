# This code is a Qiskit project.
#
# (C) Copyright IBM 2025, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit tests for fake_noise and its helper _pauli_generators."""

import numpy as np
import pytest
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, PauliLindbladMap
from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.aer_executor.fake_noise import _pauli_generators, fake_noise
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gate_set_2q():
    """2-qubit gate set: CZ + prep + meas."""
    gs = ModelGateSet(2)
    gs.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    gs.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
    gs.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
    return gs


@pytest.fixture()
def gate_set_3q_idle():
    """3-qubit gate set: CZ on (0,1) with qubit 2 idling, plus prep and meas."""
    gs = ModelGateSet(3)
    gs.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))], qubit_idxs=range(3)))
    gs.add_gate(ModelGate("P", qubit_idxs=range(3), prep_idxs=range(3)))
    gs.add_gate(ModelGate("M", qubit_idxs=range(3), meas_idxs=range(3)))
    return gs


@pytest.fixture()
def gate_set_3q_xtalk():
    """3-qubit gate set on a linear chain (0-1-2), CZ on (0,1) with qubit 2 idling.

    Qubit pair (0,2) is at coupling distance 2 — eligible for cross-talk when k=2.
    Qubit pair (1,2) is at coupling distance 1 — NOT eligible.
    Qubit pair (0,1) is a constituent gate — excluded from cross-talk.
    """
    coupling_map = CouplingMap([[0, 1], [1, 0], [1, 2], [2, 1]])
    gs = ModelGateSet(3, coupling_map=coupling_map)
    gs.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))], qubit_idxs=range(3)))
    gs.add_gate(ModelGate("P", qubit_idxs=range(3), prep_idxs=range(3)))
    gs.add_gate(ModelGate("M", qubit_idxs=range(3), meas_idxs=range(3)))
    return gs


# ---------------------------------------------------------------------------
# Tests for _pauli_generators helper
# ---------------------------------------------------------------------------


def test_pauli_generators_single_qubit_n2():
    """Single qubit at local index 0, embedded in n=2 qubit space."""
    result = _pauli_generators((0,), 2, spam=False)
    assert len(result) == 3
    assert set(result) == {"IX", "IY", "IZ"}


def test_pauli_generators_single_qubit_higher_index():
    """Single qubit at local index 1, embedded in n=2 qubit space."""
    result = _pauli_generators((1,), 2, spam=False)
    assert len(result) == 3
    assert set(result) == {"XI", "YI", "ZI"}


def test_pauli_generators_two_qubit():
    """Two qubits at local indices (0,1), n=2."""
    result = _pauli_generators((0, 1), 2, spam=False)
    assert len(result) == 15
    # All should be 2-char strings, no "II"
    assert all(len(s) == 2 for s in result)
    assert "II" not in result


def test_pauli_generators_spam_single():
    """SPAM model: only X generator for a single qubit."""
    result = _pauli_generators((0,), 2, spam=True)
    assert result == ["IX"]


def test_pauli_generators_spam_two_qubit():
    r"""SPAM model: {I,X}^2 \ {II} = 3 generators for 2 qubits."""
    result = _pauli_generators((0, 1), 2, spam=True)
    assert len(result) == 3
    assert set(result) == {"IX", "XI", "XX"}


def test_pauli_generators_no_identity():
    """All returned strings must be non-identity."""
    for n in range(1, 4):
        for spam in (True, False):
            local_idxs = tuple(range(n))
            result = _pauli_generators(local_idxs, n, spam=spam)
            assert "I" * n not in result


# ---------------------------------------------------------------------------
# Tests for fake_noise output structure
# ---------------------------------------------------------------------------


def test_output_keys(gate_set_2q):
    """Result keys match gate names."""
    result = fake_noise(gate_set_2q, rng=0)
    assert set(result) == {"CZ", "P", "M"}


def test_output_types(gate_set_2q):
    """All values are PauliLindbladMap instances."""
    result = fake_noise(gate_set_2q, rng=0)
    assert all(isinstance(v, PauliLindbladMap) for v in result.values())


def test_output_num_qubits(gate_set_2q):
    """Each map has num_qubits matching the gate's qubit count."""
    result = fake_noise(gate_set_2q, rng=0)
    for name, noise_map in result.items():
        assert noise_map.num_qubits == gate_set_2q[name].num_qubits


def test_rates_non_negative(gate_set_2q):
    """All sampled rates must be non-negative."""
    result = fake_noise(gate_set_2q, rng=42)
    for noise_map in result.values():
        assert np.all(noise_map.rates >= 0)


# ---------------------------------------------------------------------------
# Tests for generator counts
# ---------------------------------------------------------------------------


def test_generator_count_cz_gate(gate_set_2q):
    """2-qubit CZ gate: exactly 15 generators (all non-identity 2-qubit Paulis)."""
    result = fake_noise(gate_set_2q, rng=0)
    assert len(result["CZ"].rates) == 15


def test_generator_count_prep_gate(gate_set_2q):
    """Prep gate on 2 qubits (SPAM): X on each qubit → 2 generators."""
    result = fake_noise(gate_set_2q, rng=0)
    assert len(result["P"].rates) == 2


def test_generator_count_meas_gate(gate_set_2q):
    """Meas gate on 2 qubits (SPAM): X on each qubit → 2 generators."""
    result = fake_noise(gate_set_2q, rng=0)
    assert len(result["M"].rates) == 2


def test_generator_count_with_idling(gate_set_3q_idle):
    """CZ on (0,1) with qubit 2 idling: 15 (CZ) + 3 (idle) = 18 generators."""
    result = fake_noise(gate_set_3q_idle, rng=0)
    assert len(result["CZ"].rates) == 18


# ---------------------------------------------------------------------------
# Tests for rate statistics
# ---------------------------------------------------------------------------


def test_seeded_reproducibility(gate_set_2q):
    """Same seed produces identical results."""
    r1 = fake_noise(gate_set_2q, rng=7)
    r2 = fake_noise(gate_set_2q, rng=7)
    np.testing.assert_array_equal(r1["CZ"].rates, r2["CZ"].rates)


def test_different_seeds_differ(gate_set_2q):
    """Different seeds produce different rates."""
    r1 = fake_noise(gate_set_2q, rng=1)
    r2 = fake_noise(gate_set_2q, rng=2)
    assert not np.allclose(r1["CZ"].rates, r2["CZ"].rates)


def test_rng_generator_input(gate_set_2q):
    """numpy Generator object accepted as rng."""
    import numpy as np

    rng = np.random.default_rng(99)
    result = fake_noise(gate_set_2q, rng=rng)
    assert "CZ" in result


def test_high_temperature_low_variance(gate_set_2q):
    """High temperature gives rates close to means (low coefficient of variation)."""
    noise_2q = 0.005
    n_samples = 200
    cvs = []
    for seed in range(n_samples):
        r = fake_noise(gate_set_2q, noise_2q=noise_2q, temperature=1000.0, rng=seed)
        rates = r["CZ"].rates
        cvs.append(np.std(rates) / np.mean(rates))

    assert np.mean(cvs) < 0.15  # low variance: CV well below 1


def test_low_temperature_high_variance(gate_set_2q):
    """Low temperature gives rates spread more widely (high coefficient of variation)."""
    noise_2q = 0.005
    n_samples = 200
    cvs = []
    for seed in range(n_samples):
        r = fake_noise(gate_set_2q, noise_2q=noise_2q, temperature=0.01, rng=seed)
        rates = r["CZ"].rates
        cvs.append(np.std(rates) / np.mean(rates))

    assert np.mean(cvs) > 0.5  # high variance: CV well above 0


# ---------------------------------------------------------------------------
# Tests for cross-talk
# ---------------------------------------------------------------------------


def test_crosstalk_k1_no_extra_generators(gate_set_3q_xtalk):
    """k=1 (default): no cross-talk generators — same count as gate_set_3q_idle."""
    result = fake_noise(gate_set_3q_xtalk, k=1, rng=0)
    # CZ on (0,1) + qubit 2 idling → 15 + 3 = 18
    assert len(result["CZ"].rates) == 18


def test_crosstalk_k2_adds_generators(gate_set_3q_xtalk):
    """k=2: pair (0,2) at distance 2 adds cross-talk generators.

    The xtalk (0,2) segment contributes 15 generators, of which 3 overlap with
    CZ generators (IIX/Y/Z) and 3 overlap with idle generators (XII/YII/ZII),
    leaving 9 net new generators: 18 + 9 = 27 total.
    """
    result_k1 = fake_noise(gate_set_3q_xtalk, k=1, rng=0)
    result_k2 = fake_noise(gate_set_3q_xtalk, k=2, rng=0)
    assert len(result_k1["CZ"].rates) == 18
    assert len(result_k2["CZ"].rates) == 27


def test_crosstalk_excludes_constituent_pair():
    """Constituent pairs at coupling distance > 1 are excluded from cross-talk.

    Creates a CZ gate on non-adjacent qubits (0, 2) on a linear chain 0-1-2.
    The pair (0, 2) has coupling distance 2 but is a constituent gate, so k=2
    must NOT add extra cross-talk generators for it.
    """
    coupling_map = CouplingMap([[0, 1], [1, 0], [1, 2], [2, 1]])
    gs = ModelGateSet(3, coupling_map=coupling_map)
    # CZ on non-adjacent qubits (0, 2): distance 2 on the linear chain.
    gs.add_gate(ModelGate("CZ", [((0, 2), Clifford(CZGate()))]))
    gs.add_gate(ModelGate("P", qubit_idxs=range(3), prep_idxs=range(3)))
    gs.add_gate(ModelGate("M", qubit_idxs=range(3), meas_idxs=range(3)))

    result_k1 = fake_noise(gs, k=1, rng=0)
    result_k2 = fake_noise(gs, k=2, rng=0)
    # k=2 adds no extra generators: the only pair at distance 2 is (0,2),
    # which is excluded as a constituent pair.
    assert len(result_k1["CZ"].rates) == 15
    assert len(result_k2["CZ"].rates) == 15


def test_crosstalk_no_coupling_map_silently_ignored():
    """A gate set without a coupling map silently skips cross-talk even for k > 1."""
    # ModelGateSet with full coupling map: distance(0,1)=1, so k=2 still needs
    # distance > 1 pairs. Use a 2-qubit set — only one pair, at dist 1 → no xtalk.
    gs = ModelGateSet(2)
    gs.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    gs.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
    gs.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
    result_k1 = fake_noise(gs, k=1, rng=0)
    result_k2 = fake_noise(gs, k=2, rng=0)
    # No qubit pairs at distance > 1 → same generator count
    assert len(result_k2["CZ"].rates) == len(result_k1["CZ"].rates)


# ---------------------------------------------------------------------------
# Tests for process infidelity interpretation
# ---------------------------------------------------------------------------


def test_mean_sum_rates_reflects_process_infidelity(gate_set_2q):
    r"""E[sum(rates)] = (d+1)/d * noise_param per segment, so that
    e_F = d/(d+1) * sum(rates) = noise_param in expectation.
    """
    noise_2q = 0.01
    noise_spam = 0.02
    n_samples = 500

    sums_cz, sums_p = [], []
    for seed in range(n_samples):
        r = fake_noise(gate_set_2q, noise_2q=noise_2q, noise_spam=noise_spam, rng=seed)
        sums_cz.append(float(np.sum(r["CZ"].rates)))
        sums_p.append(float(np.sum(r["P"].rates)))

    # CZ: one 2-qubit segment, d=4 → E[sum] = (4+1)/4 * noise_2q = 5/4 * noise_2q
    np.testing.assert_allclose(np.mean(sums_cz), 5 / 4 * noise_2q, rtol=0.1)

    # P: two 1-qubit SPAM segments, d=2 each → E[sum] = 2 * (2+1)/2 * noise_spam = 3 * noise_spam
    np.testing.assert_allclose(np.mean(sums_p), 3 * noise_spam, rtol=0.1)


def test_process_infidelity_equals_noise_param(gate_set_2q):
    r"""The expected process infidelity of each segment equals the noise parameter.

    For a Pauli-Lindblad channel, e_F = d/(d+1) * sum(rates).  With the (d+1)/d
    rate scaling, this expectation value equals the corresponding noise parameter.
    """
    noise_2q = 0.01
    noise_spam = 0.02
    n_samples = 500

    ef_cz, ef_p = [], []
    for seed in range(n_samples):
        r = fake_noise(gate_set_2q, noise_2q=noise_2q, noise_spam=noise_spam, rng=seed)
        # CZ: 2-qubit segment → d=4
        ef_cz.append(4 / 5 * float(np.sum(r["CZ"].rates)))
        # P: 2 SPAM qubits, each 1-qubit → d=2; combined e_F = 2 * 2/3 * (sum/2)
        #    = 2/3 * sum(rates) for 2 independent 1-qubit channels
        ef_p.append(2 / 3 * float(np.sum(r["P"].rates)))

    np.testing.assert_allclose(np.mean(ef_cz), noise_2q, rtol=0.1)
    np.testing.assert_allclose(np.mean(ef_p), 2 * noise_spam, rtol=0.1)
