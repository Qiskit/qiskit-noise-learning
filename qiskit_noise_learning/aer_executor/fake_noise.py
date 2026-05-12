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

"""Utilities for generating random Pauli-Lindblad noise models for testing and simulation."""

from itertools import combinations
from itertools import product as iproduct

import numpy as np
from numpy.random import Generator
from qiskit.quantum_info import PauliLindbladMap

from ..gate_sets import GateSet


def _pauli_generators(local_idxs: tuple[int, ...], n: int, *, spam: bool) -> list[str]:
    """Return non-identity Pauli strings on ``n`` qubits acting only on ``local_idxs``.

    Args:
        local_idxs: The local qubit indices to act on (0-indexed within the gate).
        n: Total number of qubits in the gate.
        spam: If ``True``, restrict to the X Pauli (SPAM noise model); otherwise use X, Y, Z.

    Returns:
        A list of dense Pauli strings (rightmost character = qubit 0).
    """
    singles = "IX" if spam else "IXYZ"
    result = []
    for combo in iproduct(singles, repeat=len(local_idxs)):
        if all(p == "I" for p in combo):
            continue
        chars = ["I"] * n
        for local_idx, p in zip(local_idxs, combo):
            chars[n - 1 - local_idx] = p
        result.append("".join(chars))
    return result


def fake_noise(
    gate_set: GateSet,
    noise_2q: float = 0.005,
    noise_1q: float = 0.001,
    noise_idle: float = 0.001,
    noise_xtalk: float = 0.001,
    noise_spam: float = 0.02,
    temperature: float = 1.0,
    k: int = 1,
    rng: int | Generator | None = None,
) -> dict[str, PauliLindbladMap]:
    """Generate random noise models based on a gate set contents and topology.

    Each noise parameter represents an expected **process infidelity** contribution from
    the corresponding noise source, in the sense that the expected process infidelity of
    the source's Pauli-Lindblad channel — evaluated on the qubits it acts on — equals the
    parameter value. The relationship used is :math:`e_F = \\frac{d}{d+1}\\sum_P r_P` where
    :math:`d = 2^m` for an :math:`m`-qubit channel.

    Args:
        gate_set: The gateset to make noise for.
        noise_2q: The expected process infidelity per two-qubit constituent gate.
        noise_1q: The expected process infidelity per single-qubit constituent gate.
        noise_idle: The expected process infidelity per idling qubit.
        noise_xtalk: The expected process infidelity per cross-talk qubit pair.
        noise_spam: The expected process infidelity per prep/meas qubit.
        temperature: Related to the variance of sampling noise around the means. Higher values
            give rates closer to the means; lower values give higher variance.
        k: What coupling distance to include ``noise_xtalk`` at, where :math:`k<=1` adds no noise
            because it's already accounted for by ``noise_2q`` and ``noise_1q``.
        rng: Randomness seed or generator.

    Returns:
        A map from gate names to noise models in the format accepted by :class:`~.AerExecutor`.
    """
    rng = np.random.default_rng(rng)

    coupling_map = None
    if k > 1:
        try:
            coupling_map = gate_set.model_gate_set.coupling_map
        except (AttributeError, NotImplementedError):
            pass

    result = {}
    for gate_name, gate in gate_set.items():
        qubit_map = {q: i for i, q in enumerate(gate.qubit_idxs)}
        n = gate.num_qubits

        # Each segment: (local_idxs, mean_total_rate, is_spam)
        segments: list[tuple[tuple[int, ...], float, bool]] = []

        for gate_idxs in gate.constituent_gate_idxs:
            local_idxs = tuple(qubit_map[q] for q in gate_idxs)
            level = noise_2q if len(local_idxs) >= 2 else noise_1q
            segments.append((local_idxs, level, False))

        for q in gate.idling_idxs:
            segments.append(((qubit_map[q],), noise_idle, False))

        for q in gate.prep_idxs | gate.meas_idxs:
            segments.append(((qubit_map[q],), noise_spam, True))

        if k > 1 and n > 1 and coupling_map is not None:
            gate_qubits = list(gate.qubit_idxs)
            dist_matrix = coupling_map.distance_matrix
            constituent_pairs = {frozenset(gate_idxs) for gate_idxs in gate.constituent_gate_idxs}
            for i, j in combinations(range(len(gate_qubits)), 2):
                if frozenset((gate_qubits[i], gate_qubits[j])) in constituent_pairs:
                    continue
                dist = dist_matrix[gate_qubits[i], gate_qubits[j]]
                if 1 < dist <= k:
                    segments.append(((i, j), noise_xtalk, False))

        if not segments:
            continue

        # Accumulate mean rates per Pauli string, deduplicating across segments.
        # Scale by (d_local + 1) / d_local so that noise parameters represent process
        # infidelity: e_F = d/(d+1) * sum(rates) for a d-dimensional system.
        mean_rates: dict[str, float] = {}
        for local_idxs, mean_total, spam in segments:
            paulis = _pauli_generators(local_idxs, n, spam=spam)
            d_local = 2 ** len(local_idxs)
            per_pauli = mean_total * (d_local + 1) / (d_local * len(paulis))
            for p in paulis:
                mean_rates[p] = mean_rates.get(p, 0.0) + per_pauli

        pauli_list = list(mean_rates.keys())
        means = np.array(list(mean_rates.values()))

        # Sample rates from Gamma(temperature, mean/temperature):
        # E[rate] = mean, Var[rate] = mean^2 / temperature.
        rates = rng.gamma(temperature, means / temperature)

        result[gate_name] = PauliLindbladMap.from_list(list(zip(pauli_list, rates.tolist())))

    return result
