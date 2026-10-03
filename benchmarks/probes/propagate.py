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

"""Price ``ModelGate.clifford_propagate`` against a properly indexed lookup-table implementation.

Differences from the naive prototype:

* Cliffords are indexed by qubit, so only the ones actually overlapping the Pauli's support are
  visited -- the current implementation tests all of them, and a 256-qubit layer gate has 98.
* The support is read straight off ``QubitSparsePauli.paulis``/``.indices`` (codes Z=1, X=2, Y=3),
  with no dense ``Pauli`` label round trip anywhere.
"""

import heapq
import sys
import time
from itertools import product
from pathlib import Path

from qiskit.quantum_info import QubitSparsePauli

sys.path.insert(0, str(Path("benchmarks").resolve()))

from qnl_bench import case_by_name  # noqa: E402
from qnl_bench.cases import build_gate_set  # noqa: E402

REPEATS = 2000
CODE_TO_LETTER = {1: "Z", 2: "X", 3: "Y"}
LETTER_TO_CODE = {"Z": 1, "X": 2, "Y": 3}


def build_tables(gate):
    """Per-Clifford lookup tables, plus a qubit -> temporal-position index.

    Each table maps a tuple of codes on the Clifford's qubits (0 for identity) to the propagated
    tuple of codes on those same qubits. Built by calling the existing implementation, so it is
    correct by construction; a real one would read the Clifford's symplectic matrix.
    """
    tables, positions = [], {}
    for position, (qubit_idxs, _) in enumerate(gate.cliffords):
        width = len(qubit_idxs)
        entries = {}
        for codes in product((0, 1, 2, 3), repeat=width):
            if not any(codes):
                continue
            letters = "".join(CODE_TO_LETTER[c] for c in codes if c)
            idxs = [q for q, c in zip(qubit_idxs, codes) if c]
            out = gate.clifford_propagate(
                QubitSparsePauli.from_sparse_label((letters, idxs), num_qubits=gate.num_qubits)
            )
            out_map = dict(zip(out.indices.tolist(), out.paulis.tolist()))
            entries[codes] = tuple(out_map.get(q, 0) for q in qubit_idxs)
        tables.append((tuple(qubit_idxs), entries))
        for qubit in qubit_idxs:
            positions.setdefault(qubit, []).append(position)
    return tables, positions


def propagate(pauli: QubitSparsePauli, tables, positions) -> tuple:
    """Propagate by visiting only the Cliffords that overlap the (evolving) support."""
    support = dict(zip(pauli.indices.tolist(), pauli.paulis.tolist()))
    queue = sorted({p for q in support for p in positions.get(q, ())})
    heapq.heapify(queue)
    seen = set(queue)
    while queue:
        position = heapq.heappop(queue)
        qubit_idxs, entries = tables[position]
        codes = tuple(support.get(q, 0) for q in qubit_idxs)
        if not any(codes):
            continue
        for qubit, code in zip(qubit_idxs, entries[codes]):
            if code:
                support[qubit] = code
            else:
                support.pop(qubit, None)
        for qubit in qubit_idxs:
            for later in positions.get(qubit, ()):
                if later > position and later not in seen:
                    seen.add(later)
                    heapq.heappush(queue, later)
    return tuple(sorted(support.items()))


def reference(pauli: QubitSparsePauli, gate) -> tuple:
    """The current implementation's answer, in the same comparable form."""
    out = gate.clifford_propagate(pauli)
    return tuple(sorted(zip(out.indices.tolist(), out.paulis.tolist())))


for case_name in ("hex32", "hex128", "hex256", "grid256"):
    case = case_by_name(case_name)
    gate_set, topology = build_gate_set(case)
    name, gate = next((n, g) for n, g in gate_set.model_gate_set.items() if n not in ("P", "M"))
    n_qubits = gate.num_qubits

    t = time.perf_counter()
    tables, positions = build_tables(gate)
    t_build = time.perf_counter() - t

    # Check agreement on every weight-1 and weight-2 Pauli of the gate's first few edges.
    checked = 0
    for pair in topology.layers[0][:8]:
        for letters in ("X", "Y", "Z", "XX", "XY", "XZ", "YZ", "ZZ", "YY", "ZX"):
            idxs = list(pair[: len(letters)])
            sample = QubitSparsePauli.from_sparse_label((letters, idxs), num_qubits=n_qubits)
            assert propagate(sample, tables, positions) == reference(sample, gate), letters
            checked += 1

    pair = topology.layers[0][0]
    sample = QubitSparsePauli.from_sparse_label(("XZ", list(pair)), num_qubits=n_qubits)

    t = time.perf_counter()
    for _ in range(REPEATS):
        gate.clifford_propagate(sample)
    t_current = (time.perf_counter() - t) / REPEATS

    t = time.perf_counter()
    for _ in range(REPEATS):
        propagate(sample, tables, positions)
    t_table = (time.perf_counter() - t) / REPEATS

    print(
        f"{case_name}: {n_qubits} qubits, gate {name} has {len(gate.cliffords)} Cliffords, "
        f"{checked} agreement checks passed"
    )
    print(f"  clifford_propagate  {t_current * 1e6:8.1f} us/call")
    print(
        f"  indexed table       {t_table * 1e6:8.1f} us/call   ({t_current / t_table:.0f}x faster)"
    )
    print(f"  table build         {t_build * 1e3:8.1f} ms, once per gate")
