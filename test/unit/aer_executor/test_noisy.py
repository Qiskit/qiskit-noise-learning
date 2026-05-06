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

from itertools import islice
from typing import Literal

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.bit_array import BitArray
from qiskit.quantum_info import PauliLindbladMap
from qiskit_ibm_runtime.fake_provider.backends.fez import FakeFez
from qiskit_ibm_runtime.quantum_program import QuantumProgram
from samplomatic import Tag, Twirl
from samplomatic.builders.build import build

from qiskit_noise_learning.aer_executor import AerExecutor


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def circ_a():
    num_qubits = 2
    active_qubits = list(range(num_qubits))

    qc_boxed = QuantumCircuit(num_qubits, num_qubits)
    with qc_boxed.box(
        annotations=[
            Twirl(dressing="left"),
            Tag(ref="r0"),
        ]
    ):  # pyright: ignore[reportGeneralTypeIssues]
        for edge in batched(active_qubits, 2):
            if len(edge) == 2:
                qc_boxed.cz(*edge)

    with qc_boxed.box(annotations=[Twirl(dressing="right")]):
        qc_boxed.noop([0, 1])
    return qc_boxed, active_qubits


def circ_b():
    fez_backend = FakeFez()
    coupling_map = fez_backend.coupling_map
    active_qubits = list(range(fez_backend.num_qubits))

    qc_boxed = QuantumCircuit(fez_backend.num_qubits, fez_backend.num_qubits)
    with qc_boxed.box(
        annotations=[
            Twirl(dressing="left"),
            Tag(ref="r0"),
        ]
    ):  # pyright: ignore[reportGeneralTypeIssues]
        for edge in batched(active_qubits, 2):
            if edge in coupling_map:
                qc_boxed.cz(*edge)
            else:
                qc_boxed.z(edge)

    with qc_boxed.box(annotations=[Twirl(dressing="right")]):
        qc_boxed.noop(active_qubits)

    return qc_boxed, active_qubits


@pytest.mark.parametrize("case", ["a", "b"])
@pytest.mark.parametrize("noise", [True, False])
def test_small_noisy(stabilizer_simulator, noise: bool, case: Literal["a", "b"]):
    if case == "a":
        qc_boxed, active_qubits = circ_a()
    elif case == "b":
        qc_boxed, active_qubits = circ_b()
    else:
        raise ValueError("...")

    qc_boxed.measure(active_qubits, active_qubits)

    template_circuit, samplex = build(qc_boxed)

    assert template_circuit.count_ops().get("rz", 0)

    shots_per_twirl = 1024
    num_twirls = 1
    num_shots_tot = shots_per_twirl * num_twirls

    # Build a QuantumProgram using a SamplexItem
    program = QuantumProgram(shots=shots_per_twirl)
    program.append_samplex_item(
        template_circuit,
        samplex=samplex,
        shape=(num_twirls,),
    )

    def _xi(i: int, n: int = len(active_qubits)) -> str:
        ll = ["I"] * n
        ll[i] = "X"
        return "".join(reversed(ll))

    if noise:
        noise_dict = {
            "r0": PauliLindbladMap.from_list([(_xi(i), 1e-1) for i in range(len(active_qubits))]),
        }
    else:
        noise_dict = None

    # Run via AerExecutor
    executor = AerExecutor(
        stabilizer_simulator,
        noise_dict=noise_dict,
    )
    job = executor.run(program)
    result = job.result()

    assert len(result) == 1

    ba_c = BitArray.from_bool_array(result[0]["c"])
    cts = ba_c.get_counts()
    if noise:
        assert cts.get("0" * len(active_qubits), 0) < num_shots_tot
    else:
        assert cts.get("0" * len(active_qubits), 0) == num_shots_tot
