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

"""Transpiler pass that decomposes RX gates into H-RZ-H."""

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass


class DecomposeRxPass(TransformationPass):
    """Decompose every ``rx(θ)`` gate into ``h - rz(θ) - h``.

    This allows circuits containing ``rx`` to run on simulators that support
    ``rz`` and ``h`` but not ``rx`` natively (e.g. the stabilizer method).
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.op_nodes():
            if node.name != "rx":
                continue
            (theta,) = node.op.params
            qc = QuantumCircuit(1)
            qc.h(0)
            qc.rz(theta, 0)
            qc.h(0)
            dag.substitute_node_with_dag(node, circuit_to_dag(qc))
        return dag
