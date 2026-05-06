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

"""Generators for paths and path patterns.

The returned iterators should be of a form usable by :meth:`ExperimentBuilder.add_path_patterns`
or :meth:`ExperimentBuilder.add_paths`.
"""

from collections.abc import Iterator

import numpy as np
from qiskit.quantum_info import QubitSparsePauli, QubitSparsePauliList
from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.experiment_builder.utils import generate_bases
from qiskit_noise_learning.gate_sets import ModelGate
from qiskit_noise_learning.sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionPattern,
    PartialPauliPermutation,
    Path,
    PathPattern,
)


def depth0_path_generator(
    prep_gate: ModelGate,
    meas_gate: ModelGate,
    indices_list: Iterator[Iterator[int]],
    num_qubits: int,
) -> Iterator[tuple[Path, None]]:
    """Generate depth-0 paths for the given preparation and measurement gates.

    Args:
        prep_gate: The preparation gate.
        meas_gate: The measurement gate.
        indices_list: The indices of the qubits on which to compute :math:`Z` expectation values.
        num_qubits: The total number of qubits in the QPU.

    Returns:
        An iterator over depth-0 paths.
    """
    for indices in indices_list:
        yield (
            Path(
                pattern=PathPattern(
                    start_fragment=[
                        FidelityIndex(
                            gate=prep_gate,
                            pauli=QubitSparsePauli.identity(num_qubits),
                            in_bit_indices=frozenset(),
                            out_bit_indices=frozenset(indices),
                        )
                    ],
                    repeatable_fragment=[],
                    end_fragment=[
                        FidelityIndex(
                            gate=meas_gate,
                            pauli=QubitSparsePauli.identity(num_qubits),
                            in_bit_indices=frozenset(indices),
                            out_bit_indices=frozenset(),
                        )
                    ],
                ),
                depth=0,
            ),
            None,
        )


def depth1_path_generator(
    prep_gate: ModelGate,
    meas_gate: ModelGate,
    gate: ModelGate,
    input_paulis: QubitSparsePauliList,
) -> Iterator[tuple[Path, None]]:
    """Generate depth-1 paths for the given gate.

    Args:
        prep_gate: The preparation gate.
        meas_gate: The measurement gate.
        gate: The gate of interest.
        input_paulis: The Paulis to be fed into the single application of the gate.

    Returns:
        An iterator over depth-1 paths.
    """
    ident = QubitSparsePauli.identity(input_paulis.num_qubits)

    for pauli in input_paulis:
        out_pauli = gate.clifford_propagate(pauli)
        yield (
            Path(
                pattern=PathPattern(
                    start_fragment=[
                        FidelityIndex(
                            prep_gate, pauli=ident, out_bit_indices=frozenset(pauli.indices)
                        ),
                        FidelityIndex(gate, pauli=out_pauli),
                    ],
                    repeatable_fragment=[],
                    end_fragment=[
                        FidelityIndex(
                            meas_gate, pauli=ident, in_bit_indices=frozenset(out_pauli.indices)
                        )
                    ],
                ),
                depth=0,
            ),
            None,
        )


def even_depth_pattern_generator(
    prep_gate: ModelGate,
    meas_gate: ModelGate,
    gate: ModelGate,
    input_paulis: QubitSparsePauliList,
    output_paulis: QubitSparsePauliList | None = None,
) -> Iterator[tuple[PathPattern, None]]:
    """Generator for path patterns with repetitions of two applications of the given gate.

    Iterates over all well-defined path patterns with repeatable fragments consisting of two
    applications of the gate, in which an element of ``input_paulis`` is transformed into an element
    of ``output_paulis`` (with a possible layer of single qubit Cliffords between the two gate
    applications). Such a pair ``(input_pauli, output_pauli)`` generates a "well-defined" pattern if
    ``gate.clifford_propagate(input_pauli).indices ==
    gate.clifford_propagate(output_pauli, inverse=True).indices``
    and ``input_pauli.indices == output_pauli.indices``.

    Args:
        prep_gate: The preparation gate.
        meas_gate: The measurement gate.
        gate: The gate of interest.
        input_paulis: The list of Paulis to be fed into the first application of the gate.
        output_paulis: The list of Paulis to be output by the second application of the gate. If
            ``None``, defaults to ``input_paulis``.

    Returns:
        An iterator over all well-defined path patterns for the given preparation and measurement
        paulis.
    """

    output_paulis = output_paulis or input_paulis

    ident = QubitSparsePauli.identity(num_qubits=input_paulis.num_qubits)

    for input_pauli in input_paulis:
        input_fidelity = FidelityIndex.from_transition(
            gate=gate, in_pauli=input_pauli, out_pauli=gate.clifford_propagate(input_pauli)
        )
        for output_pauli in output_paulis:
            if not np.array_equal(input_pauli.indices, output_pauli.indices):
                continue
            output_fidelity = FidelityIndex(gate, pauli=output_pauli)

            if np.array_equal(
                input_fidelity.transition[1].indices, output_fidelity.transition[0].indices
            ):
                yield (
                    PathPattern(
                        start_fragment=[
                            FidelityIndex(
                                prep_gate,
                                pauli=ident,
                                out_bit_indices=frozenset(input_fidelity.transition[0].indices),
                            )
                        ],
                        repeatable_fragment=[input_fidelity, output_fidelity],
                        end_fragment=[
                            FidelityIndex(
                                meas_gate,
                                pauli=ident,
                                in_bit_indices=frozenset(output_fidelity.transition[1].indices),
                            )
                        ],
                    ),
                    None,
                )


def even_depth_vanilla_pattern_generator(
    prep_gate: ModelGate,
    meas_gate: ModelGate,
    gate: ModelGate,
    input_paulis: QubitSparsePauliList,
) -> Iterator[tuple[PathPattern, None]]:
    """Generator for path patterns with repetitions of two applications of the given gate.

    Args:
        prep_gate: The preparation gate.
        meas_gate: The measurement gate.
        gate: The gate of interest.
        input_paulis: The list of Paulis to be fed into the first application of the gate.

    Returns:
        An iterator over path patterns of two applications of the gate.
    """

    ident = QubitSparsePauli.identity(num_qubits=input_paulis.num_qubits)

    for input_pauli in input_paulis:
        output_pauli = gate.clifford_propagate(input_pauli)

        if input_pauli != gate.clifford_propagate(output_pauli):
            continue

        input_fidelity = FidelityIndex(gate, pauli=output_pauli)
        output_fidelity = FidelityIndex(gate, pauli=input_pauli)
        yield (
            PathPattern(
                start_fragment=[
                    FidelityIndex(
                        prep_gate,
                        pauli=ident,
                        out_bit_indices=frozenset(input_fidelity.transition[0].indices),
                    )
                ],
                repeatable_fragment=[input_fidelity, output_fidelity],
                end_fragment=[
                    FidelityIndex(
                        meas_gate,
                        pauli=ident,
                        in_bit_indices=frozenset(output_fidelity.transition[1].indices),
                    )
                ],
            ),
            None,
        )


def generate_vanilla_instruction_patterns(
    prep_gate: ModelGate, meas_gate: ModelGate, gate: ModelGate, coupling_map: CouplingMap
) -> list[InstructionPattern]:
    """Return instruction patterns with repetitions of two applications of the given gate.

    This method uses :func:`~generate_bases` to construct 9 instruction patterns that are sufficient
    to measure any single- and two-qubit Pauli on a given coupling map provided it is triangle free.

    Args:
        prep_gate: The preparation gate.
        meas_gate: The measurement gate.
        gate: The gate of interest.
        coupling_map: The coupling map.

    Returns:
        A list of instruction patterns.
    """
    ret = []
    all_zs = QubitSparsePauli.from_label("Z" * len(coupling_map.graph.nodes()))
    repeatable_fragment = [ApplyGate(gate), ApplyGate(gate)]
    for basis in generate_bases(coupling_map.graph):
        basis = QubitSparsePauli.from_label(basis)
        permutation = PartialPauliPermutation.from_qubit_sparse_paulis(all_zs, basis)
        ret.append(
            InstructionPattern(
                [ApplyGate(prep_gate), permutation],
                repeatable_fragment,
                [permutation.inverse, ApplyGate(meas_gate)],
            )
        )
    return ret


def yield_matching_patterns(
    path_patterns: Iterator[PathPattern], instr_patterns: list[InstructionPattern]
) -> Iterator[tuple[PathPattern, InstructionPattern]]:
    """Yields pairs of path patterns and instruction patterns that traverse them.

    Args:
        path_patterns: An iterable of path patterns.
        instruction_patterns: A list of instruction patterns.

    Yields:
       Tuples of path pattern and instruction patterns.

    Raises:
        ValueError: If any of the path patterns is not traversed by one of instructed patters.
    """
    for path_pattern, _ in path_patterns:
        for instr_pattern in instr_patterns:
            if path_pattern.is_traversed_by(instr_pattern):
                yield (path_pattern, instr_pattern)
                break
        else:
            raise ValueError(
                "Encountered a path that is not traversed by any of the instruction sequences."
            )


def standard_vanilla_pattern_generator(
    prep_gate: ModelGate,
    meas_gate: ModelGate,
    gate: ModelGate,
    input_paulis: QubitSparsePauliList,
    coupling_map: CouplingMap,
) -> Iterator[tuple[PathPattern, InstructionPattern]]:
    """Generator for path and instruction patterns with two applications of the given gate.

    This function makes use of :func:`~generate_vanilla_instruction_patterns` which ensures that
    all single- and two-qubit Pauli fidelities can be estimated in 9 instruction patterns.

    Args:
        prep_gate: The preparation gate.
        meas_gate: The measurement gate.
        gate: The gate of interest.
        input_paulis: The list of Paulis to be fed into the first application of the gate.
        coupling_map: The coupling map.

    Returns:
        An iterator over path and instruction patterns of two applications of the gate.
    """
    yield from yield_matching_patterns(
        even_depth_vanilla_pattern_generator(prep_gate, meas_gate, gate, input_paulis),
        generate_vanilla_instruction_patterns(prep_gate, meas_gate, gate, coupling_map),
    )
