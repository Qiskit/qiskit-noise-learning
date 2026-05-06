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

from itertools import count

import numpy as np
from qiskit.circuit import BoxOp, CircuitInstruction, ClassicalRegister, QuantumCircuit
from qiskit_ibm_runtime.quantum_program import QuantumProgramResult
from qiskit_ibm_runtime.quantum_program.quantum_program import SamplexItem
from samplomatic import build
from samplomatic.annotations import InjectLocalClifford, Tag, Twirl

from qiskit_noise_learning.data import RawData

from ..gate_sets import QiskitGateSet
from ..sequences import ApplyGate, InstructionSequence, PartialPauliPermutation
from .circuit_generator import CircuitGenerator

TO_SAMPLOMATIC_C1 = np.array([0, 7, 9, 13, 18, 22], dtype=np.uint8)
"""An array elements of :const:`~C1_TO_TABLEAU` to corresponding value in samplomatic."""


class ExecutorDataMapper:
    """Map executor results into standard results.

    As instruction sequences with similar structure are generated together with a single template
    circuit and different samplex arguments, the order of input sequences to
    :meth:`.ExecutorCircuitGenerator.generate` is not preserved during execution. This class
    contains properties to format the results of a
    :class:`qiskit_ibm_runtime.quantum_program.QuantumProgramResult` to the order of the
    input sequences.

    Args:
        sequence_map: A map from instruction sequence indices to result indices.
        creg_names: The name of classical registers in the circuit.
        instruction_sequences: The instruction sequences associated with the data.
        num_randomizations: The number of randomizations used per experiment.

    """

    def __init__(
        self,
        sequence_map: dict[int, tuple[int, int]],
        creg_names: list[list[str]],
        instruction_sequences: list[InstructionSequence],
        num_randomizations: int,
    ):
        self._sequence_map = sequence_map
        self._creg_names = creg_names
        self._instruction_sequences = instruction_sequences
        self._num_randomizations = num_randomizations

    @property
    def creg_names(self) -> list[list[str]]:
        """List of names of the classical registers contained in the results.

        The list at a given index corresponds to names expected in the data of the
        :class:`qiskit_ibm_runtime.quantum_program.QuantumProgramResult` at the same index.
        """
        return self._creg_names

    @property
    def instruction_sequences(self) -> list:
        """The instruction sequences corresponding to the sequence indices in the sequence map."""
        return self._instruction_sequences

    @property
    def sequence_map(self) -> dict[int, tuple[int, int]]:
        """Map from sequence indices to result indices.

        The returned tuple's first entry is the index of the input sequence's result in the
        :class:`qiskit_ibm_runtime.quantum_program.QuantumProgramResult`, while the second entry is
        the index of the data contained in the result.
        """
        return self._sequence_map

    @property
    def num_randomizations(self) -> int:
        """The number of randomizations used per experiment."""
        return self._num_randomizations


class ExecutorCircuitGenerator(
    CircuitGenerator[list[SamplexItem], ExecutorDataMapper, QuantumProgramResult]
):
    """A circuit generator that converts sequences of Qiskit gates into a samplex items.

    Args:
        gate_set: The Qiskit gate set that this generator constructs against.
        num_randomizations: The number of randomizations to use.
    """

    def __init__(self, gate_set: QiskitGateSet, num_randomizations: int = 50):
        self._gate_set = gate_set
        self._num_randomizations = num_randomizations

    @property
    def gate_set(self) -> QiskitGateSet:
        return self._gate_set

    @staticmethod
    def collect(result, data_mapper):
        # extract time bounds on a program item basis

        if hasattr(result.metadata, "chunk_timing"):
            program_item_time_lbs = [
                np.array([], dtype="datetime64[us]") for _ in range(len(result))
            ]
            program_item_time_ubs = [
                np.array([], dtype="datetime64[us]") for _ in range(len(result))
            ]
            for chunk_timing in result.metadata.chunk_timing:
                chunk_start = np.array(chunk_timing.start, dtype="datetime64[us]")
                chunk_stop = np.array(chunk_timing.stop, dtype="datetime64[us]")

                for part in chunk_timing.parts:
                    program_item_time_lbs[part.idx_item] = np.append(
                        program_item_time_lbs[part.idx_item], [chunk_start] * part.size
                    )
                    program_item_time_ubs[part.idx_item] = np.append(
                        program_item_time_lbs[part.idx_item], [chunk_stop] * part.size
                    )
        else:
            the_length = len(data_mapper.instruction_sequences) * data_mapper.num_randomizations
            program_item_time_lbs = np.full(
                (len(result), the_length), "NaT", dtype="datetime64[us]"
            )
            program_item_time_ubs = np.full(
                (len(result), the_length), "NaT", dtype="datetime64[us]"
            )

        data = []
        measurement_flips = []
        creg_bit_boundaries = []
        time_lbs = []
        time_ubs = []

        for seq_idx in range(len(data_mapper.instruction_sequences)):
            item_idx, d_idx = data_mapper.sequence_map[seq_idx]
            this_item = result[item_idx]

            this_creg_bit_boundaries = dict()
            this_data = []
            these_flips = []
            bit_count = 0
            for creg in data_mapper.creg_names[item_idx]:
                new_bit_count = bit_count + this_item[creg].shape[-1]
                this_creg_bit_boundaries[creg] = (bit_count, new_bit_count)
                bit_count = new_bit_count

                this_data.append(this_item[creg][d_idx])
                flips = this_item.get(f"measurement_flips.{creg}")
                if flips is None:
                    flips = np.zeros(
                        (this_data[-1].shape[0], 1, this_data[-1].shape[-1]), dtype=np.bool_
                    )
                else:
                    flips = flips[d_idx]
                these_flips.append(flips)

            creg_bit_boundaries.append(this_creg_bit_boundaries)
            # assuming only one register
            this_data_array = this_data[0]
            for data_array in this_data[1:]:
                this_data_array = np.append(this_data_array, data_array, axis=-1)
            data.append(this_data_array)
            these_flips_array = these_flips[0]
            for flips_array in these_flips[1:]:
                these_flips_array = np.append(these_flips_array, flips_array, axis=-1)
            measurement_flips.append(
                these_flips_array.reshape((these_flips_array.shape[0], these_flips_array.shape[-1]))
            )

            # extract time bounds from relevant section of time arrays
            time_lbs.append(
                program_item_time_lbs[item_idx][
                    data_mapper.num_randomizations * d_idx : data_mapper.num_randomizations
                    * (d_idx + 1)
                ]
            )
            time_ubs.append(
                program_item_time_ubs[item_idx][
                    data_mapper.num_randomizations * d_idx : data_mapper.num_randomizations
                    * (d_idx + 1)
                ]
            )
        return RawData.from_arrays(
            instruction_sequences=list(data_mapper.instruction_sequences),
            creg_bit_boundaries=creg_bit_boundaries,
            data=data,
            measurement_flips=measurement_flips,
            time_lbs=np.array(time_lbs, dtype="datetime64[us]"),
            time_ubs=np.array(time_ubs, dtype="datetime64[us]"),
        )

    def generate(self, sequences):
        input_sequences = sequences
        samplex_items = []
        mapper = {}
        creg_names = []
        for c_idx, partition in enumerate(self.partition(sequences)):
            sequences = []
            for p_idx, p in enumerate(partition):
                mapper[p[0]] = (c_idx, p_idx)
                sequences.append(p[1])
            samplex_items.append(samplex_item := self.generate_samplex_item(sequences))
            creg_names.append([c.name for c in samplex_item.circuit.cregs])

        return samplex_items, ExecutorDataMapper(
            mapper, creg_names, input_sequences, self._num_randomizations
        )

    def generate_samplex_item(self, sequences: list[InstructionSequence]) -> SamplexItem:
        """Generate a samplex item from instruction sequences with the same structure.

        Args:
            sequence: The similar instruction sequences to generate.

        Returns:
            A samplex item where the order of the arguments correspond to the order of
            ``sequences``.

        Raises:
            ValueError: If ``sequences`` is empty.
            ValueError: If any of the instruction sequences is not complete.
            ValueError: If any of the instruction sequences have different structure.
        """
        if (num_sequences := len(sequences)) == 0:
            raise ValueError("At least one instruction sequence is expected to generate circuits.")

        boxed_circuit = QuantumCircuit(self.gate_set.num_qubits)

        ref_iter = (f"c{idx}" for idx in count())
        creg_iter = (f"meas{idx}" for idx in count())

        gateset_idxs = list(self.gate_set.qubit_subset)
        gateset_idxs.sort()

        first_sequence = sequences[0]
        samplex_arguments = {}
        current_permutation = PartialPauliPermutation([0] * self.gate_set.num_qubits)
        for instr in first_sequence:
            if isinstance(instr, PartialPauliPermutation):
                if not instr.is_complete:
                    raise ValueError("Encountered an incomplete Pauli permutation.")
                current_permutation = instr.compose(current_permutation)
            elif isinstance(instr, ApplyGate):
                gate = self.gate_set[instr.gate.name]

                body = QuantumCircuit([boxed_circuit.qubits[idx] for idx in gate.qubit_idxs])
                ref = next(ref_iter)

                annotations = []
                for annotation in gate.annotations:
                    if isinstance(annotation, Twirl):
                        annotations.append(annotation)
                        annotations.append(InjectLocalClifford(ref, annotation.decomposition))
                    if isinstance(annotation, Tag):
                        annotations.append(annotation)

                if num_meas := len(gate.meas_idxs):
                    creg = ClassicalRegister(num_meas, next(creg_iter))
                    boxed_circuit.add_register(creg)
                    body.add_register(creg)
                    body.compose(gate.circuit, qubits=body.qubits, clbits=creg, inplace=True)
                    box = BoxOp(body, annotations=annotations)
                    boxed_circuit.append(CircuitInstruction(box, gate.qubit_idxs, creg))
                else:
                    body.compose(gate.circuit, qubits=body.qubits, inplace=True)
                    box = BoxOp(body, annotations=annotations)
                    boxed_circuit.append(CircuitInstruction(box, gate.qubit_idxs, []))

                this_arg = np.empty((num_sequences, 1, gate.num_qubits), dtype=np.uint8)
                this_arg[0, 0] = TO_SAMPLOMATIC_C1[
                    current_permutation.partial_permutation_indices[gateset_idxs]
                ]
                current_permutation = PartialPauliPermutation([0] * self.gate_set.num_qubits)
                samplex_arguments[f"local_cliffords.{ref}"] = this_arg

        for idx, following_sequence in enumerate(sequences[1:]):
            if not first_sequence.has_same_structure_as(following_sequence):
                raise ValueError(
                    "Instruction sequences require the same structure to be generated together."
                )

            current_permutation = PartialPauliPermutation([0] * self.gate_set.num_qubits)
            ref_iter = (f"c{idx}" for idx in count())
            for instr in following_sequence:
                if isinstance(instr, PartialPauliPermutation):
                    current_permutation = instr.compose(current_permutation)
                elif isinstance(instr, ApplyGate):
                    samplex_arguments[f"local_cliffords.{next(ref_iter)}"][idx + 1, 0] = (
                        TO_SAMPLOMATIC_C1[
                            current_permutation.partial_permutation_indices[gateset_idxs]
                        ]
                    )
                    current_permutation = PartialPauliPermutation([0] * self.gate_set.num_qubits)

        template, samplex = build(boxed_circuit)

        return SamplexItem(
            template,
            samplex=samplex,
            samplex_arguments=samplex_arguments,
            shape=(num_sequences, self._num_randomizations),
        )
