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
import xarray as xr
from qiskit.circuit import BoxOp, CircuitInstruction, ClassicalRegister, QuantumCircuit
from qiskit.transpiler import PassManager
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
        item_sequence_indices: For each program item, an ordered list of instruction sequence
            indices. Position in the list corresponds to the configuration index within the result
            item.
        creg_names: The name of classical registers in each program item.
        measurement_maps: For each program item, a dictionary from creg names to an ordered array
            of measured qubit indices.
        instruction_sequences: The instruction sequences associated with the data.
        num_randomizations: The number of randomizations used per experiment.

    """

    def __init__(
        self,
        item_sequence_indices: list[list[int]],
        creg_names: list[list[str]],
        measurement_maps: list[dict[str, np.ndarray[int]]],
        instruction_sequences: list[InstructionSequence],
        num_randomizations: int,
    ):
        self._item_sequence_indices = item_sequence_indices
        self._creg_names = creg_names
        self._measurement_maps = measurement_maps
        self._instruction_sequences = instruction_sequences
        self._num_randomizations = num_randomizations

    @property
    def item_sequence_indices(self) -> list[list[int]]:
        """Per program item, the instruction sequence indices corresponding to each config."""
        return self._item_sequence_indices

    @property
    def creg_names(self) -> list[list[str]]:
        """List of names of the classical registers contained in the results.

        The list at a given index corresponds to names expected in the data of the
        :class:`qiskit_ibm_runtime.quantum_program.QuantumProgramResult` at the same index.
        """
        return self._creg_names

    @property
    def measurement_maps(self) -> list[dict[str, np.ndarray[int]]]:
        """A per-program-item map from creg name to an ordered array of measured qubit indices."""
        return self._measurement_maps

    @property
    def instruction_sequences(self) -> list:
        """The instruction sequences corresponding to the sequence indices in the sequence map."""
        return self._instruction_sequences

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
        creg_prefix: The prefix assigned to all creg names used in instruction sequence
            measurements. Defaults to ``"meas"``.
        local_clifford_ref_prefix: The prefix assigned to all local Clifford parameter references
            in template circuits. Defaults to ``"c"``.
        pass_manager: An optional ``PassManager`` to apply to all template circuits produced by
            :meth:`ExecutorCircuitGenerator.generate`.
    """

    def __init__(
        self,
        gate_set: QiskitGateSet,
        num_randomizations: int = 50,
        creg_prefix: str = "meas",
        local_clifford_ref_prefix: str = "c",
        pass_manager: PassManager | None = None,
    ):
        self._gate_set = gate_set
        self._num_randomizations = num_randomizations
        self._creg_prefix = creg_prefix
        self._local_clifford_ref_prefix = local_clifford_ref_prefix
        self._pass_manager = pass_manager

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
            num_seqs_per_item = max(len(indices) for indices in data_mapper.item_sequence_indices)
            the_length = num_seqs_per_item * data_mapper.num_randomizations
            program_item_time_lbs = np.full(
                (len(result), the_length), "NaT", dtype="datetime64[us]"
            )
            program_item_time_ubs = np.full(
                (len(result), the_length), "NaT", dtype="datetime64[us]"
            )

        raw_data = RawData(datatree=xr.DataTree())

        for item_idx, seq_indices in enumerate(data_mapper.item_sequence_indices):
            this_item = result[item_idx]
            item_creg_names = data_mapper.creg_names[item_idx]
            item_measurement_map = data_mapper.measurement_maps[item_idx]

            data = []
            measurement_flips = []
            time_lbs = []
            time_ubs = []
            instruction_sequences = []

            for d_idx, seq_idx in enumerate(seq_indices):
                this_data = []
                these_flips = []
                for creg in item_creg_names:
                    this_data.append(this_item[creg][d_idx])
                    flips = this_item.get(f"measurement_flips.{creg}")
                    if flips is None:
                        flips = np.zeros(
                            (this_data[-1].shape[0], 1, this_data[-1].shape[-1]), dtype=np.bool_
                        )
                    else:
                        flips = flips[d_idx]
                    these_flips.append(flips)

                data.append(np.concatenate(this_data, axis=-1))
                flips_array = np.concatenate(these_flips, axis=-1)
                measurement_flips.append(
                    flips_array.reshape((flips_array.shape[0], flips_array.shape[-1]))
                )

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
                instruction_sequences.append(data_mapper.instruction_sequences[seq_idx])

            item_raw_data = RawData.from_arrays(
                creg_names=item_creg_names,
                measurement_map=item_measurement_map,
                instruction_sequences=instruction_sequences,
                data=data,
                measurement_flips=measurement_flips,
                time_lbs=time_lbs,
                time_ubs=time_ubs,
            )
            raw_data = raw_data.merge(item_raw_data)

        return raw_data

    def generate(self, sequences):
        samplex_items = []
        item_sequence_indices = []
        creg_names = []
        measurement_maps = []
        for partition in self.partition(sequences):
            current_sequences = []
            current_indices = []
            for seq_idx, seq in partition:
                current_indices.append(seq_idx)
                current_sequences.append(seq)
            samplex_item, current_creg_names, current_meas_map = self.generate_samplex_item(
                current_sequences
            )

            samplex_items.append(samplex_item)
            item_sequence_indices.append(current_indices)
            creg_names.append(current_creg_names)
            measurement_maps.append(current_meas_map)

        return samplex_items, ExecutorDataMapper(
            item_sequence_indices=item_sequence_indices,
            creg_names=creg_names,
            measurement_maps=measurement_maps,
            instruction_sequences=sequences,
            num_randomizations=self._num_randomizations,
        )

    def generate_samplex_item(
        self, sequences: list[InstructionSequence]
    ) -> tuple[SamplexItem, list[str], dict[str, np.ndarray[int]]]:
        """Generate a samplex item from instruction sequences with the same structure.

        Args:
            sequence: The similar instruction sequences to generate.

        Returns:
            A samplex item where the order of the arguments correspond to the order of
            ``sequences``, an ordered list of creg names, and a dictionary mapping creg names to
            the ordered list of qubit indices they measure.

        Raises:
            ValueError: If ``sequences`` is empty.
            ValueError: If any of the instruction sequences is not complete.
            ValueError: If any of the instruction sequences have different structure.
        """
        if (num_sequences := len(sequences)) == 0:
            raise ValueError("At least one instruction sequence is expected to generate circuits.")

        boxed_circuit = QuantumCircuit(self.gate_set.num_qubits)

        ref_iter = (f"{self._local_clifford_ref_prefix}{idx}" for idx in count())
        creg_iter = (f"{self._creg_prefix}{idx}" for idx in count())
        creg_names = []
        measurement_map = dict()

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
                    creg_names.append(next(creg_iter))
                    measurement_map[creg_names[-1]] = np.array(sorted(gate.meas_idxs), dtype=int)

                    creg = ClassicalRegister(num_meas, creg_names[-1])
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
        if self._pass_manager is not None:
            template = self._pass_manager.run(template)

            # add additional creg names from the pass manager
            for creg in template.cregs:
                if creg.name not in creg_names:
                    creg_names.append(creg.name)

            # add measurement map information for added cregs
            original_creg_names = set(measurement_map)
            for instruction in template.data:
                if instruction.name == "measure":
                    qubit_idx = template.find_bit(instruction.qubits[0]).index
                    clbit = instruction.clbits[0]
                    for creg in template.cregs:
                        if creg.name not in original_creg_names and clbit in creg:
                            measurement_map.setdefault(creg.name, []).append(qubit_idx)
                            break

            for key, val in measurement_map.items():
                if isinstance(val, list):
                    measurement_map[key] = np.array(sorted(val), dtype=int)

        return (
            SamplexItem(
                template,
                samplex=samplex,
                samplex_arguments=samplex_arguments,
                shape=(num_sequences, self._num_randomizations),
            ),
            creg_names,
            measurement_map,
        )
