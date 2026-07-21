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

"""Path"""

from qiskit.quantum_info import PhasedQubitSparsePauli, QubitSparsePauli

from .apply_gate import ApplyGate
from .base_sequence import BaseSequence
from .fidelity_index import FidelityIndex
from .instruction import Instruction
from .instruction_sequence import InstructionSequence
from .partial_pauli_permutation import PartialPauliPermutation


class Path(BaseSequence[FidelityIndex]):
    """A sequence of fidelity indices.

    Args:
        start_fragment: The start of the sequence.
        repeatable_fragment: The repeatable middle of the sequence.
        end_fragment: The end of the sequence.
        fragment_depth: The number of repetitions of the repeatable fragment.
    """

    @property
    def start_fragment_observable_indices(self) -> list[list[int]]:
        """The observable indices of the start fragment."""
        return [x.observable_indices for x in self.start_fragment]

    @property
    def repeatable_fragment_observable_indices(self) -> list[list[int]]:
        """The observable indices of the repeatable fragment."""
        return [x.observable_indices for x in self.repeatable_fragment]

    @property
    def end_fragment_observable_indices(self) -> list[list[int]]:
        """The observable indices of the end fragment."""
        return [x.observable_indices for x in self.end_fragment]

    def extend_permutations(
        self, instruction_sequence: InstructionSequence
    ) -> InstructionSequence | None:
        r"""Return an instruction sequence with extended permutations to traverse self.

        Given an ``instruction_sequence``, this method attempts to extend its definition by
        constructing a new instruction sequence with the same structure as the input, but for which
        the :class:`PartialPauliPermutation`\s are extended to specify the minimal superset of
        mappings required to traverse this path. If this is not possible, the function will return
        ``None``, indicating that the procedure failed.

        In terms of the specific algorithm, when encountering a :class:`PartialPauliPermutation`
        while jointly iterating through ``self`` and ``instruction_sequence``, this method will
        attempt to extend the permutation to map the current Pauli in the path to the Pauli in the
        next relevant :class:`FidelityIndex`. Note that this can lead to failures in instances when
        multiple permutations occur in a row, even if it is technically possible to find a super
        instruction sequence that traverses the path.

        Args:
            instruction_sequence: The base instruction sequence.

        Returns:
            The extended instruction sequence, or ``None`` if an extension is not possible.

        Raises:
            ValueError: If the transitions of ``self`` do not start and end with the identity.
        """
        if not self.is_unbound and self.fragment_depth != instruction_sequence.fragment_depth:
            return None

        ident = self._validate_starts_and_ends_with_identity()
        current_pauli = ident

        # get super start fragment
        if self.repeatable_fragment:
            expected_start_output = self.repeatable_fragment[0].transition[0]
        elif self.end_fragment:
            expected_start_output = self.end_fragment[0].transition[0]
        else:
            expected_start_output = ident

        new_start_fragment = _extend_instruction_fragment(
            self.start_fragment,
            instruction_sequence.start_fragment,
            input_pauli=current_pauli,
            expected_output_pauli=expected_start_output,
        )
        if new_start_fragment is None:
            return None

        # get super repeatable fragment
        new_repeatable_fragment = _extend_instruction_fragment(
            self.repeatable_fragment,
            instruction_sequence.repeatable_fragment,
            input_pauli=expected_start_output,
            expected_output_pauli=expected_start_output,
        )
        if new_repeatable_fragment is None:
            return None

        # get super end fragment
        new_end_fragment = _extend_instruction_fragment(
            self.end_fragment,
            instruction_sequence.end_fragment,
            input_pauli=expected_start_output,
            expected_output_pauli=ident,
        )
        if new_end_fragment is None:
            return None

        return InstructionSequence(
            start_fragment=new_start_fragment,
            repeatable_fragment=new_repeatable_fragment,
            end_fragment=new_end_fragment,
            fragment_depth=instruction_sequence.fragment_depth,
        )

    def is_traversed_by(self, instruction_sequence: InstructionSequence) -> bool:
        """Whether or not this path is traversed by the instruction sequence.

        Whether or not a :class:`Path` is traversed by an :class:`InstructionSequence` is determined
        on a fragment-by-fragment basis.

        Requires the path starts and ends at the identity.

        Args:
            instruction_sequence: The instruction sequence.

        Returns:
            Whether or not the path is traversed.

        Raises:
            ValueError: If the path does not start and end at the identity.
        """
        if not self.is_unbound and self.fragment_depth != instruction_sequence.fragment_depth:
            return False

        current_pauli = self._validate_starts_and_ends_with_identity()

        if instruction_sequence.start_fragment:
            current_pauli = _instruction_fragment_traverses_path_fragment(
                current_pauli,
                self.start_fragment,
                instruction_sequence.start_fragment,
            )
            if current_pauli is None:
                return False
        elif self.start_fragment:
            return False

        if instruction_sequence.repeatable_fragment:
            out_pauli = _instruction_fragment_traverses_path_fragment(
                current_pauli,
                self.repeatable_fragment,
                instruction_sequence.repeatable_fragment,
            )
            # check consistency of input and output for repeatable fragment
            if out_pauli is None or out_pauli != current_pauli:
                return False
            current_pauli = out_pauli
        elif self.repeatable_fragment:
            return False

        if instruction_sequence.end_fragment:
            current_pauli = _instruction_fragment_traverses_path_fragment(
                current_pauli,
                self.end_fragment,
                instruction_sequence.end_fragment,
            )
            if current_pauli is None:
                return False
        elif self.end_fragment:
            return False

        return True

    def fragment_sign_flips(self, instruction_sequence: InstructionSequence) -> tuple[bool, bool]:
        """Whether the instruction sequence fragments flip the observable sign when traversing self.

        Requires the path starts and ends at the identity. This method ignores the fragment depths
        of the path and instruction sequence, operating only on the fragment structure.

        Args:
            instruction_sequence: An instruction sequence that traverses this path.

        Returns:
            A tuple of booleans, the first indicating whether the combined action of the start
            and end fragments flip the sign, and the second indicating whether the repeatable
            fragment flips the sign.

        Raises:
            ValueError: If the path does not satisfy the assumptions.
            ValueError: If the instruction sequence does not traverse the path.
        """
        # validate and turn into phased pauli
        current_pauli = self._validate_starts_and_ends_with_identity()
        current_pauli = _unphased_pauli_to_phased(negative=False, pauli=current_pauli)

        start_error = ValueError(
            "Cannot compute signs as instruction sequence start fragment does not traverse "
            "path start fragment."
        )
        if instruction_sequence.start_fragment:
            out_pauli = _instruction_fragment_traverses_path_fragment(
                current_pauli,
                self.start_fragment,
                instruction_sequence.start_fragment,
            )
            if out_pauli is None:
                raise start_error

            start_flip = out_pauli.phase != current_pauli.phase
            current_pauli = out_pauli
        elif self.start_fragment:
            raise start_error

        # default value
        repeatable_flip = False
        repeatable_error = ValueError(
            "Cannot compute signs as instruction sequence repeatable fragment does not traverse "
            "path repeatable fragment."
        )
        # default value
        repeatable_flip = False
        if instruction_sequence.repeatable_fragment:
            out_pauli = _instruction_fragment_traverses_path_fragment(
                current_pauli,
                self.repeatable_fragment,
                instruction_sequence.repeatable_fragment,
            )
            # check consistency of input and output for repeatable fragment
            if (
                out_pauli is None
                or _phased_pauli_to_unphased(out_pauli)[1]
                != _phased_pauli_to_unphased(current_pauli)[1]
            ):
                raise repeatable_error

            repeatable_flip = out_pauli.phase != current_pauli.phase
            current_pauli = out_pauli
        elif self.repeatable_fragment:
            raise repeatable_error

        end_error = ValueError(
            "Cannot compute signs as instruction sequence end fragment does not traverse "
            "path end fragment."
        )
        if instruction_sequence.end_fragment:
            out_pauli = _instruction_fragment_traverses_path_fragment(
                current_pauli,
                self.end_fragment,
                instruction_sequence.end_fragment,
            )
            if out_pauli is None:
                raise end_error

            end_flip = out_pauli.phase != current_pauli.phase
        elif self.end_fragment:
            raise end_error

        return start_flip ^ end_flip, repeatable_flip

    def to_instruction_sequence(self) -> InstructionSequence:
        r"""Return a minimally-specified instruction sequence that traverses this path.

        The single-qubit Clifford layers between gate set elements are given as
        :class:`.PartialPauliPermutation`\s specifying only the mappings required to traverse
        this path. The fragment depth of the returned instruction sequence is
        ``self.fragment_depth``.

        Returns:
            An instruction sequence traversing this path.
        """

        # initialize output fragments
        start_fragment = _path_fragment_to_instruction_fragment(self.start_fragment)
        repeatable_fragment = _path_fragment_to_instruction_fragment(self.repeatable_fragment)
        end_fragment = _path_fragment_to_instruction_fragment(self.end_fragment)

        # handle boundaries
        if self.repeatable_fragment:
            if self.start_fragment:
                # add the final permutation for exiting out of the start fragment
                start_fragment.append(
                    PartialPauliPermutation.from_qubit_sparse_paulis(
                        self.start_fragment[-1].transition[1],
                        self.repeatable_fragment[0].transition[0],
                    )
                )

            # add permutation that loops back to beginning of repeatable fragment
            repeatable_fragment_final_permutation = (
                PartialPauliPermutation.from_qubit_sparse_paulis(
                    self.repeatable_fragment[-1].transition[1],
                    self.repeatable_fragment[0].transition[0],
                )
            )
            repeatable_fragment.append(repeatable_fragment_final_permutation)

            if self.end_fragment:
                # finally, determine first permutation in end fragment. This is an intermediate
                # permutation, that needs to be modified to compensate for the permutation at the
                # end of repeatable_fragment
                end_fragment_intermediate_permutation = (
                    PartialPauliPermutation.from_qubit_sparse_paulis(
                        self.repeatable_fragment[-1].transition[1],
                        self.end_fragment[0].transition[0],
                    )
                )

                new_sets = []
                for in_set, out_set in zip(
                    repeatable_fragment_final_permutation.to_sets(),
                    end_fragment_intermediate_permutation.to_sets(),
                ):
                    # both should either contain one or no elements
                    if in_set:
                        # the output of the "in" permutation needs to be mapped to the output of the
                        # "out" permutation
                        new_sets.append({(next(iter(in_set))[1], next(iter(out_set))[1])})
                    else:
                        new_sets.append(set())

                end_fragment.insert(0, PartialPauliPermutation.from_sets(new_sets))
        elif len(self.start_fragment) != 0 and len(self.end_fragment) != 0:
            # if self.repeatable_fragment is empty, add a boundary permutation to start_fragment
            start_fragment.append(
                PartialPauliPermutation.from_qubit_sparse_paulis(
                    self.start_fragment[-1].transition[1], self.end_fragment[0].transition[0]
                )
            )

        return InstructionSequence(
            start_fragment=start_fragment,
            repeatable_fragment=repeatable_fragment,
            end_fragment=end_fragment,
            fragment_depth=self.fragment_depth,
        )

    def _validate_starts_and_ends_with_identity(self) -> QubitSparsePauli:
        """Validate the path starts and ends at identity, and return the identity.

        Returns:
            The appropriately sized identity.

        Raises:
            ValueError: If the path does not start and end at the identity.
        """

        start_pauli = None
        for fragment in [self.start_fragment, self.repeatable_fragment, self.end_fragment]:
            if fragment:
                start_pauli = fragment[0].transition[0]
                break

        if start_pauli != QubitSparsePauli.identity(start_pauli.num_qubits):
            raise ValueError("Path does not begin with identity.")

        final_pauli = None
        for fragment in [self.end_fragment, self.repeatable_fragment, self.start_fragment]:
            if fragment:
                final_pauli = fragment[-1].transition[1]
                break

        if final_pauli != QubitSparsePauli.identity(final_pauli.num_qubits):
            raise ValueError("Path does not end with identity.")

        return start_pauli

    def __hash__(self) -> int:
        if not hasattr(self, "_hash"):
            self._hash = hash(
                (
                    tuple(self.start_fragment),
                    tuple(self.repeatable_fragment),
                    tuple(self.end_fragment),
                    self.fragment_depth,
                )
            )
        return self._hash


def _path_fragment_to_instruction_fragment(path_fragment: list[FidelityIndex]) -> list[Instruction]:
    """Given a path fragment, build a minimal instruction fragment that traverses it.

    See the documentation for :meth:`Path.to_instruction_sequence` for a description of the
    overall algorithm.
    """
    if not path_fragment:
        return []

    instruction_fragment = []

    fidelity_idx0 = path_fragment[0]
    instruction_fragment.append(ApplyGate(fidelity_idx0.gate_name))
    for fidelity_idx1 in path_fragment[1:]:
        # connect the input and output Paulis
        instruction_fragment.append(
            PartialPauliPermutation.from_qubit_sparse_paulis(
                fidelity_idx0.transition[1], fidelity_idx1.transition[0]
            )
        )
        instruction_fragment.append(ApplyGate(fidelity_idx1.gate_name))
        fidelity_idx0 = fidelity_idx1

    return instruction_fragment


def _instruction_fragment_traverses_path_fragment(
    in_pauli: QubitSparsePauli | PhasedQubitSparsePauli,
    path_fragment: list[FidelityIndex],
    instruction_fragment: list[Instruction],
) -> QubitSparsePauli | PhasedQubitSparsePauli | None:
    """Whether the input Pauli traverses the given path on the instruction fragment.

    Args:
        in_pauli: The input Pauli.
        path_fragment: The path fragment.
        instruction_fragment: The instruction fragment.

    Returns:
        ``None`` if the path is not traversed, or the output Pauli if it is traversed.
    """

    try:
        path_idx = 0
        current_pauli = in_pauli

        for instruction in instruction_fragment:
            # iterate through any partial permutations
            if isinstance(instruction, PartialPauliPermutation):
                current_pauli = instruction.propagate(current_pauli)
            elif isinstance(instruction, ApplyGate):
                if isinstance(current_pauli, QubitSparsePauli):
                    # return None if gate differs from what is expected for the current path entry
                    if (
                        path_idx >= len(path_fragment)
                        or instruction.gate_name != path_fragment[path_idx].gate_name
                        or current_pauli != path_fragment[path_idx].transition[0]
                    ):
                        return None

                    current_pauli = path_fragment[path_idx].transition[1]
                else:
                    negative, unphased_pauli = _phased_pauli_to_unphased(current_pauli)

                    # return None if gate differs from what is expected for the current path entry
                    if (
                        path_idx >= len(path_fragment)
                        or instruction.gate_name != path_fragment[path_idx].gate_name
                        or unphased_pauli != path_fragment[path_idx].transition[0]
                    ):
                        return None

                    current_pauli = _unphased_pauli_to_phased(
                        negative=negative ^ path_fragment[path_idx].sign_flip,
                        pauli=path_fragment[path_idx].transition[1],
                    )

                path_idx += 1

        # if still elements remaining, return None
        if path_idx != len(path_fragment):
            return None

        return current_pauli
    except ValueError as e:
        # failure is due to lack of definition on Pauli, return None
        if str(e) == "PartialPauliPermutation is undefined on given pauli.":
            return None
        raise e


def _phased_pauli_to_unphased(pauli: PhasedQubitSparsePauli) -> tuple[bool, QubitSparsePauli]:
    return pauli.phase == 2, QubitSparsePauli.from_raw_parts(
        num_qubits=pauli.num_qubits, paulis=pauli.paulis, indices=pauli.indices
    )


def _unphased_pauli_to_phased(negative: bool, pauli: QubitSparsePauli) -> PhasedQubitSparsePauli:
    return PhasedQubitSparsePauli.from_raw_parts(
        num_qubits=pauli.num_qubits,
        paulis=pauli.paulis,
        indices=pauli.indices,
        phase=2 if negative else 0,
    )


def _extend_instruction_fragment(
    path_fragment: list[FidelityIndex],
    instruction_fragment: list[Instruction],
    input_pauli: QubitSparsePauli,
    expected_output_pauli: QubitSparsePauli,
) -> list[Instruction] | None:
    """Return an instruction fragment that traverses a superset of paths including path fragment."""
    if len(instruction_fragment) == 0:
        if len(path_fragment) == 0:
            return []
        elif len(path_fragment) != 0:
            return None

    new_fragment = []
    path_idx = 0
    current_pauli = input_pauli
    for instruction in instruction_fragment:
        if isinstance(instruction, ApplyGate):
            if (
                path_idx >= len(path_fragment)
                or instruction.gate_name != path_fragment[path_idx].gate_name
                or current_pauli != path_fragment[path_idx].transition[0]
                or path_idx == len(path_fragment)
            ):
                return None

            new_fragment.append(instruction)
            current_pauli = path_fragment[path_idx].transition[1]
            path_idx += 1
        elif isinstance(instruction, PartialPauliPermutation):
            if path_idx == len(path_fragment):
                next_pauli = expected_output_pauli
            else:
                next_pauli = path_fragment[path_idx].transition[0]
            permutation = PartialPauliPermutation.from_qubit_sparse_paulis(
                current_pauli, next_pauli
            )
            if not permutation.is_mergeable_with(instruction):
                return None
            new_fragment.append(permutation.merge(instruction))
            current_pauli = next_pauli

    if path_idx != len(path_fragment) or current_pauli != expected_output_pauli:
        return None

    return new_fragment
