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

from qiskit_noise_learning.models import FidelityModel
from qiskit_noise_learning.sequences import InstructionSequence, Path


class ExecutorDataMapper:
    """Map executor results into standard results.

    As instruction sequences with similar structure are generated together with a single template
    circuit and different samplex arguments, the order of input sequences to
    :meth:`.ExecutorCircuitGenerator.generate` is not preserved during execution. This class
    contains properties to format the results of a
    :class:`qiskit_ibm_runtime.results.QuantumProgramResult` to the order of the input sequences.

    Args:
        item_sequence_indices: For each program item, an ordered list of instruction sequence
            indices. Position in the list corresponds to the configuration index within the result
            item.
        creg_names: The name of classical registers in each program item.
        measurement_maps: For each program item, a dictionary from creg names to an ordered array
            of measured qubit indices.
        instruction_sequences: The instruction sequences associated with the data.
        num_randomizations: The number of randomizations used per experiment.
        fidelity_model: The fidelity model used in the experiment.
        paths: The analysis paths.
        relations: Path-to-sequence relations.

    """

    def __init__(
        self,
        item_sequence_indices: list[list[int]],
        creg_names: list[list[str]],
        measurement_maps: list[dict[str, np.ndarray[int]]],
        instruction_sequences: list[InstructionSequence],
        num_randomizations: int,
        fidelity_model: FidelityModel | None = None,
        paths: list[Path] | None = None,
        relations: set[tuple[int, int]] | None = None,
    ):
        self._item_sequence_indices = item_sequence_indices
        self._creg_names = creg_names
        self._measurement_maps = measurement_maps
        self._instruction_sequences = instruction_sequences
        self._num_randomizations = num_randomizations
        self._fidelity_model = fidelity_model
        self._paths = paths
        self._relations = relations

    @property
    def item_sequence_indices(self) -> list[list[int]]:
        """Per program item, the instruction sequence indices corresponding to each config."""
        return self._item_sequence_indices

    @property
    def creg_names(self) -> list[list[str]]:
        """List of names of the classical registers contained in the results.

        The list at a given index corresponds to names expected in the data of the
        :class:`qiskit_ibm_runtime.results.QuantumProgramResult` at the same index.
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

    @property
    def fidelity_model(self) -> FidelityModel | None:
        """The fidelity model used in the experiment."""
        return self._fidelity_model

    @property
    def paths(self) -> list[Path] | None:
        """The analysis paths."""
        return self._paths

    @property
    def relations(self) -> set[tuple[int, int]] | None:
        """Path-to-sequence relations."""
        return self._relations
