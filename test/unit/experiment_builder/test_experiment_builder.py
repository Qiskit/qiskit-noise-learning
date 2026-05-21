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

from itertools import product

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CZGate, XGate
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.experiment_builder.experiment import Experiment
from qiskit_noise_learning.experiment_builder.experiment_builder import (
    ExperimentBuilder,
    minimize_instruction_sequences,
)
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.math import IndexedMatrix
from qiskit_noise_learning.models import CompleteFidelityModel
from qiskit_noise_learning.sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionSequence,
    PartialPauliPermutation,
    Path,
)


@pytest.fixture()
def gate_set_1q():
    model_gate_set = ModelGateSet(1)
    ident = Clifford(QuantumCircuit(1))
    model_gate_set.add_gate(ModelGate("P", [((0,), ident)], prep_idxs=range(1)))
    model_gate_set.add_gate(ModelGate("M", [((0,), ident)], meas_idxs=range(1)))
    # Clifford maps X -> -Y, Y -> Z, Z -> -X
    model_gate_set.add_gate(
        ModelGate("L0", [((0,), Clifford([[True, True, True], [True, False, True]]))])
    )
    model_gate_set.add_gate(ModelGate("L1", [((0,), Clifford(XGate()))]))
    return model_gate_set


@pytest.fixture()
def gate_set_cz():
    model_gate_set = ModelGateSet(2)
    model_gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    model_gate_set.add_gate(
        ModelGate("P", [((0, 1), Clifford(QuantumCircuit(2)))], prep_idxs=range(2))
    )
    model_gate_set.add_gate(
        ModelGate("M", [((0, 1), Clifford(QuantumCircuit(2)))], meas_idxs=range(2))
    )
    return model_gate_set


@pytest.fixture()
def unbound_path_ix(gate_set_cz):
    return Path(
        start_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["P"],
                in_pauli=QubitSparsePauli("II"),
                out_pauli=QubitSparsePauli("IZ"),
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("IX"),
                out_pauli=QubitSparsePauli("ZX"),
            ),
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("ZX"),
                out_pauli=QubitSparsePauli("IX"),
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["M"],
                in_pauli=QubitSparsePauli("IZ"),
                out_pauli=QubitSparsePauli("II"),
            )
        ],
    )


@pytest.fixture()
def unbound_path_xi(gate_set_cz):
    return Path(
        start_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["P"],
                in_pauli=QubitSparsePauli("II"),
                out_pauli=QubitSparsePauli("ZI"),
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("XI"),
                out_pauli=QubitSparsePauli("XZ"),
            ),
            FidelityIndex.from_transition(
                gate=gate_set_cz["CZ"],
                in_pauli=QubitSparsePauli("XZ"),
                out_pauli=QubitSparsePauli("XI"),
            ),
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_cz["M"],
                in_pauli=QubitSparsePauli("ZI"),
                out_pauli=QubitSparsePauli("II"),
            )
        ],
    )


class TestExperimentBuilder:
    """Tests for ExperimentBuilder."""

    def test_construction(self, gate_set_1q):
        experiment_builder = ExperimentBuilder(gate_set_1q)
        assert experiment_builder.gate_set == gate_set_1q
        assert experiment_builder.paths == []
        assert experiment_builder.instruction_sequences == []
        assert experiment_builder.relations == set()
        assert experiment_builder.design_matrix == IndexedMatrix()

        fidelity_model = CompleteFidelityModel(gate_set_1q)
        experiment_builder = ExperimentBuilder(fidelity_model)
        assert experiment_builder.gate_set == gate_set_1q
        assert experiment_builder.fidelity_model == fidelity_model
        assert experiment_builder.paths == []
        assert experiment_builder.instruction_sequences == []
        assert experiment_builder.relations == set()
        assert experiment_builder.design_matrix == IndexedMatrix()

    @pytest.mark.parametrize("attempt_instruction_merge", [True, False])
    def test_add_paths_unbound_merge(self, gate_set_cz, attempt_instruction_merge):
        experiment_builder = ExperimentBuilder(gate_set_cz)

        path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
        )
        mergeable_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("ZI")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZI"), QubitSparsePauli("ZI")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZI"), QubitSparsePauli("ZI")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("ZI"), QubitSparsePauli("II")
                )
            ],
        )

        experiment_builder.add_paths([(path, None)])
        assert experiment_builder.paths == [path]
        assert experiment_builder.instruction_sequences == [path.to_instruction_sequence()]
        assert experiment_builder.relations == {(0, 0)}
        expected_matrix = IndexedMatrix()
        expected_matrix.add_rows(
            row_indices=[path],
            rows=[experiment_builder.fidelity_model.row_from_path(path)],
        )
        assert experiment_builder.design_matrix == expected_matrix

        # add mergeable path
        experiment_builder.add_paths(
            [(mergeable_path, None)], attempt_instruction_merge=attempt_instruction_merge
        )
        if not attempt_instruction_merge:
            experiment_builder.merge_instruction_sequences()

        assert experiment_builder.paths == [path, mergeable_path]
        assert experiment_builder.instruction_sequences == [
            path.to_instruction_sequence().merge(mergeable_path.to_instruction_sequence())
        ]
        assert experiment_builder.relations == {(0, 0), (1, 0)}
        expected_matrix.add_rows(
            row_indices=[mergeable_path],
            rows=[experiment_builder.fidelity_model.row_from_path(mergeable_path)],
        )
        assert experiment_builder.design_matrix == expected_matrix

    def test_add_paths_unbound_no_merge(self, gate_set_cz):
        path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
        )
        mergeable_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("ZI")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZI"), QubitSparsePauli("ZI")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZI"), QubitSparsePauli("ZI")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("ZI"), QubitSparsePauli("II")
                )
            ],
        )

        experiment_builder = ExperimentBuilder(gate_set_cz)
        experiment_builder.add_paths(
            [(path, None), (mergeable_path, None)], attempt_instruction_merge=False
        )
        assert experiment_builder.paths == [path, mergeable_path]
        assert experiment_builder.instruction_sequences == [
            path.to_instruction_sequence(),
            mergeable_path.to_instruction_sequence(),
        ]
        assert experiment_builder.relations == {(0, 0), (1, 1)}

        row_indices = [path, mergeable_path]
        rows = [experiment_builder.fidelity_model.row_from_path(p) for p in row_indices]
        expected_matrix = IndexedMatrix()
        expected_matrix.add_rows(row_indices=row_indices, rows=rows)
        assert experiment_builder.design_matrix == expected_matrix

    def test_add_paths_unbound_linearly_dependent(self, gate_set_cz):
        path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IX"), QubitSparsePauli("ZX")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZX"), QubitSparsePauli("IX")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
        )
        dependent_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("ZZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZX"), QubitSparsePauli("IX")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IX"), QubitSparsePauli("ZX")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("ZZ"), QubitSparsePauli("II")
                )
            ],
        )
        experiment_builder = ExperimentBuilder(gate_set_cz)
        experiment_builder.add_paths([(path, None), (dependent_path, None)])
        assert experiment_builder.paths == [path]
        assert experiment_builder.instruction_sequences == [path.to_instruction_sequence()]
        assert experiment_builder.relations == {(0, 0)}
        expected_matrix = IndexedMatrix()
        expected_matrix.add_rows(
            row_indices=[path],
            rows=[experiment_builder.fidelity_model.row_from_path(path)],
        )
        assert experiment_builder.design_matrix == expected_matrix

        # manually override rank checking
        experiment_builder = ExperimentBuilder(gate_set_cz)
        experiment_builder.add_paths(
            [(path, None), (dependent_path, None)],
            rank_reduce=False,
            attempt_instruction_merge=False,
        )
        assert experiment_builder.paths == [path, dependent_path]
        assert experiment_builder.instruction_sequences == [
            path.to_instruction_sequence(),
            dependent_path.to_instruction_sequence(),
        ]
        assert experiment_builder.relations == {(0, 0), (1, 1)}
        expected_matrix.add_rows(
            row_indices=[dependent_path],
            rows=[experiment_builder.fidelity_model.row_from_path(dependent_path)],
        )
        assert experiment_builder.design_matrix == expected_matrix

    def test_add_paths_bound(self, gate_set_cz):
        experiment_builder = ExperimentBuilder(gate_set_cz)

        path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
            depth=5,
        )
        mergeable_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("ZI")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZI"), QubitSparsePauli("ZI")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZI"), QubitSparsePauli("ZI")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("ZI"), QubitSparsePauli("II")
                )
            ],
            depth=5,
        )

        paths = [path, mergeable_path]
        experiment_builder.add_paths([(x, None) for x in paths])
        assert experiment_builder.paths == paths
        assert experiment_builder.instruction_sequences == [
            path.to_instruction_sequence().merge(mergeable_path.to_instruction_sequence())
        ]
        assert experiment_builder.relations == {(0, 0), (1, 0)}
        expected_matrix = IndexedMatrix()
        expected_matrix.add_rows(
            row_indices=paths,
            rows=[experiment_builder.fidelity_model.row_from_path(x) for x in paths],
        )
        assert experiment_builder.design_matrix == expected_matrix

        # manually override merging
        experiment_builder = ExperimentBuilder(gate_set_cz)
        experiment_builder.add_paths([(x, None) for x in paths], attempt_instruction_merge=False)
        assert experiment_builder.paths == [path, mergeable_path]
        assert experiment_builder.instruction_sequences == [
            path.to_instruction_sequence(),
            mergeable_path.to_instruction_sequence(),
        ]
        assert experiment_builder.relations == {(0, 0), (1, 1)}
        assert experiment_builder.design_matrix == expected_matrix

        # test linearly dependent
        path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
            depth=2,
        )
        dependent_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
            depth=1,
        )
        experiment_builder = ExperimentBuilder(gate_set_cz)
        experiment_builder.add_paths([(x, None) for x in [path, dependent_path]])
        assert experiment_builder.paths == [path]
        assert experiment_builder.instruction_sequences == [path.to_instruction_sequence()]
        assert experiment_builder.relations == {(0, 0)}
        expected_matrix = IndexedMatrix()
        expected_matrix.add_rows(
            row_indices=[path],
            rows=[experiment_builder.fidelity_model.row_from_path(path)],
        )
        assert experiment_builder.design_matrix == expected_matrix

        # manually override rank checking
        experiment_builder = ExperimentBuilder(gate_set_cz)
        experiment_builder.add_paths(
            [(x, None) for x in [path, dependent_path]],
            attempt_instruction_merge=False,
            rank_reduce=False,
        )
        assert experiment_builder.paths == [path, dependent_path]
        assert experiment_builder.instruction_sequences == [
            path.to_instruction_sequence(),
            dependent_path.to_instruction_sequence(),
        ]
        assert experiment_builder.relations == {(0, 0), (1, 1)}
        expected_matrix = IndexedMatrix()
        expected_matrix.add_rows(
            row_indices=[path, dependent_path],
            rows=[
                experiment_builder.fidelity_model.row_from_path(x) for x in [path, dependent_path]
            ],
        )
        assert experiment_builder.design_matrix == expected_matrix

    def test_complete(self, gate_set_1q):
        experiment_builder = ExperimentBuilder(gate_set_1q)

        unbound_path = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
                ),
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("Z"), QubitSparsePauli("Z")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
                )
            ],
        )
        bound_path = unbound_path.bind_at(3)

        experiment_builder.add_paths([(unbound_path, None), (bound_path, None)])
        assert not experiment_builder.is_complete

        experiment_builder.complete()

        assert experiment_builder.is_complete
        assert all(seq.is_complete for seq in experiment_builder.instruction_sequences)

    def test_identify_relations(self, gate_set_cz):
        path_IX = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["P"],
                    in_pauli=QubitSparsePauli("II"),
                    out_pauli=QubitSparsePauli("IZ"),
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["CZ"],
                    in_pauli=QubitSparsePauli("IX"),
                    out_pauli=QubitSparsePauli("ZX"),
                ),
                FidelityIndex.from_transition(
                    gate=gate_set_cz["CZ"],
                    in_pauli=QubitSparsePauli("ZX"),
                    out_pauli=QubitSparsePauli("IX"),
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["M"],
                    in_pauli=QubitSparsePauli("IZ"),
                    out_pauli=QubitSparsePauli("II"),
                )
            ],
        )

        path_XI = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["P"],
                    in_pauli=QubitSparsePauli("II"),
                    out_pauli=QubitSparsePauli("ZI"),
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["CZ"],
                    in_pauli=QubitSparsePauli("XI"),
                    out_pauli=QubitSparsePauli("XZ"),
                ),
                FidelityIndex.from_transition(
                    gate=gate_set_cz["CZ"],
                    in_pauli=QubitSparsePauli("XZ"),
                    out_pauli=QubitSparsePauli("XI"),
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["M"],
                    in_pauli=QubitSparsePauli("ZI"),
                    out_pauli=QubitSparsePauli("II"),
                )
            ],
        )
        path_XX = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["P"],
                    in_pauli=QubitSparsePauli("II"),
                    out_pauli=QubitSparsePauli("ZZ"),
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["CZ"],
                    in_pauli=QubitSparsePauli("XX"),
                    out_pauli=QubitSparsePauli("YY"),
                ),
                FidelityIndex.from_transition(
                    gate=gate_set_cz["CZ"],
                    in_pauli=QubitSparsePauli("YY"),
                    out_pauli=QubitSparsePauli("XX"),
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["M"],
                    in_pauli=QubitSparsePauli("ZZ"),
                    out_pauli=QubitSparsePauli("II"),
                )
            ],
        )

        # these could all be merged, so all possible relations should be present
        experiment_builder = ExperimentBuilder(gate_set_cz)
        experiment_builder.add_paths(
            ((x, None) for x in [path_IX, path_XI, path_XX]),
            attempt_instruction_merge=False,
        )
        experiment_builder.identify_relations()

        assert experiment_builder.relations == {(a, b) for a, b in product(range(3), range(3))}

        # without extension, none are recognized to be related due to the single qubit layers
        # between gate applications
        experiment_builder = ExperimentBuilder(gate_set_cz)
        experiment_builder.add_paths(
            ((x, None) for x in [path_IX, path_XI, path_XX]),
            attempt_instruction_merge=False,
        )
        experiment_builder.identify_relations(attempt_instruction_extension=False)
        assert experiment_builder.relations == {(0, 0), (1, 1), (2, 2)}

        # manually build instruction sequences without single qubit layers between gate applications
        # Only new relations will be IX and XI will be related to XX
        experiment_builder = ExperimentBuilder(gate_set_cz)
        instruction_sequence_IX = InstructionSequence(
            start_fragment=[
                ApplyGate(gate_set_cz["P"]),
                PartialPauliPermutation.from_sets([{("Z", "X")}, set()]),
            ],
            repeatable_fragment=[ApplyGate(gate_set_cz["CZ"])] * 2,
            end_fragment=[
                PartialPauliPermutation.from_sets([{("X", "Z")}, set()]),
                ApplyGate(gate_set_cz["M"]),
            ],
        )
        instruction_sequence_XI = InstructionSequence(
            start_fragment=[
                ApplyGate(gate_set_cz["P"]),
                PartialPauliPermutation.from_sets([set(), {("Z", "X")}]),
            ],
            repeatable_fragment=[ApplyGate(gate_set_cz["CZ"])] * 2,
            end_fragment=[
                PartialPauliPermutation.from_sets([set(), {("X", "Z")}]),
                ApplyGate(gate_set_cz["M"]),
            ],
        )
        instruction_sequence_XX = InstructionSequence(
            start_fragment=[
                ApplyGate(gate_set_cz["P"]),
                PartialPauliPermutation.from_sets([{("Z", "X")}] * 2),
            ],
            repeatable_fragment=[ApplyGate(gate_set_cz["CZ"])] * 2,
            end_fragment=[
                PartialPauliPermutation.from_sets([{("X", "Z")}] * 2),
                ApplyGate(gate_set_cz["M"]),
            ],
        )
        experiment_builder.add_paths(
            zip(
                [path_IX, path_XI, path_XX],
                [instruction_sequence_IX, instruction_sequence_XI, instruction_sequence_XX],
            ),
            attempt_instruction_merge=False,
        )
        experiment_builder.identify_relations(attempt_instruction_extension=False)
        assert experiment_builder.relations == {(0, 0), (1, 1), (2, 2), (0, 2), (1, 2)}

    def test_identify_relations_bound(self, gate_set_cz):
        path_IX = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["P"],
                    in_pauli=QubitSparsePauli("II"),
                    out_pauli=QubitSparsePauli("IZ"),
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["CZ"],
                    in_pauli=QubitSparsePauli("IX"),
                    out_pauli=QubitSparsePauli("ZX"),
                ),
                FidelityIndex.from_transition(
                    gate=gate_set_cz["CZ"],
                    in_pauli=QubitSparsePauli("ZX"),
                    out_pauli=QubitSparsePauli("IX"),
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["M"],
                    in_pauli=QubitSparsePauli("IZ"),
                    out_pauli=QubitSparsePauli("II"),
                )
            ],
            depth=2,
        )

        path_XI = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["P"],
                    in_pauli=QubitSparsePauli("II"),
                    out_pauli=QubitSparsePauli("ZI"),
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["CZ"],
                    in_pauli=QubitSparsePauli("XI"),
                    out_pauli=QubitSparsePauli("XZ"),
                ),
                FidelityIndex.from_transition(
                    gate=gate_set_cz["CZ"],
                    in_pauli=QubitSparsePauli("XZ"),
                    out_pauli=QubitSparsePauli("XI"),
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["M"],
                    in_pauli=QubitSparsePauli("ZI"),
                    out_pauli=QubitSparsePauli("II"),
                )
            ],
            depth=2,
        )
        path_XX = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["P"],
                    in_pauli=QubitSparsePauli("II"),
                    out_pauli=QubitSparsePauli("ZZ"),
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["CZ"],
                    in_pauli=QubitSparsePauli("XX"),
                    out_pauli=QubitSparsePauli("YY"),
                ),
                FidelityIndex.from_transition(
                    gate=gate_set_cz["CZ"],
                    in_pauli=QubitSparsePauli("YY"),
                    out_pauli=QubitSparsePauli("XX"),
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate=gate_set_cz["M"],
                    in_pauli=QubitSparsePauli("ZZ"),
                    out_pauli=QubitSparsePauli("II"),
                )
            ],
            depth=2,
        )

        paths = [path_IX, path_XI, path_XX]

        # these could all be merged, so all possible relations should be present
        experiment_builder = ExperimentBuilder(gate_set_cz)
        experiment_builder.add_paths([(x, None) for x in paths], attempt_instruction_merge=False)
        experiment_builder.identify_relations()
        assert experiment_builder.relations == {(a, b) for a, b in product(range(3), range(3))}

        # without extension, none are recognized to be related due to the single qubit layers
        # between gate applications
        experiment_builder = ExperimentBuilder(gate_set_cz)
        experiment_builder.add_paths([(x, None) for x in paths], attempt_instruction_merge=False)
        experiment_builder.identify_relations(attempt_instruction_extension=False)
        assert experiment_builder.relations == {(0, 0), (1, 1), (2, 2)}

        # manually build instruction sequences without single qubit layers between gate applications
        # Only new relations will be IX and XI will be related to XX
        instruction_sequences = [
            InstructionSequence(
                start_fragment=[
                    ApplyGate(gate_set_cz["P"]),
                    PartialPauliPermutation.from_sets([{("Z", "X")}, set()]),
                ],
                repeatable_fragment=[ApplyGate(gate_set_cz["CZ"])] * 2,
                end_fragment=[
                    PartialPauliPermutation.from_sets([{("X", "Z")}, set()]),
                    ApplyGate(gate_set_cz["M"]),
                ],
                depth=2,
            ),
            InstructionSequence(
                start_fragment=[
                    ApplyGate(gate_set_cz["P"]),
                    PartialPauliPermutation.from_sets([set(), {("Z", "X")}]),
                ],
                repeatable_fragment=[ApplyGate(gate_set_cz["CZ"])] * 2,
                end_fragment=[
                    PartialPauliPermutation.from_sets([set(), {("X", "Z")}]),
                    ApplyGate(gate_set_cz["M"]),
                ],
                depth=2,
            ),
            InstructionSequence(
                start_fragment=[
                    ApplyGate(gate_set_cz["P"]),
                    PartialPauliPermutation.from_sets([{("Z", "X")}] * 2),
                ],
                repeatable_fragment=[ApplyGate(gate_set_cz["CZ"])] * 2,
                end_fragment=[
                    PartialPauliPermutation.from_sets([{("X", "Z")}] * 2),
                    ApplyGate(gate_set_cz["M"]),
                ],
                depth=2,
            ),
        ]
        experiment_builder = ExperimentBuilder(gate_set_cz)
        experiment_builder.add_paths(
            zip(paths, instruction_sequences), attempt_instruction_merge=False
        )
        experiment_builder.identify_relations(attempt_instruction_extension=False)
        assert experiment_builder.relations == {(0, 0), (1, 1), (2, 2), (0, 2), (1, 2)}

    def test_merge_instruction_sequences_basic(self, gate_set_cz, unbound_path_ix, unbound_path_xi):
        """Verify two mergeable patterns are combined into one."""
        eb = ExperimentBuilder(gate_set_cz)
        eb.add_paths(
            [(unbound_path_ix, None), (unbound_path_xi, None)], attempt_instruction_merge=False
        )

        assert len(eb.instruction_sequences) == 2
        assert eb.relations == {(0, 0), (1, 1)}

        eb.merge_instruction_sequences()

        assert len(eb.instruction_sequences) == 1
        assert eb.relations == {(0, 0), (1, 0)}

        merged_pattern = eb.instruction_sequences[0]
        assert unbound_path_ix.is_traversed_by(merged_pattern)
        assert unbound_path_xi.is_traversed_by(merged_pattern)

    def test_merge_instruction_sequences_groups(self, gate_set_cz):
        """Verify patterns are grouped and merged."""
        unbound_path_ix = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IX"), QubitSparsePauli("ZX")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZX"), QubitSparsePauli("IX")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
        )

        unbound_path_xi = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("ZI")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("XI"), QubitSparsePauli("XZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("XZ"), QubitSparsePauli("XI")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("ZI"), QubitSparsePauli("II")
                )
            ],
        )

        # Create Group B: IZ and ZI (mergeable with each other, but NOT with Group A)
        unbound_path_iz = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
        )

        unbound_path_zi = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("ZI")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZI"), QubitSparsePauli("ZI")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZI"), QubitSparsePauli("ZI")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("ZI"), QubitSparsePauli("II")
                )
            ],
        )

        eb = ExperimentBuilder(gate_set_cz)
        eb.add_paths(
            [
                (unbound_path_ix, None),
                (unbound_path_xi, None),
                (unbound_path_iz, None),
                (unbound_path_zi, None),
            ],
            attempt_instruction_merge=False,
        )

        assert len(eb.instruction_sequences) == 4
        assert len(eb.relations) == 4

        eb.merge_instruction_sequences()

        assert len(eb.instruction_sequences) == 2
        assert len(eb.relations) == 4

    def test_merge_instruction_sequences_no_merge(self, gate_set_1q):
        """Verify that when all patterns are incompatible, no merging occurs."""
        pattern1 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
                ),
                FidelityIndex.from_transition(
                    gate_set_1q["L0"], QubitSparsePauli("Y"), QubitSparsePauli("Z")
                ),
                FidelityIndex.from_transition(
                    gate_set_1q["L0"], QubitSparsePauli("Z"), QubitSparsePauli("X")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
                )
            ],
        )

        pattern2 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("X"), QubitSparsePauli("X")
                ),
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("Y"), QubitSparsePauli("Y")
                ),
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("Z"), QubitSparsePauli("Z")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
                )
            ],
        )

        pattern3 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["L0"], QubitSparsePauli("Z"), QubitSparsePauli("X")
                ),
                FidelityIndex.from_transition(
                    gate_set_1q["L1"], QubitSparsePauli("X"), QubitSparsePauli("X")
                ),
                FidelityIndex.from_transition(
                    gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
                )
            ],
        )

        eb = ExperimentBuilder(gate_set_1q)
        eb.add_paths([(pattern1, None), (pattern2, None), (pattern3, None)])

        n_patterns_before = len(eb.instruction_sequences)
        relations_before = eb.relations.copy()
        eb.merge_instruction_sequences()

        assert len(eb.instruction_sequences) == n_patterns_before
        assert eb.relations == relations_before

    def test_merge_instruction_sequences_relation_remapping(self, gate_set_cz):
        """Verify that pattern relations are correctly remapped after merging."""
        pp0 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IX"), QubitSparsePauli("ZX")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZX"), QubitSparsePauli("IX")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
        )

        pp1 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IZ"), QubitSparsePauli("IZ")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
        )

        pp2 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("ZI")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("XI"), QubitSparsePauli("XZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("XZ"), QubitSparsePauli("XI")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("ZI"), QubitSparsePauli("II")
                )
            ],
        )

        eb = ExperimentBuilder(gate_set_cz)
        eb.add_paths([(pp0, None), (pp1, None), (pp2, None)], attempt_instruction_merge=False)
        eb.merge_instruction_sequences()

        assert len(eb.instruction_sequences) == 2

        pattern_to_inst = {pp_idx: ip_idx for pp_idx, ip_idx in eb.relations}
        assert len(set(pattern_to_inst.values())) == 2
        assert set(pattern_to_inst.keys()) == {0, 1, 2}

    def test_merge_instruction_sequences_single(self, gate_set_cz, unbound_path_ix):
        """Verify behavior with only one instruction pattern."""
        eb = ExperimentBuilder(gate_set_cz)
        eb.add_paths([(unbound_path_ix, None)])

        assert len(eb.instruction_sequences) == 1
        original_relations = eb.relations.copy()

        eb.merge_instruction_sequences()

        assert len(eb.instruction_sequences) == 1
        assert eb.relations == original_relations
        assert unbound_path_ix.is_traversed_by(eb.instruction_sequences[0])

    def test_merge_instruction_sequences_three_way_merge(self, gate_set_cz):
        """Verify that three patterns can be merged into one when all are mutually compatible."""
        pp0 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IX"), QubitSparsePauli("ZX")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZX"), QubitSparsePauli("IX")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
        )

        pp1 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("ZI")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("XI"), QubitSparsePauli("XZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("XZ"), QubitSparsePauli("XI")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("ZI"), QubitSparsePauli("II")
                )
            ],
        )

        pp2 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("ZZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("XX"), QubitSparsePauli("YY")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("YY"), QubitSparsePauli("XX")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("ZZ"), QubitSparsePauli("II")
                )
            ],
        )

        eb = ExperimentBuilder(gate_set_cz)
        eb.add_paths([(pp0, None), (pp1, None), (pp2, None)], attempt_instruction_merge=False)

        assert len(eb.instruction_sequences) == 3

        eb.merge_instruction_sequences()

        assert len(eb.instruction_sequences) == 1
        assert eb.relations == {(0, 0), (1, 0), (2, 0)}

        merged = eb.instruction_sequences[0]
        assert pp0.is_traversed_by(merged)
        assert pp1.is_traversed_by(merged)
        assert pp2.is_traversed_by(merged)

    def test_merge_instruction_sequences_traversal(
        self, gate_set_cz, unbound_path_ix, unbound_path_xi
    ):
        """Verify the merged instruction sequences can still traverse all original paths."""
        eb = ExperimentBuilder(gate_set_cz)
        eb.add_paths(
            [(unbound_path_ix, None), (unbound_path_xi, None)], attempt_instruction_merge=False
        )
        original_paths = eb.paths.copy()
        eb.merge_instruction_sequences()

        for path_idx, inst_idx in eb.relations:
            path = original_paths[path_idx]
            instruction_sequence = eb.instruction_sequences[inst_idx]
            assert path.is_traversed_by(
                instruction_sequence
            ), f"Path {path_idx} cannot be traversed by instruction sequence {inst_idx}"

    def test_merge_instruction_sequences_existing_relations(self, gate_set_cz):
        """Verify that merging works correctly when patterns already have multiple relations."""
        # Create path patterns
        pp0 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("IZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("IX"), QubitSparsePauli("ZX")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("ZX"), QubitSparsePauli("IX")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("IZ"), QubitSparsePauli("II")
                )
            ],
        )

        pp1 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("ZI")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("XI"), QubitSparsePauli("XZ")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("XZ"), QubitSparsePauli("XI")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("ZI"), QubitSparsePauli("II")
                )
            ],
        )

        pp2 = Path(
            start_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["P"], QubitSparsePauli("II"), QubitSparsePauli("ZZ")
                )
            ],
            repeatable_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("YY"), QubitSparsePauli("XX")
                ),
                FidelityIndex.from_transition(
                    gate_set_cz["CZ"], QubitSparsePauli("XX"), QubitSparsePauli("YY")
                ),
            ],
            end_fragment=[
                FidelityIndex.from_transition(
                    gate_set_cz["M"], QubitSparsePauli("ZZ"), QubitSparsePauli("II")
                )
            ],
        )

        eb = ExperimentBuilder(gate_set_cz)

        eb.add_paths([(pp0, None), (pp1, None), (pp2, None)], attempt_instruction_merge=False)
        eb.identify_relations()
        eb.rank_reduce()

        assert len(eb.instruction_sequences) == 3
        assert len(eb.relations) == 5

        eb.merge_instruction_sequences()

        assert len(eb.instruction_sequences) == 2
        assert len(eb.relations) == 3

        path_indices = {path_idx for path_idx, _ in eb.relations}
        assert path_indices == {0, 1, 2}

    def test_build(self, gate_set_cz, unbound_path_ix, unbound_path_xi):
        eb = ExperimentBuilder(gate_set_cz)
        eb.add_paths([(unbound_path_ix, None), (unbound_path_xi, None)])

        fixed_path = unbound_path_ix.bind_at(7)
        eb.add_paths([(fixed_path, None)])

        depths = [2, 4]
        shots = 1024
        experiment = eb.build(depths=depths, shots=shots)

        assert isinstance(experiment, Experiment)
        assert experiment.shots == shots
        assert experiment.fidelity_model is eb.fidelity_model

        expected_sequences = []
        for seq in eb.instruction_sequences:
            if seq.is_unbound:
                expected_sequences.extend(seq.bind_at(d).complete() for d in depths)
            else:
                expected_sequences.append(seq.complete())
        assert len(experiment.sequences) == len(expected_sequences)
        for seq in expected_sequences:
            assert seq in experiment.sequences

        expected_paths = set()
        for path in eb.paths:
            if path.is_unbound:
                expected_paths.update(path.bind_at(d) for d in depths)
            else:
                expected_paths.add(path)
        assert set(experiment.paths) == expected_paths


class TestMinimizeInstructionSequences:
    """Tests the ``minimize_instruction_sequences`` function."""

    def test_basic(self):
        """Test a basic use case."""
        patterns = []
        for perm in [("Z", "Z"), ("X", "Z"), ("Y", "Z")]:
            partial_perm = PartialPauliPermutation.from_sets([{perm}])
            patterns.append(InstructionSequence([], [], [partial_perm]))

        minimized_patterns, _ = minimize_instruction_sequences(patterns)
        assert len(minimized_patterns) == 3

        patterns.append(
            InstructionSequence([], [], [PartialPauliPermutation.from_sets([{("Z", "Z")}])])
        )
        minimized_patterns, reorg_dict = minimize_instruction_sequences(patterns)
        assert len(minimized_patterns) == 3
        assert reorg_dict[0] == reorg_dict[3]

    def test_single_qubit_basis_patterns(self):
        """Test that the optimal number of single-qubit patterns is found.."""
        patterns = []
        for basis in "IZXY":
            for i in range(num_qubits := 10):
                in_pauli = QubitSparsePauli.from_sparse_label(("Z", (i,)), num_qubits=num_qubits)
                out_pauli = QubitSparsePauli.from_sparse_label((basis, (i,)), num_qubits=num_qubits)
                partial_perm = PartialPauliPermutation.from_qubit_sparse_paulis(in_pauli, out_pauli)
                patterns.append(InstructionSequence([], [], [partial_perm]))

        minimized_patterns, _ = minimize_instruction_sequences(patterns)
        assert len(minimized_patterns) == 3

    def test_two_qubit_basis_patterns(self):
        """Test that the optimal number of two-qubit patterns is found on a ring."""
        patterns = []
        num_qubits = 10
        for basis_0, basis_1 in product("IXYZ", repeat=2):
            in_basis = basis_0 + basis_1
            out_basis = "".join("Z" if b != "I" else b for b in in_basis)
            for idx in range(num_qubits):
                subs = (idx, (idx + 1) % num_qubits)
                in_pauli = QubitSparsePauli.from_sparse_label(
                    (in_basis, subs), num_qubits=num_qubits
                )
                out_pauli = QubitSparsePauli.from_sparse_label(
                    (out_basis, subs), num_qubits=num_qubits
                )
                partial_perm = PartialPauliPermutation.from_qubit_sparse_paulis(in_pauli, out_pauli)
                patterns.append(InstructionSequence([], [], [partial_perm]))

        minimized_patterns, _ = minimize_instruction_sequences(patterns)
        assert len(minimized_patterns) == 9
