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

import pytest

from qiskit_noise_learning.experiment_builder.experiment import Experiment
from qiskit_noise_learning.math import IndexedMatrix
from qiskit_noise_learning.models import CompleteFidelityModel


class TestExperimentConstruction:
    """Tests for Experiment construction and property access."""

    def test_empty_construction(self):
        exp = Experiment()
        assert exp.fidelity_model is None
        assert exp.gate_set is None
        assert exp.paths is None
        assert exp.instruction_sequences is None
        assert exp.relations is None
        assert exp.shots == 20
        assert exp.randomizations == 50
        assert exp.randomization_multipliers is None

    def test_construction_with_gate_set(self, gate_set_cz):
        exp = Experiment(fidelity_model=gate_set_cz)
        assert isinstance(exp.fidelity_model, CompleteFidelityModel)
        assert exp.gate_set == gate_set_cz

    def test_construction_with_fidelity_model(self, gate_set_cz):
        model = CompleteFidelityModel(gate_set_cz)
        exp = Experiment(fidelity_model=model)
        assert exp.fidelity_model is model
        assert exp.gate_set == gate_set_cz

    def test_missing_multipliers(self, gate_set_cz, unbound_path_ix):
        seq = unbound_path_ix.to_instruction_sequence().bind_at(2).complete()
        with pytest.raises(ValueError, match="must both be None or both be non-None"):
            Experiment(
                fidelity_model=gate_set_cz,
                paths=[unbound_path_ix],
                instruction_sequences=[seq],
                shots=100,
                randomizations=50,
            )

    def test_construction_with_all_fields(self, gate_set_cz, unbound_path_ix):
        seq = unbound_path_ix.to_instruction_sequence()
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix],
            instruction_sequences=[seq],
            relations={(0, 0)},
            shots=100,
            randomizations=50,
            randomization_multipliers=[1],
        )
        assert exp.paths == [unbound_path_ix]
        assert exp.instruction_sequences == [seq]
        assert exp.relations == {(0, 0)}
        assert exp.shots == 100
        assert exp.randomizations == 50
        assert exp.randomization_multipliers == [1]


class TestExperimentDesignMatrix:
    """Tests for the design_matrix property."""

    def test_design_matrix_computed(self, gate_set_cz, unbound_path_ix, unbound_path_xi):
        exp = Experiment(fidelity_model=gate_set_cz, paths=[unbound_path_ix, unbound_path_xi])
        model = CompleteFidelityModel(gate_set_cz)

        expected = IndexedMatrix()
        expected.add_rows(
            row_indices=[unbound_path_ix, unbound_path_xi],
            rows=[model.row_from_path(unbound_path_ix), model.row_from_path(unbound_path_xi)],
        )
        assert exp.design_matrix == expected

    def test_design_matrix_raises_without_model(self, unbound_path_ix):
        exp = Experiment(paths=[unbound_path_ix])
        with pytest.raises(ValueError, match="fidelity_model is None"):
            _ = exp.design_matrix

    def test_design_matrix_raises_without_paths(self, gate_set_cz):
        exp = Experiment(fidelity_model=gate_set_cz)
        with pytest.raises(ValueError, match="paths is None"):
            _ = exp.design_matrix

    def test_design_matrix_cached(self, gate_set_cz, unbound_path_ix):
        exp = Experiment(fidelity_model=gate_set_cz, paths=[unbound_path_ix])
        dm1 = exp.design_matrix
        dm2 = exp.design_matrix
        assert dm1 is dm2


class TestExperimentIsExecutable:
    """Tests for the is_executable property."""

    def test_not_executable_no_sequences(self, gate_set_cz, unbound_path_ix):
        exp = Experiment(fidelity_model=gate_set_cz, paths=[unbound_path_ix])
        assert not exp.is_executable

    def test_not_executable_unbound_sequences(self, gate_set_cz, unbound_path_ix):
        seq = unbound_path_ix.to_instruction_sequence()
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix],
            instruction_sequences=[seq],
            shots=100,
            randomizations=50,
            randomization_multipliers=[1],
        )
        assert not exp.is_executable

    def test_executable(self, gate_set_cz, unbound_path_ix):
        seq = unbound_path_ix.to_instruction_sequence().bind_at(2).complete()
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix],
            instruction_sequences=[seq],
            shots=100,
            randomizations=50,
            randomization_multipliers=[1],
        )
        assert exp.is_executable


class TestExperimentReplace:
    """Tests for the replace() method."""

    def test_basic_replace(self, gate_set_cz, unbound_path_ix):
        exp = Experiment(fidelity_model=gate_set_cz)
        new_exp = exp.replace(shots=100)
        assert new_exp.shots == 100
        assert exp.shots == 20

    def test_replace_unknown_field(self, gate_set_cz):
        exp = Experiment(fidelity_model=gate_set_cz)
        with pytest.raises(TypeError, match="no field 'unknown'"):
            exp.replace(unknown=42)

    def test_replace_co_replacement_violation(self, gate_set_cz, unbound_path_ix):
        seq = unbound_path_ix.to_instruction_sequence()
        exp = Experiment(fidelity_model=gate_set_cz)
        with pytest.raises(ValueError, match="both be None or both be non-None"):
            exp.replace(instruction_sequences=[seq])

    def test_replace_length_consistency(self, gate_set_cz, unbound_path_ix):
        seq = unbound_path_ix.to_instruction_sequence()
        exp = Experiment(fidelity_model=gate_set_cz)
        with pytest.raises(ValueError, match="does not match"):
            exp.replace(instruction_sequences=[seq], randomization_multipliers=[1, 2])

    def test_replace_relations_bounds_check(self, gate_set_cz, unbound_path_ix):
        seq = unbound_path_ix.to_instruction_sequence()
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix],
            instruction_sequences=[seq],
            randomization_multipliers=[1],
        )
        with pytest.raises(ValueError, match="out of bounds"):
            exp.replace(relations={(5, 0)})

    def test_replace_soft_invalidation(self, gate_set_cz, unbound_path_ix, unbound_path_xi):
        seq_ix = unbound_path_ix.to_instruction_sequence()
        seq_xi = unbound_path_xi.to_instruction_sequence()
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix, unbound_path_xi],
            instruction_sequences=[seq_ix, seq_xi],
            relations={(0, 0), (1, 1)},
            randomization_multipliers=[1, 1],
        )
        with pytest.warns(UserWarning, match="invalidates relations"):
            new_exp = exp.replace(paths=[unbound_path_ix])
        assert new_exp.relations is None

    def test_replace_no_validate(self, gate_set_cz, unbound_path_ix):
        seq = unbound_path_ix.to_instruction_sequence()
        exp = Experiment(fidelity_model=gate_set_cz)
        new_exp = exp.replace(validate=False, instruction_sequences=[seq])
        assert new_exp.instruction_sequences == [seq]
        assert new_exp.randomization_multipliers is None

    def test_replace_invalidates_design_matrix_cache(
        self, gate_set_cz, unbound_path_ix, unbound_path_xi
    ):
        exp = Experiment(fidelity_model=gate_set_cz, paths=[unbound_path_ix, unbound_path_xi])
        _ = exp.design_matrix
        new_exp = exp.replace(validate=False, paths=[unbound_path_ix])
        assert new_exp._design_matrix_cache is None  # noqa: SLF001


class TestExperimentAdd:
    """Tests for the __add__() method."""

    def test_add_concatenates_lists(self, gate_set_cz, unbound_path_ix, unbound_path_xi):
        model = CompleteFidelityModel(gate_set_cz)
        seq_ix = unbound_path_ix.to_instruction_sequence()
        seq_xi = unbound_path_xi.to_instruction_sequence()
        exp1 = Experiment(
            fidelity_model=model,
            paths=[unbound_path_ix],
            instruction_sequences=[seq_ix],
            relations={(0, 0)},
            shots=100,
            randomizations=50,
            randomization_multipliers=[1],
        )
        exp2 = Experiment(
            fidelity_model=model,
            paths=[unbound_path_xi],
            instruction_sequences=[seq_xi],
            relations={(0, 0)},
            shots=100,
            randomizations=50,
            randomization_multipliers=[2],
        )
        result = exp1 + exp2
        assert result.paths == [unbound_path_ix, unbound_path_xi]
        assert result.instruction_sequences == [seq_ix, seq_xi]
        assert result.relations == {(0, 0), (1, 1)}
        assert result.randomization_multipliers == [1, 2]

    def test_add_scalar_mismatch_raises(self, gate_set_cz):
        model = CompleteFidelityModel(gate_set_cz)
        exp1 = Experiment(fidelity_model=model, shots=100)
        exp2 = Experiment(fidelity_model=model, shots=200)
        with pytest.raises(ValueError, match="does not match"):
            _ = exp1 + exp2

    def test_add_none_lists(self, gate_set_cz):
        model = CompleteFidelityModel(gate_set_cz)
        exp1 = Experiment(fidelity_model=model)
        exp2 = Experiment(fidelity_model=model)
        result = exp1 + exp2
        assert result.paths is None
        assert result.instruction_sequences is None

    def test_add_one_none_one_list(self, gate_set_cz, unbound_path_ix):
        model = CompleteFidelityModel(gate_set_cz)
        exp1 = Experiment(fidelity_model=model, paths=[unbound_path_ix])
        exp2 = Experiment(fidelity_model=model)
        result = exp1 + exp2
        assert result.paths == [unbound_path_ix]

    def test_add_relation_offsetting(self, gate_set_cz, unbound_path_ix, unbound_path_xi):
        model = CompleteFidelityModel(gate_set_cz)
        seq_ix = unbound_path_ix.to_instruction_sequence()
        seq_xi = unbound_path_xi.to_instruction_sequence()
        exp1 = Experiment(
            fidelity_model=model,
            paths=[unbound_path_ix, unbound_path_xi],
            instruction_sequences=[seq_ix, seq_xi],
            relations={(0, 0), (1, 1)},
            randomization_multipliers=[1, 1],
        )
        exp2 = Experiment(
            fidelity_model=model,
            paths=[unbound_path_ix],
            instruction_sequences=[seq_ix],
            relations={(0, 0)},
            randomization_multipliers=[1],
        )
        result = exp1 + exp2
        assert result.relations == {(0, 0), (1, 1), (2, 2)}

    def test_add_not_implemented_for_non_experiment(self, gate_set_cz):
        exp = Experiment(fidelity_model=gate_set_cz)
        assert exp.__add__(42) is NotImplemented
