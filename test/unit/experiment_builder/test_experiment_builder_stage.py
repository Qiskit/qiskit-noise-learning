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
from qiskit_noise_learning.experiment_builder.experiment_builder_stage import (
    ExperimentBuilder,
    ExperimentBuilderStage,
)


class _SimpleStage(ExperimentBuilderStage):
    """Minimal concrete stage for testing."""

    def __init__(self, shots):
        self._shots = shots

    def _run(self, experiment):
        return experiment.replace(shots=self._shots)


class _RequiresPaths(ExperimentBuilderStage):
    """Stage that requires paths to be set."""

    required_fields = ("paths",)

    def _run(self, experiment):
        return experiment


class _PopulatesPaths(ExperimentBuilderStage):
    """Stage that populates paths."""

    populates_fields = ("paths",)

    def _run(self, experiment):
        return experiment.replace(validate=False, paths=[])


class _RequiresPathsPopulatesSequences(ExperimentBuilderStage):
    """Stage that requires paths and populates instruction_sequences."""

    required_fields = ("paths",)
    populates_fields = ("instruction_sequences", "randomization_multipliers")

    def _run(self, experiment):
        return experiment.replace(
            validate=False, instruction_sequences=[], randomization_multipliers=[]
        )


class TestExperimentBuilderStage:
    """Tests for the ExperimentBuilderStage base class."""

    def test_run_calls_subclass(self):
        stage = _SimpleStage(42)
        exp = Experiment()
        result = stage.run(exp)
        assert result.shots == 42

    def test_required_fields_validation_passes(self, make_cz_path):
        unbound_path_ix = make_cz_path("IX")
        stage = _RequiresPaths()
        exp = Experiment(paths=[unbound_path_ix])
        result = stage.run(exp)
        assert result.paths == [unbound_path_ix]

    def test_required_fields_validation_raises(self):
        stage = _RequiresPaths()
        exp = Experiment()
        with pytest.raises(ValueError, match="requires 'paths' to be set"):
            stage.run(exp)

    def test_add_stages_produces_builder(self):
        s1 = _SimpleStage(10)
        s2 = _SimpleStage(20)
        builder = s1 + s2
        assert isinstance(builder, ExperimentBuilder)
        assert builder.stages == (s1, s2)

    def test_add_non_stage_returns_not_implemented(self):
        s = _SimpleStage(10)
        assert s.__add__(42) is NotImplemented

    def test_repr(self):
        stage = _SimpleStage(10)
        assert repr(stage) == "_SimpleStage()"


class TestExperimentBuilder:
    """Tests for the ExperimentBuilder composite stage."""

    def test_chains_stages_sequentially(self):
        builder = ExperimentBuilder(_SimpleStage(10), _SimpleStage(20))
        result = builder.run(Experiment())
        assert result.shots == 20

    def test_flattens_nested_builders(self):
        inner = ExperimentBuilder(_SimpleStage(10), _SimpleStage(20))
        outer = ExperimentBuilder(inner, _SimpleStage(30))
        assert len(outer.stages) == 3
        result = outer.run(Experiment())
        assert result.shots == 30

    def test_add_builder_to_stage(self):
        builder = ExperimentBuilder(_SimpleStage(10))
        combined = builder + _SimpleStage(20)
        assert isinstance(combined, ExperimentBuilder)
        assert len(combined.stages) == 2

    def test_required_fields_in_chain(self, make_cz_path):
        builder = ExperimentBuilder(_RequiresPaths(), _SimpleStage(10))
        with pytest.raises(ValueError, match="requires 'paths'"):
            builder.run(Experiment())

        result = builder.run(Experiment(paths=[make_cz_path("IX")]))
        assert result.shots == 10

    def test_repr(self):
        builder = ExperimentBuilder(_SimpleStage(10), _SimpleStage(20))
        assert repr(builder) == "ExperimentBuilder(_SimpleStage(), _SimpleStage())"

    def test_required_fields_excludes_populated_by_prior_stage(self):
        builder = ExperimentBuilder(_PopulatesPaths(), _RequiresPaths())
        assert builder.required_fields == ()

    def test_required_fields_includes_unsatisfied(self):
        builder = ExperimentBuilder(_RequiresPaths(), _SimpleStage(10))
        assert builder.required_fields == ("paths",)

    def test_required_fields_multiple_stages(self):
        builder = ExperimentBuilder(
            _PopulatesPaths(), _RequiresPathsPopulatesSequences(), _RequiresPaths()
        )
        assert builder.required_fields == ()

    def test_required_fields_order_matters(self):
        builder = ExperimentBuilder(_RequiresPathsPopulatesSequences(), _PopulatesPaths())
        assert builder.required_fields == ("paths",)

    def test_populates_fields_union(self):
        builder = ExperimentBuilder(_PopulatesPaths(), _RequiresPathsPopulatesSequences())
        assert set(builder.populates_fields) == {
            "paths",
            "instruction_sequences",
            "randomization_multipliers",
        }

    def test_nested_builder_required_fields(self):
        inner = ExperimentBuilder(_PopulatesPaths(), _RequiresPathsPopulatesSequences())
        outer = ExperimentBuilder(inner, _RequiresPaths())
        assert outer.required_fields == ()

    def test_nested_builder_populates_fields(self):
        inner = ExperimentBuilder(_PopulatesPaths())
        outer = ExperimentBuilder(inner, _RequiresPathsPopulatesSequences())
        assert set(outer.populates_fields) == {
            "paths",
            "instruction_sequences",
            "randomization_multipliers",
        }

    def test_run_raises_early_for_unsatisfied_fields(self):
        builder = ExperimentBuilder(_RequiresPathsPopulatesSequences(), _SimpleStage(10))
        with pytest.raises(ValueError, match="ExperimentBuilder requires 'paths'"):
            builder.run(Experiment())
