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
from qiskit_noise_learning.experiment_builder.stage import ExperimentBuilder, ExperimentBuilderStage


class _SetShots(ExperimentBuilderStage):
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


class TestExperimentBuilderStage:
    """Tests for the ExperimentBuilderStage base class."""

    def test_run_calls_subclass(self):
        stage = _SetShots(42)
        exp = Experiment()
        result = stage.run(exp)
        assert result.shots == 42

    def test_required_fields_validation_passes(self, unbound_path_ix):
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
        s1 = _SetShots(10)
        s2 = _SetShots(20)
        builder = s1 + s2
        assert isinstance(builder, ExperimentBuilder)
        assert builder.stages == (s1, s2)

    def test_add_non_stage_returns_not_implemented(self):
        s = _SetShots(10)
        assert s.__add__(42) is NotImplemented

    def test_repr(self):
        stage = _SetShots(10)
        assert repr(stage) == "_SetShots()"


class TestExperimentBuilder:
    """Tests for the ExperimentBuilder composite stage."""

    def test_chains_stages_sequentially(self):
        builder = ExperimentBuilder(_SetShots(10), _SetShots(20))
        result = builder.run(Experiment())
        assert result.shots == 20

    def test_flattens_nested_builders(self):
        inner = ExperimentBuilder(_SetShots(10), _SetShots(20))
        outer = ExperimentBuilder(inner, _SetShots(30))
        assert len(outer.stages) == 3
        result = outer.run(Experiment())
        assert result.shots == 30

    def test_add_builder_to_stage(self):
        builder = ExperimentBuilder(_SetShots(10))
        combined = builder + _SetShots(20)
        assert isinstance(combined, ExperimentBuilder)
        assert len(combined.stages) == 2

    def test_required_fields_in_chain(self, unbound_path_ix):
        builder = ExperimentBuilder(_RequiresPaths(), _SetShots(10))
        with pytest.raises(ValueError, match="requires 'paths'"):
            builder.run(Experiment())

        result = builder.run(Experiment(paths=[unbound_path_ix]))
        assert result.shots == 10

    def test_repr(self):
        builder = ExperimentBuilder(_SetShots(10), _SetShots(20))
        assert repr(builder) == "ExperimentBuilder(_SetShots(), _SetShots())"
