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

from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from qiskit_noise_learning.analysis import (
    AnalysisPipeline,
    AnalysisStage,
    Fit,
)
from qiskit_noise_learning.analysis.fit import AbsentType, SkippedType
from qiskit_noise_learning.data import AveragedData, ModelData, ObservableData, RawData


@dataclass(frozen=True)
class MockPathPattern:
    name: str


@dataclass(frozen=True)
class MockPath:
    pattern: MockPathPattern
    depth: int


class MockFidelityModel:
    def __init__(self, rows: dict):
        self._rows = rows

    def multiplicative_row_from_path_pattern(self, path_pattern):
        return self._rows[path_pattern]


class _StubRawToObs(AnalysisStage):
    @property
    def input_level(self):
        return RawData

    @property
    def output_level(self):
        return ObservableData

    def _run(self, fit):
        fit[ObservableData] = ObservableData.from_arrays(
            path_patterns=[],
            depths=[],
            observables=np.empty((0, 0)),
            time_lbs=np.empty((0, 0), dtype="datetime64[us]"),
            time_ubs=np.empty((0, 0), dtype="datetime64[us]"),
        )


class _StubObsToAveraged(AnalysisStage):
    @property
    def input_level(self):
        return ObservableData

    @property
    def output_level(self):
        return AveragedData

    def _run(self, fit):
        fit[AveragedData] = AveragedData.from_arrays(
            path_patterns=[],
            depths=[],
            observables=np.array([]),
            std=np.array([]),
            time_lbs=np.empty(0, dtype="datetime64[us]"),
            time_ubs=np.empty(0, dtype="datetime64[us]"),
        )


class _StubAveragedToModel(AnalysisStage):
    @property
    def input_level(self):
        return AveragedData

    @property
    def output_level(self):
        return ModelData

    def _run(self, fit):
        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=[],
            parameter_values=np.array([]),
            covariance=np.empty((0, 0)),
            time_lbs=np.empty(0, dtype="datetime64[us]"),
            time_ubs=np.empty(0, dtype="datetime64[us]"),
        )


class _StubRawToAveraged(AnalysisStage):
    @property
    def input_level(self):
        return RawData

    @property
    def output_level(self):
        return AveragedData

    def _run(self, fit):
        fit[AveragedData] = AveragedData.from_arrays(
            path_patterns=[],
            depths=[],
            observables=np.array([]),
            std=np.array([]),
            time_lbs=np.empty(0, dtype="datetime64[us]"),
            time_ubs=np.empty(0, dtype="datetime64[us]"),
        )


class TestAnalysisPipeline:
    """Test initialization of an analysis pipeline."""

    def test_single_stage_pipeline(self):
        """Test initializing a pipeline."""
        pipeline = AnalysisPipeline(_StubRawToObs())
        assert pipeline.input_level is RawData
        assert pipeline.output_level is ObservableData

    def test_stages_property(self):
        """Test the stages property."""
        stages = (_StubRawToObs(), _StubObsToAveraged(), _StubAveragedToModel())
        pipeline = AnalysisPipeline(*stages)

        assert pipeline.input_level is RawData
        assert pipeline.output_level is ModelData
        assert pipeline.stages == stages

    def test_incompatible_stages_raises(self):
        """Test that the output of a level must match the input of the subsequent level."""
        with pytest.raises(ValueError, match="does not match"):
            AnalysisPipeline(_StubObsToAveraged(), _StubRawToObs())

    def test_skipped_and_absent_marking(self):
        """Test skipped and absent levels are marked accordingly."""
        stage = _StubRawToAveraged()
        raw = RawData(datatree=xr.DataTree())
        fit = Fit()
        fit[RawData] = raw
        result = stage.run(fit)

        assert result[RawData] is raw
        assert isinstance(result[ObservableData], SkippedType)
        assert isinstance(result[AveragedData], AveragedData)
        assert isinstance(result[ModelData], AbsentType)

    def test_pipeline_run_does_not_mutate_original(self):
        """Test that returns a new fit and does not mutate the input fit."""
        pipeline = AnalysisPipeline(_StubObsToAveraged(), _StubAveragedToModel())
        obs = ObservableData.from_arrays(
            path_patterns=[],
            depths=[],
            observables=np.empty((0, 0)),
            time_lbs=np.empty((0, 0), dtype="datetime64[us]"),
            time_ubs=np.empty((0, 0), dtype="datetime64[us]"),
        )
        fit = Fit()
        fit[ObservableData] = obs
        new_fit = pipeline.run(fit)

        assert isinstance(fit[AveragedData], AbsentType)
        assert isinstance(fit[ModelData], AbsentType)
        assert isinstance(new_fit[AveragedData], AveragedData)
        assert isinstance(new_fit[ModelData], ModelData)

    def test_history_through_pipeline(self):
        """Test that the data history is set appropriately."""
        pipeline = AnalysisPipeline(_StubObsToAveraged(), _StubAveragedToModel())
        obs = ObservableData.from_arrays(
            path_patterns=[],
            depths=[],
            observables=np.empty((0, 0)),
            time_lbs=np.empty((0, 0), dtype="datetime64[us]"),
            time_ubs=np.empty((0, 0), dtype="datetime64[us]"),
        )
        fit = Fit()
        fit[ObservableData] = obs
        result = pipeline.run(fit)

        averaged_hist = result.history.averaged_data
        assert isinstance(averaged_hist[0], AbsentType)
        assert isinstance(averaged_hist[1], SkippedType)
        assert isinstance(averaged_hist[-1], AveragedData)

        model_hist = result.history.model_data
        assert isinstance(model_hist[0], AbsentType)
        assert isinstance(model_hist[1], AbsentType)
        assert isinstance(model_hist[-1], ModelData)

    def test_history_preserves_input_data(self):
        """Test that the history contains the input data."""
        pipeline = AnalysisPipeline(_StubObsToAveraged(), _StubAveragedToModel())
        obs = ObservableData.from_arrays(
            path_patterns=[],
            depths=[],
            observables=np.empty((0, 0)),
            time_lbs=np.empty((0, 0), dtype="datetime64[us]"),
            time_ubs=np.empty((0, 0), dtype="datetime64[us]"),
        )
        fit = Fit()
        fit[ObservableData] = obs
        result = pipeline.run(fit)

        obs_hist = result.history.observable_data
        assert obs in obs_hist

    def test_nested_pipeline_levels(self):
        """Test properties for nested pipelines (constructor flattens)."""
        inner = AnalysisPipeline(_StubRawToObs(), _StubObsToAveraged())
        outer = AnalysisPipeline(inner, _StubAveragedToModel())
        assert outer.input_level is RawData
        assert outer.output_level is ModelData
        assert len(outer.stages) == 3  # flattened

    def test_nested_pipeline_run(self):
        """Test the run method for nested pipelines."""
        inner = AnalysisPipeline(_StubRawToObs(), _StubObsToAveraged())
        outer = AnalysisPipeline(inner, _StubAveragedToModel())
        raw = RawData(datatree=xr.DataTree())
        fit = Fit()
        fit[RawData] = raw
        result = outer.run(fit)

        assert isinstance(result[ModelData], ModelData)
        assert isinstance(result[AveragedData], AveragedData)

    def test_deeply_nested_pipeline(self):
        """Test a deeply nested pipeline (constructor flattens at each level)."""
        p1 = AnalysisPipeline(_StubRawToObs())
        p2 = AnalysisPipeline(p1, _StubObsToAveraged())
        p3 = AnalysisPipeline(p2, _StubAveragedToModel())
        assert p3.input_level is RawData
        assert p3.output_level is ModelData
        assert len(p3.stages) == 3  # flattened

        raw = RawData(datatree=xr.DataTree())
        fit = Fit()
        fit[RawData] = raw
        result = p3.run(fit)
        assert isinstance(result[ModelData], ModelData)

    def test_add_creates_pipeline(self):
        """Test that + on two stages returns an AnalysisPipeline."""
        stage1, stage2 = _StubRawToObs(), _StubObsToAveraged()
        result = stage1 + stage2
        assert isinstance(result, AnalysisPipeline)
        assert result.stages == (stage1, stage2)

    def test_add_flattens_pipelines(self):
        """Test that + flattens when one operand is a pipeline."""
        s1, s2, s3 = _StubRawToObs(), _StubObsToAveraged(), _StubAveragedToModel()
        pipeline = AnalysisPipeline(s1, s2)
        result = pipeline + s3
        assert result.stages == (s1, s2, s3)

        result2 = s1 + AnalysisPipeline(s2, s3)
        assert result2.stages == (s1, s2, s3)

    def test_add_incompatible_raises(self):
        """Test that + on incompatible stages raises ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            _StubObsToAveraged() + _StubRawToObs()

    def test_iadd_appends(self):
        """Test that += appends a stage to the pipeline."""
        s1, s2, s3 = _StubRawToObs(), _StubObsToAveraged(), _StubAveragedToModel()
        pipeline = AnalysisPipeline(s1, s2)
        pipeline += s3
        assert pipeline.stages == (s1, s2, s3)
        assert pipeline.output_level is ModelData

    def test_iadd_flattens(self):
        """Test that += flattens when appending a pipeline."""
        s1, s2, s3 = _StubRawToObs(), _StubObsToAveraged(), _StubAveragedToModel()
        pipeline = AnalysisPipeline(s1)
        pipeline += AnalysisPipeline(s2, s3)
        assert pipeline.stages == (s1, s2, s3)

    def test_iadd_incompatible_raises(self):
        """Test that += raises ValueError for incompatible stages."""
        pipeline = AnalysisPipeline(_StubRawToObs())
        with pytest.raises(ValueError, match="does not match"):
            pipeline += _StubRawToObs()

    def test_repr_stage(self):
        """Test the repr of a bare stage."""
        assert repr(_StubRawToObs()) == "_StubRawToObs()"

    def test_repr_pipeline(self):
        """Test the repr of a pipeline."""
        s1, s2 = _StubRawToObs(), _StubObsToAveraged()
        pipeline = AnalysisPipeline(s1, s2)
        assert repr(pipeline) == "AnalysisPipeline(_StubRawToObs(), _StubObsToAveraged())"

    def test_pipeline_run_with_leveled_data(self):
        """Test that the run method on leveled data returns a fit with the right data."""
        pipeline = AnalysisPipeline(_StubObsToAveraged(), _StubAveragedToModel())
        obs = ObservableData.from_arrays(
            path_patterns=[],
            depths=[],
            observables=np.empty((0, 0)),
            time_lbs=np.empty((0, 0), dtype="datetime64[us]"),
            time_ubs=np.empty((0, 0), dtype="datetime64[us]"),
        )
        result = pipeline.run(obs)

        assert isinstance(result, Fit)
        assert isinstance(result[ModelData], ModelData)
