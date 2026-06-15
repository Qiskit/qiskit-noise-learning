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

"""Tests for the bootstrap analysis stage and resamplers."""

from dataclasses import dataclass

import numpy as np
import pytest

from qiskit_noise_learning.analysis import (
    AnalysisPipeline,
    ArchResampler,
    AverageObservables,
    Bootstrap,
    CurveFitObservables,
    Fit,
    NNLSSolve,
    NumpyResampler,
    ScipyResampler,
)
from qiskit_noise_learning.analysis.bootstrap import _row_valid_indices
from qiskit_noise_learning.data import AveragedData, ModelData, ObservableData
from qiskit_noise_learning.math import IndexedVector


@dataclass(frozen=True)
class MockUnboundPath:
    name: str


class MockFidelityModel:
    def __init__(self, rows: dict):
        self._rows = rows

    def row_from_path(self, unbound_path):
        return self._rows[unbound_path]


def _make_observable_data(
    pattern_depths: dict[str, list[int]],
    a_values: dict[str, float],
    f_values: dict[str, float],
    n_rand: int = 30,
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple[ObservableData, dict[str, MockUnboundPath]]:
    rng = np.random.default_rng(seed)

    all_observables = []
    all_unbound_paths = []
    all_depths = []
    patterns: dict[str, MockUnboundPath] = {}

    for pp_name, depths in pattern_depths.items():
        pp = MockUnboundPath(pp_name)
        patterns[pp_name] = pp
        for d in depths:
            true_val = a_values[pp_name] * f_values[pp_name] ** d
            values = true_val + rng.normal(0, noise_std, size=n_rand)
            all_observables.append(values)
            all_unbound_paths.append(pp)
            all_depths.append(d)

    n = len(all_observables)
    observables = np.stack(all_observables)
    return ObservableData.from_arrays(
        unbound_paths=all_unbound_paths,
        depths=all_depths,
        observables=observables,
        time_lbs=np.empty((n, n_rand), dtype="datetime64[us]"),
        time_ubs=np.empty((n, n_rand), dtype="datetime64[us]"),
    ), patterns


def _make_model(patterns: dict[str, MockUnboundPath]) -> MockFidelityModel:
    return MockFidelityModel(
        {pp: IndexedVector({f"r_{name}": 1.0}) for name, pp in patterns.items()}
    )


class TestRowValidIndices:
    def test_no_nan(self):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        valid = _row_valid_indices(arr)
        assert len(valid) == 2
        np.testing.assert_array_equal(valid[0], [0, 1, 2])
        np.testing.assert_array_equal(valid[1], [0, 1, 2])

    def test_with_nan(self):
        arr = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, 6.0]])
        valid = _row_valid_indices(arr)
        np.testing.assert_array_equal(valid[0], [0, 2])
        np.testing.assert_array_equal(valid[1], [2])


class TestNumpyResampler:
    def test_yields_correct_count(self):
        obs, _ = _make_observable_data(
            pattern_depths={"p": [1, 2]},
            a_values={"p": 0.9},
            f_values={"p": 0.8},
            n_rand=10,
        )
        resampler = NumpyResampler(seed=0)
        replicates = list(resampler.resample(obs, n_resamples=5))
        assert len(replicates) == 5

    def test_replicates_have_same_shape(self):
        obs, _ = _make_observable_data(
            pattern_depths={"p": [1, 2, 3]},
            a_values={"p": 0.9},
            f_values={"p": 0.8},
            n_rand=10,
        )
        resampler = NumpyResampler(seed=0)
        for replicate in resampler.resample(obs, n_resamples=3):
            assert replicate.observables.shape == obs.observables.shape

    def test_draws_from_original_values(self):
        obs, _ = _make_observable_data(
            pattern_depths={"p": [1]},
            a_values={"p": 0.9},
            f_values={"p": 0.8},
            n_rand=10,
        )
        resampler = NumpyResampler(seed=0)
        original = set(obs.observables.data[0].tolist())
        for replicate in resampler.resample(obs, n_resamples=4):
            for v in replicate.observables.data[0].tolist():
                assert v in original

    def test_seed_reproducibility(self):
        obs, _ = _make_observable_data(
            pattern_depths={"p": [1]},
            a_values={"p": 0.9},
            f_values={"p": 0.8},
            n_rand=10,
        )
        r1 = list(NumpyResampler(seed=123).resample(obs, n_resamples=3))
        r2 = list(NumpyResampler(seed=123).resample(obs, n_resamples=3))
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a.observables.data, b.observables.data)


class TestScipyResampler:
    def test_yields_correct_count(self):
        obs, _ = _make_observable_data(
            pattern_depths={"p": [1, 2]},
            a_values={"p": 0.9},
            f_values={"p": 0.8},
            n_rand=10,
        )
        resampler = ScipyResampler(seed=0)
        replicates = list(resampler.resample(obs, n_resamples=4))
        assert len(replicates) == 4

    def test_replicates_have_same_shape(self):
        obs, _ = _make_observable_data(
            pattern_depths={"p": [1, 2]},
            a_values={"p": 0.9},
            f_values={"p": 0.8},
            n_rand=10,
        )
        resampler = ScipyResampler(seed=0)
        for replicate in resampler.resample(obs, n_resamples=3):
            assert replicate.observables.shape == obs.observables.shape

    def test_draws_from_original_values(self):
        obs, _ = _make_observable_data(
            pattern_depths={"p": [1]},
            a_values={"p": 0.9},
            f_values={"p": 0.8},
            n_rand=10,
        )
        resampler = ScipyResampler(seed=0)
        original = set(obs.observables.data[0].tolist())
        for replicate in resampler.resample(obs, n_resamples=4):
            for v in replicate.observables.data[0].tolist():
                assert v in original


class TestArchResampler:
    def test_yields_correct_count(self):
        obs, _ = _make_observable_data(
            pattern_depths={"p": [1, 2]},
            a_values={"p": 0.9},
            f_values={"p": 0.8},
            n_rand=10,
        )
        resampler = ArchResampler(seed=0)
        replicates = list(resampler.resample(obs, n_resamples=4))
        assert len(replicates) == 4

    def test_draws_from_original_values(self):
        obs, _ = _make_observable_data(
            pattern_depths={"p": [1]},
            a_values={"p": 0.9},
            f_values={"p": 0.8},
            n_rand=10,
        )
        resampler = ArchResampler(seed=0)
        original = set(obs.observables.data[0].tolist())
        for replicate in resampler.resample(obs, n_resamples=4):
            for v in replicate.observables.data[0].tolist():
                assert v in original


class TestBootstrapValidation:
    def test_rejects_non_observable_input_level(self):
        # AverageObservables has input ObservableData -> ok; we need a non-observable input.
        # Compose a pipeline ending in ModelData but starting at AveragedData (NNLSSolve alone).
        with pytest.raises(ValueError, match="ObservableData"):
            Bootstrap(NNLSSolve(), n_resamples=5, resampler=NumpyResampler(seed=0))

    def test_rejects_non_param_output_level(self):
        # FlipPostSelect: RawData -> RawData, not allowed either way; use a stage with output
        # not in {AveragedData, ModelData}. We can build one inline.
        from qiskit_noise_learning.analysis import AnalysisStage
        from qiskit_noise_learning.data import RawData

        class ObsToRaw(AnalysisStage):
            @property
            def input_level(self):
                return ObservableData

            @property
            def output_level(self):
                return RawData

            def _run(self, fit):
                pass

        with pytest.raises(ValueError, match="AveragedData or ModelData"):
            Bootstrap(ObsToRaw(), n_resamples=5, resampler=NumpyResampler(seed=0))

    def test_rejects_invalid_n_resamples(self):
        with pytest.raises(ValueError, match="n_resamples"):
            Bootstrap(
                CurveFitObservables() + NNLSSolve(),
                n_resamples=0,
                resampler=NumpyResampler(seed=0),
            )

    def test_rejects_invalid_confidence_level(self):
        with pytest.raises(ValueError, match="confidence_level"):
            Bootstrap(
                CurveFitObservables() + NNLSSolve(),
                n_resamples=5,
                resampler=NumpyResampler(seed=0),
                confidence_level=1.0,
            )


class TestBootstrapAveragedData:
    def test_bootstrap_average_observables(self):
        obs, _ = _make_observable_data(
            pattern_depths={"p": [1]},
            a_values={"p": 0.9},
            f_values={"p": 0.8},
            n_rand=50,
        )
        bootstrap_stage = Bootstrap(
            AverageObservables(), n_resamples=200, resampler=NumpyResampler(seed=0)
        )
        assert bootstrap_stage.input_level is ObservableData
        assert bootstrap_stage.output_level is AveragedData

        result = bootstrap_stage.run(obs)
        avg = result.averaged_data
        assert isinstance(avg, AveragedData)
        ds = avg.dataset
        assert "bootstrap_samples" in ds
        assert "bootstrap_std" in ds
        assert "bootstrap_ci_low" in ds
        assert "bootstrap_ci_high" in ds
        assert ds["bootstrap_samples"].dims == ("resample", "observable")
        assert ds["bootstrap_samples"].sizes["resample"] == 200
        # Bootstrap std should be close to the analytical SEM.
        analytical_sem = float(np.std(obs.observables.data[0], ddof=1) / np.sqrt(50))
        assert abs(float(ds["bootstrap_std"].data[0]) - analytical_sem) < 0.5 * analytical_sem


class TestBootstrapModelData:
    def _setup(self, resampler, n_resamples=100, n_rand=40, seed=7):
        f_true = 0.85
        obs, patterns = _make_observable_data(
            pattern_depths={"p": [1, 2, 3, 4, 5]},
            a_values={"p": 0.95},
            f_values={"p": f_true},
            n_rand=n_rand,
            noise_std=0.01,
            seed=seed,
        )
        model = _make_model(patterns)
        inner = AnalysisPipeline(CurveFitObservables(), NNLSSolve())
        bootstrap_stage = Bootstrap(inner, n_resamples=n_resamples, resampler=resampler)
        fit = Fit(model=model)
        fit[ObservableData] = obs
        return f_true, bootstrap_stage.run(fit)

    def test_numpy_resampler_recovers_decay(self):
        f_true, result = self._setup(NumpyResampler(seed=0))
        md = result.model_data
        assert isinstance(md, ModelData)
        ds = md.dataset
        # Point estimate is close to truth.
        param = float(ds["parameter_values"].sel(parameter="r_p").item())
        assert abs(param - (-np.log(f_true))) < 0.05
        # Bootstrap std and CI exist with consistent shapes.
        assert ds["bootstrap_samples"].dims == ("resample", "parameter")
        assert ds["bootstrap_samples"].sizes["resample"] == 100
        # CI brackets the point estimate.
        ci_low = float(ds["bootstrap_ci_low"].sel(parameter="r_p").item())
        ci_high = float(ds["bootstrap_ci_high"].sel(parameter="r_p").item())
        assert ci_low <= param <= ci_high

    def test_scipy_resampler_runs(self):
        _, result = self._setup(ScipyResampler(seed=0), n_resamples=50)
        ds = result.model_data.dataset
        assert ds["bootstrap_samples"].sizes["resample"] == 50
        # std should be positive and finite for an under-determined-but-clean fit.
        assert float(ds["bootstrap_std"].sel(parameter="r_p").item()) > 0

    def test_arch_resampler_runs(self):
        _, result = self._setup(ArchResampler(seed=0), n_resamples=50)
        ds = result.model_data.dataset
        assert ds["bootstrap_samples"].sizes["resample"] == 50
        assert float(ds["bootstrap_std"].sel(parameter="r_p").item()) > 0

    def test_metadata_recorded(self):
        _, result = self._setup(NumpyResampler(seed=0), n_resamples=10)
        attrs = result.model_data.dataset.attrs
        assert attrs["bootstrap_n_resamples"] == 10
        assert attrs["bootstrap_confidence_level"] == 0.95
        assert attrs["bootstrap_resampler"] == "NumpyResampler"

    def test_resamplers_give_similar_std(self):
        """Sanity check: naive numpy and arch resamplers yield comparable std estimates."""
        rng_seed = 11
        _, np_result = self._setup(NumpyResampler(seed=rng_seed), n_resamples=300)
        _, arch_result = self._setup(ArchResampler(seed=rng_seed), n_resamples=300)
        np_std = float(np_result.model_data.dataset["bootstrap_std"].sel(parameter="r_p").item())
        arch_std = float(
            arch_result.model_data.dataset["bootstrap_std"].sel(parameter="r_p").item()
        )
        # Within 30% of each other (both estimating the same underlying quantity).
        assert abs(np_std - arch_std) / max(np_std, arch_std) < 0.3
