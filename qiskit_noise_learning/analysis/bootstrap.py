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

"""Bootstrap-based uncertainty quantification for the analysis pipeline.

This module provides a :class:`Bootstrap` :class:`~.AnalysisStage` that resamples randomizations
of an :class:`~.ObservableData` instance, repeatedly runs an inner sub-pipeline, and aggregates
the resulting parameter (or observable) distributions into bootstrap standard deviations and
percentile confidence intervals.

The bootstrap stage decouples *resampling strategy* (how we draw bootstrap samples) from
*aggregation* (how we summarize the resampled outputs) by delegating the former to a
:class:`Resampler` instance. Three resamplers are provided:

* :class:`NumpyResampler`: a naive resampler using :func:`numpy.random.Generator.choice` for
  per-row independent resampling. Cheap, simple, and useful as a sanity check.
* :class:`ScipyResampler`: builds on :func:`scipy.stats.bootstrap` to obtain BCa or percentile
  confidence intervals from the per-observable randomization arrays.
* :class:`ArchResampler`: uses :class:`arch.bootstrap.IIDBootstrap`, which provides richer
  bootstrap diagnostics. Enabled only when ``arch`` is installed.

Example::

    from qiskit_noise_learning.analysis import (
        Bootstrap,
        ComputeObservables,
        CurveFitObservables,
        NNLSSolve,
        NumpyResampler,
    )

    inner = CurveFitObservables() + NNLSSolve()
    pipeline = ComputeObservables() + Bootstrap(
        inner, n_resamples=200, resampler=NumpyResampler(seed=42)
    )
    fit = pipeline.run(fit)
    fit.model_data.dataset["bootstrap_std"]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qiskit_noise_learning.analysis.analysis_pipeline import AnalysisStage
from qiskit_noise_learning.analysis.fit import Fit
from qiskit_noise_learning.data import (
    AveragedData,
    LeveledData,
    ModelData,
    ObservableData,
)

if TYPE_CHECKING:
    from arch.bootstrap import IIDBootstrap


# Mapping from output-level types to the dataset variable that holds the per-parameter values
# whose bootstrap distribution we summarize.
_VALUE_VAR: dict[type[LeveledData], str] = {
    AveragedData: "observables",
    ModelData: "parameter_values",
}


class Resampler(ABC):
    """Abstract resampler producing bootstrap samples of an :class:`~.ObservableData`.

    A resampler is responsible for drawing ``n_resamples`` bootstrap replicates of the
    per-randomization observable values. Subclasses implement different strategies (e.g. naive
    numpy, scipy, arch).

    Each resampler must yield ``ObservableData`` instances that have the same dataset shape as
    the input (same ``observable`` and ``randomization`` dimensions, same coordinates), with the
    ``observables`` data variable filled in by drawing -- with replacement, by default -- from
    the non-NaN entries of each row.
    """

    @abstractmethod
    def resample(
        self, observable_data: ObservableData, n_resamples: int
    ) -> Iterator[ObservableData]:
        """Yield ``n_resamples`` bootstrap replicates of ``observable_data``.

        Args:
            observable_data: The observable data to resample.
            n_resamples: The number of bootstrap replicates to produce.

        Yields:
            ``ObservableData`` instances with the same dataset shape as the input.
        """


def _row_valid_indices(values: np.ndarray) -> list[np.ndarray]:
    """For each row of ``values``, return the indices of non-NaN entries."""
    valid: list[np.ndarray] = []
    for row in values:
        valid.append(np.flatnonzero(~np.isnan(row)))
    return valid


def _build_resampled_dataset(
    observable_data: ObservableData, resampled_values: np.ndarray
) -> ObservableData:
    """Build a new :class:`~.ObservableData` whose ``observables`` array is replaced.

    Coordinates and other variables are preserved from the original dataset.

    Args:
        observable_data: The original observable data.
        resampled_values: An array with the same shape as ``observable_data.observables.data``.
    """
    ds = observable_data.dataset.copy()
    ds = ds.assign(
        observables=xr.DataArray(data=resampled_values, dims=["observable", "randomization"])
    )
    return ObservableData(dataset=ds)


class NumpyResampler(Resampler):
    """Naive bootstrap resampler using :class:`numpy.random.Generator`.

    For each row (i.e. each observable), this resampler draws ``num_valid`` indices uniformly
    with replacement from the row's non-NaN entries, where ``num_valid`` is the number of
    non-NaN entries in that row. Different rows are resampled independently.

    This is the simplest possible bootstrap; it does not preserve correlations between
    observables that came from the same underlying instruction-pattern randomization. It is
    primarily useful as a sanity check against more sophisticated resamplers.

    Args:
        seed: Optional seed for reproducibility.
    """

    def __init__(self, seed: int | np.random.Generator | None = None):
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    def resample(
        self, observable_data: ObservableData, n_resamples: int
    ) -> Iterator[ObservableData]:
        values = observable_data.observables.data
        valid_indices = _row_valid_indices(values)

        for _ in range(n_resamples):
            new_values = np.full_like(values, np.nan)
            for r_idx, idxs in enumerate(valid_indices):
                if idxs.size == 0:
                    continue
                draws = self._rng.choice(idxs, size=idxs.size, replace=True)
                new_values[r_idx, : idxs.size] = values[r_idx, draws]
            yield _build_resampled_dataset(observable_data, new_values)


class ScipyResampler(Resampler):
    """Bootstrap resampler delegating to :func:`scipy.stats.bootstrap`.

    Each row of the observable array is treated as an independent 1D sample passed to
    :func:`scipy.stats.bootstrap` with ``method="percentile"`` (or another caller-provided
    method) and ``statistic=lambda x: x``, which returns the resampled draws themselves. The
    resulting ``bootstrap_distribution`` has shape ``(num_valid, n_resamples)``.

    This produces bootstrap replicates equivalent to :class:`NumpyResampler` but routed through
    SciPy's machinery (which is convenient when the analyst wants to use SciPy's confidence
    interval methods downstream as a comparison).

    Args:
        seed: Optional seed for reproducibility.
        method: Confidence-interval method passed through to ``scipy.stats.bootstrap``. The
            value here only affects internal SciPy bookkeeping; the bootstrap stage always
            recomputes its own percentile intervals from the parameter distribution.
    """

    def __init__(
        self,
        seed: int | np.random.Generator | None = None,
        method: str = "percentile",
    ):
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._method = method

    def resample(
        self, observable_data: ObservableData, n_resamples: int
    ) -> Iterator[ObservableData]:
        from scipy.stats import bootstrap as scipy_bootstrap

        values = observable_data.observables.data
        valid_indices = _row_valid_indices(values)

        # For each row, run scipy.stats.bootstrap with the identity statistic to obtain a
        # (n_resamples, num_valid) array of resampled draws.
        per_row_distributions: list[np.ndarray] = []
        for r_idx, idxs in enumerate(valid_indices):
            if idxs.size == 0:
                per_row_distributions.append(np.empty((n_resamples, 0)))
                continue
            data = values[r_idx, idxs]
            if idxs.size == 1:
                # scipy.stats.bootstrap requires sample size >= 2; degenerate row -> repeat.
                per_row_distributions.append(np.tile(data[None, :], (n_resamples, 1)))
                continue
            res = scipy_bootstrap(
                (data,),
                statistic=lambda x, axis=-1: x,
                n_resamples=n_resamples,
                method=self._method,
                random_state=self._rng,
                vectorized=True,
                paired=False,
            )
            # bootstrap_distribution has shape (n_resamples, sample_size) for identity statistic.
            per_row_distributions.append(np.asarray(res.bootstrap_distribution))

        for k in range(n_resamples):
            new_values = np.full_like(values, np.nan)
            for r_idx, dist in enumerate(per_row_distributions):
                if dist.size == 0:
                    continue
                draws = dist[k, :]
                new_values[r_idx, : draws.size] = draws
            yield _build_resampled_dataset(observable_data, new_values)


class ArchResampler(Resampler):
    """Bootstrap resampler using :class:`arch.bootstrap.IIDBootstrap`.

    Each row is resampled independently using ``arch``'s IID bootstrap, which is the most
    rigorous of the three options here in the sense that it cleanly separates the bootstrap
    *iteration* from any subsequent statistic computation, and it natively supports more
    complex bootstrap variants (block, stationary, etc.) that can be substituted by passing a
    different ``bootstrap_cls``.

    This resampler requires the optional ``arch`` package (``pip install arch``).

    Args:
        seed: Optional seed for reproducibility.
        bootstrap_cls: The ``arch.bootstrap`` class to use (defaults to ``IIDBootstrap``). Pass
            e.g. ``arch.bootstrap.StationaryBootstrap`` if there's structure in randomization
            ordering you want to preserve.
        bootstrap_kwargs: Additional keyword arguments forwarded to ``bootstrap_cls`` after the
            data array (e.g. ``block_size`` for block bootstraps).
    """

    def __init__(
        self,
        seed: int | np.random.Generator | None = None,
        bootstrap_cls: type[IIDBootstrap] | None = None,
        **bootstrap_kwargs,
    ):
        try:
            from arch.bootstrap import IIDBootstrap
        except ImportError as exc:  # pragma: no cover - exercised only without arch installed
            raise ImportError(
                "ArchResampler requires the 'arch' package. Install it with `pip install arch`."
            ) from exc

        self._bootstrap_cls = bootstrap_cls or IIDBootstrap
        self._bootstrap_kwargs = bootstrap_kwargs
        if isinstance(seed, np.random.Generator):
            self._seed = seed
        elif seed is None:
            self._seed = np.random.default_rng()
        else:
            self._seed = np.random.default_rng(seed)

    def resample(
        self, observable_data: ObservableData, n_resamples: int
    ) -> Iterator[ObservableData]:
        values = observable_data.observables.data
        valid_indices = _row_valid_indices(values)

        # Pre-build per-row bootstrap distributions of shape (num_valid, n_resamples).
        per_row_distributions: list[np.ndarray] = []
        for r_idx, idxs in enumerate(valid_indices):
            if idxs.size == 0:
                per_row_distributions.append(np.empty((0, n_resamples)))
                continue
            data = values[r_idx, idxs]
            bs = self._bootstrap_cls(data, seed=self._seed, **self._bootstrap_kwargs)
            row_draws = np.empty((idxs.size, n_resamples), dtype=float)
            for k, (pos_args, _) in enumerate(bs.bootstrap(n_resamples)):
                row_draws[:, k] = pos_args[0]
            per_row_distributions.append(row_draws)

        for k in range(n_resamples):
            new_values = np.full_like(values, np.nan)
            for r_idx, dist in enumerate(per_row_distributions):
                if dist.size == 0:
                    continue
                draws = dist[:, k]
                new_values[r_idx, : draws.size] = draws
            yield _build_resampled_dataset(observable_data, new_values)


class Bootstrap(AnalysisStage):
    """Wrap an analysis sub-pipeline and bootstrap its parameter estimates.

    This stage:

    1. Runs ``inner`` once on the original :class:`~.ObservableData` to compute the point
       estimate (which is set as the current value at the output level).
    2. Asks ``resampler`` for ``n_resamples`` bootstrap replicates of the observable data.
    3. Runs ``inner`` on each replicate, extracting the parameter array from the output level
       (``observables`` for :class:`~.AveragedData` outputs, ``parameter_values`` for
       :class:`~.ModelData` outputs).
    4. Stacks the per-replicate parameter arrays and computes:

       * ``bootstrap_samples``: a 2D array of shape ``(resample, parameter)`` containing every
         bootstrap replicate's parameter estimates,
       * ``bootstrap_std``: the standard deviation across resamples,
       * ``bootstrap_ci_low`` and ``bootstrap_ci_high``: percentile confidence-interval bounds.

       These are attached as additional data variables on the output dataset, alongside the
       point-estimate values.

    The output level coordinates are inherited from the point-estimate run, so downstream
    stages and visualizations can use the bootstrap results without any changes.

    Args:
        inner: The :class:`~.AnalysisStage` to bootstrap. Its ``input_level`` must be
            :class:`~.ObservableData`, and its ``output_level`` must be either
            :class:`~.AveragedData` or :class:`~.ModelData`.
        n_resamples: The number of bootstrap replicates to draw.
        resampler: The :class:`Resampler` to use to draw replicates.
        confidence_level: Two-sided confidence level for the percentile interval, in ``(0, 1)``.
            Defaults to ``0.95``.
    """

    def __init__(
        self,
        inner: AnalysisStage,
        n_resamples: int,
        resampler: Resampler,
        confidence_level: float = 0.95,
    ):
        if inner.input_level is not ObservableData:
            raise ValueError(
                "Bootstrap requires `inner.input_level` to be ObservableData; got "
                f"{inner.input_level.__name__}."
            )
        if inner.output_level not in _VALUE_VAR:
            raise ValueError(
                "Bootstrap requires `inner.output_level` to be AveragedData or ModelData; got "
                f"{inner.output_level.__name__}."
            )
        if n_resamples < 1:
            raise ValueError(f"`n_resamples` must be >= 1; got {n_resamples}.")
        if not 0 < confidence_level < 1:
            raise ValueError(f"`confidence_level` must be in (0, 1); got {confidence_level}.")

        self._inner = inner
        self._n_resamples = n_resamples
        self._resampler = resampler
        self._confidence_level = confidence_level

    @property
    def input_level(self) -> type[LeveledData]:
        return ObservableData

    @property
    def output_level(self) -> type[LeveledData]:
        return self._inner.output_level

    @property
    def inner(self) -> AnalysisStage:
        """The wrapped sub-pipeline."""
        return self._inner

    @property
    def n_resamples(self) -> int:
        """The number of bootstrap replicates."""
        return self._n_resamples

    @property
    def resampler(self) -> Resampler:
        """The :class:`Resampler` used to draw replicates."""
        return self._resampler

    @property
    def confidence_level(self) -> float:
        """The two-sided confidence level for percentile intervals."""
        return self._confidence_level

    def _run(self, fit: Fit) -> None:
        observable_data = fit[ObservableData]
        if not isinstance(observable_data, ObservableData):
            raise TypeError(
                "Bootstrap stage expected ObservableData at the input level but got "
                f"{type(observable_data).__name__}."
            )

        # Point estimate: run the inner pipeline on the original data, preserving model/paths.
        point_fit = self._inner.run(_make_inner_fit(fit, observable_data))
        point_output = point_fit[self.output_level]
        if not isinstance(point_output, LeveledData):
            raise RuntimeError("Inner pipeline did not produce a value at its output level.")

        # Bootstrap loop.
        value_var = _VALUE_VAR[self.output_level]
        replicate_arrays = list(
            _iter_replicate_values(
                inner=self._inner,
                base_fit=fit,
                value_var=value_var,
                replicates=self._resampler.resample(observable_data, self._n_resamples),
                point_output=point_output,
            )
        )
        replicate_stack = np.stack(replicate_arrays, axis=0)  # (n_resamples, n_params)

        # Aggregate.
        alpha = 1 - self._confidence_level
        ci_low = np.nanpercentile(replicate_stack, 100 * alpha / 2, axis=0)
        ci_high = np.nanpercentile(replicate_stack, 100 * (1 - alpha / 2), axis=0)
        std = np.nanstd(replicate_stack, axis=0, ddof=1)

        # Attach to the point-estimate output's dataset along the parameter axis.
        ds = point_output.dataset.copy()
        param_dim = _parameter_dim_for(self.output_level)
        ds = ds.assign(
            bootstrap_samples=xr.DataArray(data=replicate_stack, dims=["resample", param_dim]),
            bootstrap_std=xr.DataArray(data=std, dims=[param_dim]),
            bootstrap_ci_low=xr.DataArray(data=ci_low, dims=[param_dim]),
            bootstrap_ci_high=xr.DataArray(data=ci_high, dims=[param_dim]),
        )
        ds.attrs = {
            **ds.attrs,
            "bootstrap_n_resamples": self._n_resamples,
            "bootstrap_confidence_level": self._confidence_level,
            "bootstrap_resampler": type(self._resampler).__name__,
        }

        fit[self.output_level] = type(point_output)(dataset=ds)


def _parameter_dim_for(output_level: type[LeveledData]) -> str:
    """Return the dataset dimension that indexes parameters for the given output level."""
    if output_level is ModelData:
        return "parameter"
    if output_level is AveragedData:
        return "observable"
    raise ValueError(f"Unsupported bootstrap output level: {output_level.__name__}.")


def _make_inner_fit(base_fit: Fit, observable_data: ObservableData) -> Fit:
    """Construct a fresh :class:`Fit` for the inner pipeline.

    Preserves the model and paths from ``base_fit``, and seeds the :class:`~.ObservableData`
    level with ``observable_data``. Other levels start :data:`Absent`.
    """
    inner_fit = Fit(model=base_fit.model, paths=list(base_fit.paths))
    inner_fit[ObservableData] = observable_data
    return inner_fit


def _iter_replicate_values(
    *,
    inner: AnalysisStage,
    base_fit: Fit,
    value_var: str,
    replicates: Iterable[ObservableData],
    point_output: LeveledData,
) -> Iterator[np.ndarray]:
    """Yield the per-parameter value array from running ``inner`` on each replicate.

    The arrays are aligned to the point-estimate's parameter coordinate so that they stack
    cleanly into a 2D ``(n_resamples, n_params)`` array.

    Args:
        inner: The inner sub-pipeline.
        base_fit: The :class:`Fit` that holds the model/paths context.
        value_var: The dataset variable name carrying the per-parameter values
            (e.g. ``"parameter_values"`` for :class:`~.ModelData`).
        replicates: The bootstrap replicates to run ``inner`` on.
        point_output: The point-estimate output, used to align replicate parameter coordinates.
    """
    point_ds = point_output.dataset
    point_values = point_ds[value_var].data
    param_dim = point_ds[value_var].dims[0]
    point_param_index = point_ds[param_dim].data

    for replicate in replicates:
        replicate_fit = inner.run(_make_inner_fit(base_fit, replicate))
        replicate_output = replicate_fit[type(point_output)]
        if not isinstance(replicate_output, LeveledData):
            raise RuntimeError(
                "Inner pipeline did not produce a value at its output level for a replicate."
            )
        rep_ds = replicate_output.dataset
        rep_param_index = rep_ds[param_dim].data
        if rep_param_index.shape == point_param_index.shape and np.array_equal(
            rep_param_index, point_param_index
        ):
            yield rep_ds[value_var].data
        else:
            # Align by parameter label, padding with NaN where labels are missing in the
            # replicate (e.g. NNLS dropped a parameter to zero on this resample).
            aligned = np.full(point_values.shape, np.nan, dtype=float)
            label_to_idx = {label: i for i, label in enumerate(rep_param_index)}
            for i, label in enumerate(point_param_index):
                j = label_to_idx.get(label)
                if j is not None:
                    aligned[i] = rep_ds[value_var].data[j]
            yield aligned
