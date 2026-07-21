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

"""Fit."""

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Self

from qiskit_noise_learning.data import (
    AveragedData,
    LeveledData,
    ModelData,
    ObservableData,
    RawData,
)
from qiskit_noise_learning.models import FidelityModel, get_noise_site, is_fidelity_model
from qiskit_noise_learning.sequences import InstructionSequence, Path

if TYPE_CHECKING:
    import plotly.graph_objects as go

LEVELS = (RawData, ObservableData, AveragedData, ModelData)
"""The levels of the analysis hierarchy."""


class AbsentType:
    """Sentinel for a data level that has not yet been computed."""


class SkippedType:
    """Sentinel for a data level that was intentionally bypassed by a stage."""


Absent = AbsentType()
"""Singleton sentinel indicating a data level has not yet been computed."""

Skipped = SkippedType()
"""Singleton sentinel indicating a data level was intentionally bypassed."""

_LevelData = LeveledData | AbsentType | SkippedType
_LevelHistory = dict[type[LeveledData], list[_LevelData]]


class FitHistory:
    """A view of the full write history for each level in a :class:`Fit` container."""

    def __init__(self, store: _LevelHistory):
        self._store = store

    @property
    def raw_data(self) -> list[RawData | AbsentType | SkippedType]:
        """History of values written to the :class:`RawData` level."""
        return self._store[RawData]

    @property
    def observable_data(self) -> list[ObservableData | AbsentType | SkippedType]:
        """History of values written to the :class:`ObservableData` level."""
        return self._store[ObservableData]

    @property
    def averaged_data(self) -> list[AveragedData | AbsentType | SkippedType]:
        """History of values written to the :class:`AveragedData` level."""
        return self._store[AveragedData]

    @property
    def model_data(self) -> list[ModelData | AbsentType | SkippedType]:
        """History of values written to the :class:`ModelData` level."""
        return self._store[ModelData]


class Fit:
    """Container for data at each level of the analysis hierarchy.

    The :data:`LEVELS` represent different stages of processed data. In order, they are:

    * :class:`RawData` corresponding to bit counts of executed instruction sequences.
    * :class:`ObservableData` corresponding to observables of randomizations of paths that traverse
      the instruction sequences.
    * :class:`AveragedData` corresponding to a combination of observables averaged across
      randomizations.
    * :class:`ModelData` corresponding to model fit parameters.

    Each level holds a history of all values written to it; the current value is always
    the most recent. New levels start as :data:`Absent`. Levels bypassed by a stage are
    marked :data:`Skipped`.

    Example::

        result = my_stage.run(my_raw_data)
        result.raw_data           # current RawData
        result.history.raw_data   # full write history at the RawData level

    Args:
        model: The model to fit. If given, it must be a fidelity model, i.e. a
            :class:`~.LinearMap` whose output space is a :class:`~.LogFidelitySpace` (see
            :func:`~.is_fidelity_model`).
        paths: The paths to analyze.
        instruction_sequences: The instruction sequences used in the experiment.
        relations: A pre-computed set of ``(path_idx, sequence_idx)`` tuples indicating which paths
            are traversed by which instruction sequences.

    Raises:
        TypeError: If ``model`` is not ``None`` and is not a fidelity model.
    """

    def __init__(
        self,
        *,
        model: FidelityModel | None = None,
        paths: list[Path] | None = None,
        instruction_sequences: list[InstructionSequence] | None = None,
        relations: set[tuple[int, int]] | None = None,
        _store: _LevelHistory | None = None,
    ):
        if model is not None and not is_fidelity_model(model):
            raise TypeError(
                "Fit requires the model to be a fidelity model (a LinearMap whose output space is "
                f"a LogFidelitySpace), but got {type(model).__name__}."
            )
        self._model = model
        self._paths = paths or []
        self._instruction_sequences = instruction_sequences
        self._relations = relations
        self._store: _LevelHistory = {t: [Absent] for t in LEVELS} if _store is None else _store

    def copy(self) -> Self:
        """Return a copy preserving the full history at each level."""
        return Fit(
            model=self._model,
            paths=list(self._paths),
            instruction_sequences=(
                list(self._instruction_sequences)
                if self._instruction_sequences is not None
                else None
            ),
            relations=set(self._relations) if self._relations is not None else None,
            _store={t: list(h) for t, h in self._store.items()},
        )

    def __getitem__(self, level_type: type[LeveledData]) -> _LevelData:
        return self._store[level_type][-1]

    def __setitem__(self, level_type: type[LeveledData], value: _LevelData) -> None:
        self._store[level_type].append(value)

    @property
    def history(self) -> FitHistory:
        """The full write history for each level."""
        return FitHistory(self._store)

    @property
    def model(self) -> FidelityModel | None:
        """The fidelity model used for design matrix construction."""
        return self._model

    @property
    def raw_data(self) -> RawData | AbsentType | SkippedType:
        """Current data at the :class:`RawData` level."""
        return self[RawData]

    @property
    def observable_data(self) -> ObservableData | AbsentType | SkippedType:
        """Current data at the :class:`ObservableData` level."""
        return self[ObservableData]

    @property
    def averaged_data(self) -> AveragedData | AbsentType | SkippedType:
        """Current data at the :class:`AveragedData` level."""
        return self[AveragedData]

    @property
    def model_data(self) -> ModelData | AbsentType | SkippedType:
        """Current data at the :class:`ModelData` level."""
        return self[ModelData]

    @property
    def paths(self) -> list[Path]:
        """The paths to compute observables for."""
        return self._paths

    @property
    def instruction_sequences(self) -> list[InstructionSequence] | None:
        """The instruction sequences used in the experiment, or ``None`` if not set."""
        return self._instruction_sequences

    @property
    def relations(self) -> set[tuple[int, int]] | None:
        """Path-to-sequence relations, or ``None`` if not set."""
        return self._relations

    def plot_qubit_pair_decays(
        self,
        pairs: Sequence[tuple[int, int]],
        *,
        observable_type: Literal["raw", "means", "both"] | None = None,
        exponential_fit: bool = False,
        model_prediction: bool = False,
        observable_marker_kwargs: Mapping[str, object] | None = None,
        means_marker_kwargs: Mapping[str, object] | None = None,
        exponential_fit_line_kwargs: Mapping[str, object] | None = None,
        model_line_kwargs: Mapping[str, object] | None = None,
        num_cols: int = 3,
        noise_site: Mapping[str, str] | None = None,
        paths: Sequence[Path] | None = None,
        fragment_depths: Sequence[float] | None = None,
        title: str | None = None,
    ) -> "go.Figure":
        """Plot a grid of fidelity decays over qubit pairs, drawn from this fit's data.

        One subplot per pair, sharing labels/colors across pairs. Which decays are drawn is
        controlled by the toggles below; all default to off, so enable the ones you want. A
        requested decay whose data has not been computed on this fit yet is skipped with a warning.

        Args:
            pairs: The qubit pairs to plot, one subplot each.
            observable_type: How to draw the empirical observable data: ``"raw"`` (raw
                per-randomization scatter), ``"means"`` (per-fragment-depth means with error bars),
                ``"both"``, or ``None`` (the default) to omit the empirical points. Uses this fit's
                :class:`~.ObservableData`.
            exponential_fit: Whether to draw the fitted exponential decay curve, from this fit's
                :class:`~.AveragedData` (its ``fragment_depth == -1`` fitted parameters). Defaults
                to ``False``.
            model_prediction: Whether to draw the model-predicted decay curve, from this fit's
                model and :class:`~.ModelData`. Defaults to ``False``.
            observable_marker_kwargs: Optional ``marker`` overrides for the raw observable points.
            means_marker_kwargs: Optional ``marker`` overrides for the observable-means points.
            exponential_fit_line_kwargs: Optional ``line`` overrides for the exponential-fit curve.
            model_line_kwargs: Optional ``line`` overrides for the model curve.
            num_cols: The number of subplot columns; rows are derived from the pair count.
            noise_site: An optional noise-site mapping forwarded to the label formatter (with the
                default ``"formula"`` label style this yields the compact ``f^{gate}_{pauli}``
                label). Defaults to the noise site of the fit's model when it is, or contains, a
                single :class:`~.PauliLindbladModel`.
            paths: The paths to draw across all layers. Defaults to the decay paths found in this
                fit's observable/averaged data, falling back to the fit's own ``paths`` when no such
                data is present. Supply this to draw model-prediction curves for a fit that carries
                only a model (no observable or averaged data to derive the paths from).
            fragment_depths: The fragment-depth range for the curves. Defaults to ``0`` through the
                largest fragment depth in the empirical data present, or ``0``–``10`` when there is
                none.
            title: An optional figure title.

        Returns:
            A plotly Figure.

        Raises:
            ValueError: If the fit has no model (and hence no gate set) to build labels from.
            ImportError: If ``plotly`` is not installed.
        """
        from ..visualizations.path_data.orchestrators import (
            plot_qubit_pair_decays as _plot_qubit_pair_decays,
        )

        def _present(level: _LevelData) -> LeveledData | None:
            return level if isinstance(level, LeveledData) else None

        if self._model is None:
            raise ValueError(
                "Fit.plot_qubit_pair_decays needs a model carrying a gate set to build labels."
            )
        gate_set = self._model.output_space.gate_set

        if noise_site is None:
            noise_site = get_noise_site(self._model)

        observable_data = _present(self.observable_data) if observable_type is not None else None
        averaged_data = _present(self.averaged_data) if exponential_fit else None
        model_data = _present(self.model_data) if model_prediction else None
        model = self._model if model_data is not None else None

        # Warn (rather than silently skip) when a decay was requested but its data is not available.
        for requested, resolved, name in (
            (observable_type is not None, observable_data, "observable points"),
            (exponential_fit, averaged_data, "exponential-fit curve"),
            (model_prediction, model_data, "model-prediction curve"),
        ):
            if requested and resolved is None:
                warnings.warn(
                    f"{name} requested but the required data has not been computed on this fit; "
                    "skipping it.",
                    stacklevel=2,
                )

        # Fall back to the fit's own paths only when there is no empirical data to derive them from
        # (e.g. a model-only prediction plot); explicit paths always win.
        if paths is None and observable_data is None and averaged_data is None:
            paths = self._paths or None

        return _plot_qubit_pair_decays(
            pairs,
            observable_data=observable_data,
            observable_type=observable_type or "raw",
            observable_marker_kwargs=observable_marker_kwargs,
            means_marker_kwargs=means_marker_kwargs,
            averaged_data=averaged_data,
            exponential_fit_line_kwargs=exponential_fit_line_kwargs,
            model=model,
            model_data=model_data,
            model_line_kwargs=model_line_kwargs,
            gate_set=gate_set,
            num_cols=num_cols,
            noise_site=noise_site,
            paths=paths,
            fragment_depths=fragment_depths,
            title=title,
        )
