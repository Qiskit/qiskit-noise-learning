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

from typing import Self

from qiskit_noise_learning.data import (
    AveragedData,
    LeveledData,
    ModelData,
    ObservableData,
    RawData,
)
from qiskit_noise_learning.models import FidelityModel, is_fidelity_model
from qiskit_noise_learning.sequences import InstructionSequence, Path

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
