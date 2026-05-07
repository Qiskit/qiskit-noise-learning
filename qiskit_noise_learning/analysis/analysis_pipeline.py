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

"""Analysis pipeline."""

from abc import ABC, abstractmethod

from qiskit_noise_learning.analysis.fit import LEVELS, Absent, Fit, Skipped
from qiskit_noise_learning.data import LeveledData


class AnalysisStage(ABC):
    """Abstract base for a stage in the analysis pipeline.

    Each stage declares the data level it consumes (:attr:`input_level`) and produces
    (:attr:`output_level`). Stages may skip intermediate levels, e.g. going directly from
    :class:`RawData` to :class:`DecayData`.

    To implement a stage, subclass this and override :meth:`_run`. The public :meth:`run`
    method handles shallow-copying the :class:`Fit` container and marking skipped levels;
    :meth:`_run` receives the copy and may mutate it in place.

    :attr:`input_level` and :attr:`output_level` can be declared as class attributes::

        class MyStage(AnalysisStage):
            input_level = RawData
            output_level = ObservableData

            def _run(self, fit):
                fit[ObservableData] = compute(fit[RawData])
    """

    @property
    @abstractmethod
    def input_level(self) -> type[LeveledData]:
        """The data level this stage reads."""

    @property
    @abstractmethod
    def output_level(self) -> type[LeveledData]:
        """The data level this stage writes."""

    @abstractmethod
    def _run(self, fit: Fit):
        """Implement the stage: read ``fit[self.input_level]``, write ``fit[self.output_level]``.

        The fit container may be mutated in place.
        """

    def __add__(self, other: "AnalysisStage") -> "AnalysisPipeline":
        if not isinstance(other, AnalysisStage):
            return NotImplemented
        return AnalysisPipeline(self, other)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def run(self, fit: Fit | LeveledData) -> Fit:
        """Run this stage, returning a new :class:`Fit` with the output level populated.

        Shallow-copies ``fit``, marks any :data:`Absent` intermediate levels as :data:`Skipped`,
        calls :meth:`_run` on the copy, and returns it. The original ``fit`` is not modified.
        """
        if isinstance(fit, LeveledData):
            result = Fit()
            result[type(fit)] = fit
        else:
            result = fit.copy()

        start = LEVELS.index(self.input_level)
        end = LEVELS.index(self.output_level)
        for level_type in LEVELS[start + 1 : end]:
            result[level_type] = Skipped
        for level_type in LEVELS[end:]:
            result[level_type] = Absent
        self._run(result)
        return result


class AnalysisPipeline(AnalysisStage):
    """A composite :class:`AnalysisStage` that chains stages sequentially.

    Because :class:`AnalysisPipeline` is itself an :class:`AnalysisStage`, pipelines can be
    nested and used anywhere a single stage is expected.

    Consecutive stages must connect: the :attr:`~AnalysisStage.output_level` of each stage must
    equal the :attr:`~AnalysisStage.input_level` of the next.

    Args:
        stages: The analysis stages to use in this pipeline in sequential order.
    """

    def __init__(self, *stages: AnalysisStage):
        flat: list[AnalysisStage] = []
        for stage in stages:
            if isinstance(stage, AnalysisPipeline):
                flat.extend(stage._stages)  # noqa: SLF001
            else:
                flat.append(stage)
        self._stages = tuple(flat)
        for stage0, stage1 in zip(self._stages, self._stages[1:]):
            if stage0.output_level != stage1.input_level:
                raise ValueError(
                    f"The output level of {stage0} does not match the input level of {stage1}."
                )

    @property
    def input_level(self) -> type[LeveledData]:
        return self._stages[0].input_level

    @property
    def output_level(self) -> type[LeveledData]:
        return self._stages[-1].output_level

    @property
    def stages(self) -> tuple[AnalysisStage, ...]:
        """The stages in this pipeline."""
        return self._stages

    def __iadd__(self, other: AnalysisStage) -> "AnalysisPipeline":
        if not isinstance(other, AnalysisStage):
            return NotImplemented
        new_stages = list(other._stages) if isinstance(other, AnalysisPipeline) else [other]  # noqa: SLF001
        if self._stages and new_stages:
            if self._stages[-1].output_level != new_stages[0].input_level:
                raise ValueError(
                    f"The output level of {self._stages[-1]} does not match the input level of "
                    f"{new_stages[0]}."
                )
        self._stages = (*self._stages, *new_stages)
        return self

    def __repr__(self) -> str:
        stages_repr = ", ".join(repr(s) for s in self._stages)
        return f"AnalysisPipeline({stages_repr})"

    def _run(self, fit: Fit):
        for stage in self._stages:
            stage._run(fit)  # noqa: SLF001 — intentional: bypass copy for pipeline chaining
