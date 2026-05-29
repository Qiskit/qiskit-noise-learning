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

"""Experiment builder stage abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .experiment import Experiment


class ExperimentBuilderStage(ABC):
    """Abstract base for a stage in the experiment-building pipeline.

    Each stage takes an :class:`Experiment` and returns a new :class:`Experiment` with additional
    fields populated or existing fields transformed.

    Subclasses declare which :class:`Experiment` fields they require via the
    :attr:`required_fields` class attribute. The public :meth:`run` method validates that those
    fields are not ``None`` before dispatching to :meth:`_run`.

    Args:
        required_fields: Tuple of :class:`Experiment` property names that must be non-``None``
            before this stage can execute.
    """

    required_fields: tuple[str, ...] = ()

    def run(self, experiment: Experiment) -> Experiment:
        """Validate required fields, then apply this stage.

        Args:
            experiment: The input experiment.

        Returns:
            A new :class:`Experiment` with this stage's transformations applied.

        Raises:
            ValueError: If any field listed in :attr:`required_fields` is ``None``.
        """
        for field in self.required_fields:
            if getattr(experiment, field) is None:
                raise ValueError(
                    f"Stage {type(self).__name__} requires '{field}' to be set on the experiment."
                )
        return self._run(experiment)

    @abstractmethod
    def _run(self, experiment: Experiment) -> Experiment:
        """Implement the stage logic.

        Subclasses can assume that all :attr:`required_fields` are non-``None``.

        Args:
            experiment: The input experiment (validated).

        Returns:
            A new :class:`Experiment`.
        """

    def __add__(self, other: ExperimentBuilderStage) -> ExperimentBuilder:
        if not isinstance(other, ExperimentBuilderStage):
            return NotImplemented
        return ExperimentBuilder(self, other)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ExperimentBuilder(ExperimentBuilderStage):
    """A composite :class:`ExperimentBuilderStage` that chains stages sequentially.

    Because :class:`ExperimentBuilder` is itself an :class:`ExperimentBuilderStage`, builders
    can be nested and used anywhere a single stage is expected.

    Args:
        stages: The stages to chain in sequential order.
    """

    def __init__(self, *stages: ExperimentBuilderStage):
        flat: list[ExperimentBuilderStage] = []
        for stage in stages:
            if isinstance(stage, ExperimentBuilder):
                flat.extend(stage._stages)  # noqa: SLF001
            else:
                flat.append(stage)
        self._stages = tuple(flat)

    @property
    def stages(self) -> tuple[ExperimentBuilderStage, ...]:
        """The stages in this builder."""
        return self._stages

    def _run(self, experiment: Experiment) -> Experiment:
        for stage in self._stages:
            experiment = stage.run(experiment)
        return experiment

    def __repr__(self) -> str:
        stages_repr = ", ".join(repr(s) for s in self._stages)
        return f"ExperimentBuilder({stages_repr})"
