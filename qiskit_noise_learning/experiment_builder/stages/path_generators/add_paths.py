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

"""AddPaths stage."""

from collections.abc import Iterator

from qiskit_noise_learning.sequences import Path

from ...experiment import Experiment
from ...experiment_builder_stage import ExperimentBuilderStage


class AddPaths(ExperimentBuilderStage):
    """Add paths to an experiment.

    This is both a concrete stage (pass path iterators directly) and the base class for
    path-generator stages that compute paths from the experiment at runtime.

    Subclasses should override :meth:`_generate_paths` to yield paths computed from the
    experiment's fidelity model or other data.

    Args:
        path_iterators: One or more iterators of :class:`~.Path` instances.
    """

    populates_fields = ("paths",)

    def __init__(self, *path_iterators: Iterator[Path]):
        self._path_iterators = path_iterators

    def _run(self, experiment: Experiment) -> Experiment:
        existing_paths = list(experiment.paths) if experiment.paths is not None else []
        existing_paths.extend(self._generate_paths(experiment))
        return experiment.replace(validate=False, paths=existing_paths)

    def _generate_paths(self, experiment: Experiment) -> Iterator[Path]:
        """Yield paths to add to the experiment.

        The default implementation yields from the iterators passed at construction.
        Subclasses override this to generate paths from the experiment's data.
        """
        for iterator in self._path_iterators:
            yield from iterator
