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

"""The instrumented experiment build.

:func:`timed_build` runs exactly the pipeline the utility-benchmark notebook runs, but with the
stages invoked one at a time instead of composed with ``+`` so that each can be timed. Since
:meth:`~.ExperimentBuilder._run` is itself just a loop over ``stage.run``, this changes nothing
about the work performed.

Two liberties are taken with the notebook, both of which only *split* a measurement rather than
change one:

* ``experiment.design_matrix`` is forced before each :class:`~.RankReducePaths` call. The property
  is lazily cached, so this separates the cost of building the design matrix from the cost of the
  Gram-Schmidt elimination that consumes it, and the stage itself then finds the matrix already
  built.
* :attr:`~.IndexedMatrix.rank` is **not** computed unless explicitly asked for. The notebook prints
  it at the end; its :func:`~numpy.linalg.svd` costs about as much as the entire build, so leaving
  it in would swamp everything the profile is trying to show.
"""

import platform
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from qiskit_noise_learning.experiment_builder import (
    BindFragmentDepths,
    CompleteSequences,
    Depth1Paths,
    EvenDepthPaths,
    Experiment,
    GenerateInstructionSequences,
    MergeInstructionSequences,
    RankReducePaths,
    SPAMPaths,
)

from .cases import BenchmarkCase, build_gate_set, build_model
from .fingerprint import experiment_fingerprint
from .instrument import CallRecord, ComponentTimers, StageTimeline

#: Gate names that are preparation and measurement rather than layer gates.
SPAM_GATE_NAMES = ("P", "M")


@dataclass
class BuildResult:
    """Everything one instrumented build measured.

    Args:
        case: The case that was built.
        num_edges: Couplings in the topology.
        num_layers: Disjoint two-qubit layers, and hence layer gates.
        generators: Generator count per model gate.
        num_paths: Paths surviving the overall rank reduction.
        num_sequences: Instruction sequences after merging, before depth binding.
        num_bound_sequences: Instruction sequences after depth binding.
        design_matrix_shape: Shape of the design matrix entering the overall rank reduction.
        design_matrix_nnz: Its nonzero count.
        rank: The design matrix rank, if it was requested.
        seconds: Total wall time of the instrumented build.
        peak_rss_mb: Peak resident set size of the process, in MiB.
        timeline: Per-stage timings in execution order.
        components: Per-method timings.
        fingerprint: Digests of the build's output; see :mod:`.fingerprint`.
        environment: Interpreter and platform details, so results can be compared honestly.
    """

    case: BenchmarkCase
    num_edges: int = 0
    num_layers: int = 0
    generators: dict[str, int] = field(default_factory=dict)
    num_paths: int = 0
    num_sequences: int = 0
    num_bound_sequences: int = 0
    design_matrix_shape: tuple[int, int] = (0, 0)
    design_matrix_nnz: int = 0
    rank: int | None = None
    seconds: float = 0.0
    peak_rss_mb: float = 0.0
    timeline: StageTimeline = field(default_factory=StageTimeline)
    components: dict[str, CallRecord] = field(default_factory=dict)
    fingerprint: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)

    @property
    def design_matrix_density(self) -> float:
        """The fraction of design-matrix entries that are nonzero."""
        rows, cols = self.design_matrix_shape
        return self.design_matrix_nnz / (rows * cols) if rows and cols else 0.0

    @property
    def total_generators(self) -> int:
        """The total number of model generators over all gates."""
        return sum(self.generators.values())


def peak_rss_mb() -> float:
    """The process's peak resident set size in MiB.

    ``ru_maxrss`` is bytes on macOS and kibibytes on Linux; both are normalized here.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024**2 if sys.platform == "darwin" else raw / 1024


def _environment() -> dict[str, str]:
    """Interpreter, numpy, and platform identification for the run."""
    import numpy
    import qiskit

    import qiskit_noise_learning

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "qiskit": qiskit.__version__,
        "qiskit_noise_learning": getattr(qiskit_noise_learning, "__version__", "unknown"),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
    }


def timed_build(
    case: BenchmarkCase,
    *,
    compute_rank: bool = False,
    fingerprint: bool = True,
    progress: Callable[[str], None] | None = None,
) -> BuildResult:
    """Run and time the experiment build for one case.

    Args:
        case: The case to build.
        compute_rank: Whether to compute the design matrix rank at the end. This is off by
            default because its dense SVD costs roughly as much as the whole build.
        fingerprint: Whether to digest the build's output. Cheap next to the build itself, but it
            does walk every path, so it can be turned off for pure timing runs.
        progress: Called with a short status line as each phase starts.

    Returns:
        The measurement.
    """
    say = progress or (lambda _message: None)
    result = BuildResult(case=case, environment=_environment())
    timeline = result.timeline

    start = time.perf_counter()
    with ComponentTimers() as components:
        say(f"{case.name}: gate set")
        with timeline.time("model/gate_set"):
            gate_set, topology = build_gate_set(case)
        with timeline.time("model/generators"):
            model = build_model(case, gate_set, topology)

        result.num_edges = topology.num_edges
        result.num_layers = len(topology.layers)
        result.generators = {name: len(gens) for name, gens in model.generators.items()}

        layer_gates = {
            name: gate
            for name, gate in gate_set.model_gate_set.items()
            if name not in SPAM_GATE_NAMES
        }

        empty = Experiment(
            fidelity_model=model, shots=case.shots, randomizations=case.randomizations
        )
        experiment = empty

        for kind, stage_class in (("mult", EvenDepthPaths), ("add", Depth1Paths)):
            for name, gate in layer_gates.items():
                say(f"{case.name}: {kind} {name}")
                with timeline.time(f"{kind}/{name}/paths"):
                    partial = stage_class(gates=[gate]).run(empty)
                with timeline.time(f"{kind}/{name}/design_matrix"):
                    partial.design_matrix  # noqa: B018
                with timeline.time(f"{kind}/{name}/rank_reduce"):
                    partial = RankReducePaths().run(partial)
                with timeline.time(f"{kind}/{name}/concat"):
                    experiment = experiment + partial

        say(f"{case.name}: spam")
        with timeline.time("spam/paths"):
            partial = SPAMPaths().run(empty)
        with timeline.time("spam/design_matrix"):
            partial.design_matrix  # noqa: B018
        with timeline.time("spam/rank_reduce"):
            partial = RankReducePaths().run(partial)
        with timeline.time("spam/concat"):
            experiment = experiment + partial

        say(f"{case.name}: overall rank reduction")
        with timeline.time("overall/design_matrix"):
            design_matrix = experiment.design_matrix
        result.design_matrix_shape = tuple(design_matrix.shape)
        result.design_matrix_nnz = int((design_matrix.data != 0).sum())
        with timeline.time("overall/rank_reduce"):
            experiment = RankReducePaths().run(experiment)

        result.num_paths = len(experiment.paths)

        say(f"{case.name}: finalize")
        with timeline.time("finalize/generate"):
            experiment = GenerateInstructionSequences().run(experiment)
        with timeline.time("finalize/merge"):
            experiment = MergeInstructionSequences().run(experiment)
        with timeline.time("finalize/complete"):
            experiment = CompleteSequences().run(experiment)
        result.num_sequences = len(experiment.instruction_sequences)
        with timeline.time("finalize/bind"):
            experiment = BindFragmentDepths(list(case.fragment_depths)).run(experiment)
        result.num_bound_sequences = len(experiment.instruction_sequences)

    result.seconds = time.perf_counter() - start
    result.components = components.records
    result.peak_rss_mb = peak_rss_mb()

    if fingerprint:
        say(f"{case.name}: fingerprint")
        result.fingerprint = experiment_fingerprint(experiment, design_matrix=False)
    if compute_rank:
        say(f"{case.name}: rank")
        result.rank = experiment.design_matrix.rank

    return result
