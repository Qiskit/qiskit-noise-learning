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

"""Wall-clock instrumentation for the benchmark builds.

Two views of the same build are recorded:

* a **stage** view, which times each :class:`~.ExperimentBuilderStage` invocation, and
* a **component** view, which times individual hot methods wherever they are called from.

The component view is what tells you whether a stage's cost is linear algebra or Python
bookkeeping, so it is worth the monkeypatching it costs. Patches are installed by
:class:`ComponentTimers` for the duration of a ``with`` block and removed afterwards.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from qiskit_noise_learning.math import IndexedMatrix
from qiskit_noise_learning.models import PauliLindbladModel
from qiskit_noise_learning.sequences import LogPathMap

#: The methods the component view patches, as ``(label, owner, attribute)``. Private attributes are
#: unavoidable here: these are exactly the internals whose cost is being attributed, and the
#: benchmark is allowed to know about them in a way the library's own code is not.
_PATCH_TARGETS: tuple[tuple[str, type, str], ...] = (
    ("PauliLindbladModel.rows", PauliLindbladModel, "rows"),
    ("LogPathMap.rows", LogPathMap, "rows"),
    ("IndexedMatrix.add_rows", IndexedMatrix, "add_rows"),
    ("IndexedMatrix._matmul_matrix", IndexedMatrix, "_matmul_matrix"),
    ("IndexedMatrix.linearly_independent_rows", IndexedMatrix, "linearly_independent_rows"),
)


@dataclass
class CallRecord:
    """Accumulated timing for one instrumented method.

    Args:
        label: The method's display name.
        calls: How many times it was called.
        seconds: Total wall time spent inside it, excluding nested instrumented calls.
        gross_seconds: Total wall time spent inside it, including nested instrumented calls.
        call_log: One ``(self seconds, operand description)`` entry per call, in call order. This
            is what identifies the handful of individual calls that dominate a build.
    """

    label: str
    calls: int = 0
    seconds: float = 0.0
    gross_seconds: float = 0.0
    call_log: list[tuple[float, str]] = field(default_factory=list)


class ComponentTimers:
    """Context manager that times the methods in :data:`_PATCH_TARGETS`.

    Nested instrumented calls are subtracted from their caller's ``seconds``, so the recorded
    self-times sum to the instrumented portion of the build without double counting. ``add_rows``
    called from inside ``rows``, for example, is charged only to ``add_rows``.
    """

    def __init__(self):
        self.records: dict[str, CallRecord] = {
            label: CallRecord(label) for label, _, _ in _PATCH_TARGETS
        }
        self._originals: list[tuple[type, str, object]] = []
        # Time consumed by instrumented calls nested inside the call currently on top of the stack.
        self._child_time: list[float] = []

    def __enter__(self) -> "ComponentTimers":
        for label, owner, attribute in _PATCH_TARGETS:
            original = getattr(owner, attribute)
            self._originals.append((owner, attribute, original))
            setattr(owner, attribute, self._wrap(label, original))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        for owner, attribute, original in self._originals:
            setattr(owner, attribute, original)
        self._originals.clear()
        return False

    def _wrap(self, label: str, original):
        record = self.records[label]

        def wrapper(*args, **kwargs):
            self._child_time.append(0.0)
            start = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            finally:
                gross = time.perf_counter() - start
                children = self._child_time.pop()
                record.calls += 1
                record.gross_seconds += gross
                record.seconds += gross - children
                if self._child_time:
                    self._child_time[-1] += gross
            record.call_log.append((gross - children, _describe(label, args, kwargs, result)))
            return result

        return wrapper


def _describe(label: str, args: tuple, kwargs: dict, result: object) -> str:
    """A one-line operand-size description of an instrumented call.

    Args:
        label: The instrumented method's name.
        args: Its positional arguments, including the receiver.
        kwargs: Its keyword arguments. Callers inside the library pass some of these hot
            arguments by keyword, so both have to be consulted.
        result: Its return value.

    Returns:
        A short description, or the empty string if the method has no interesting operands.
    """
    receiver = args[0] if args else None

    def positional_or_keyword(position: int, name: str) -> object:
        if len(args) > position:
            return args[position]
        return kwargs.get(name)

    if label == "IndexedMatrix._matmul_matrix":
        other = positional_or_keyword(1, "other")
        return f"{receiver.shape} @ {getattr(other, 'shape', '?')}"
    if label == "IndexedMatrix.linearly_independent_rows":
        return f"{receiver.shape} -> {result.shape}"
    if label == "IndexedMatrix.add_rows":
        rows = positional_or_keyword(2, "rows")
        return f"+{len(rows) if rows is not None else '?'} rows into {receiver.shape}"
    if label in ("PauliLindbladModel.rows", "LogPathMap.rows"):
        return f"{result.shape}"
    return ""


@dataclass
class GroupTiming:
    """Accumulated timing for a set of stage invocations sharing a label pattern.

    Args:
        label: The group's pattern, with the collapsed segment written as ``*``.
        seconds: Total wall time over the group's invocations.
        calls: How many invocations the group covers.
    """

    label: str
    seconds: float = 0.0
    calls: int = 0


@dataclass
class StageTimeline:
    """An ordered record of stage timings.

    Args:
        entries: ``(label, seconds)`` in execution order.
    """

    entries: list[tuple[str, float]] = field(default_factory=list)

    @contextmanager
    def time(self, label: str):
        """Time a block and append it to the timeline.

        Args:
            label: The name to record the block under.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.entries.append((label, time.perf_counter() - start))

    @property
    def total(self) -> float:
        """Total recorded time."""
        return sum(seconds for _, seconds in self.entries)

    def grouped(self) -> dict[str, GroupTiming]:
        """Timings summed over entries whose labels agree up to their middle segment.

        Labels are ``phase/subject/step``, and the subject -- which gate a per-gate stage ran for
        -- is what gets collapsed, so ``mult/layer_1/paths`` and ``mult/layer_2/paths`` are summed
        into ``mult/*/paths``. Labels with fewer segments group as themselves. Insertion order is
        preserved, so the grouping reads in pipeline order.

        Returns:
            A mapping from group label to its accumulated timing.
        """
        groups: dict[str, GroupTiming] = {}
        for label, seconds in self.entries:
            parts = label.split("/")
            key = f"{parts[0]}/*/{parts[-1]}" if len(parts) > 2 else label
            group = groups.setdefault(key, GroupTiming(key))
            group.seconds += seconds
            group.calls += 1
        return groups
