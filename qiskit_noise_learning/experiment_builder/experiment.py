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

"""Experiment class."""

from __future__ import annotations

import warnings
from copy import copy

from qiskit_noise_learning.gate_sets import ModelGateSet
from qiskit_noise_learning.math import IndexedMatrix
from qiskit_noise_learning.models._legacy import CompleteFidelityModel, FidelityModel
from qiskit_noise_learning.sequences import InstructionSequence, Path


class Experiment:
    """A learning experiment specification.

    An :class:`Experiment` collects all the data needed to define a noise-learning experiment:
    a fidelity model, analysis paths, instruction sequences, their relations, and execution
    parameters (shots and randomizations).

    All fields are optional and may be progressively populated via
    :class:`~.ExperimentBuilderStage` instances.

    Args:
        fidelity_model: A fidelity model or a model gate set (which is wrapped in a
            :class:`~.CompleteFidelityModel`).
        paths: Paths to analyze.
        instruction_sequences: Instruction sequences (may include both bound and unbound).
        relations: Set of ``(path_idx, sequence_idx)`` tuples indicating which paths are
            traversed by which instruction sequences.
        shots: Global number of shots (default 20).
        randomizations: Global number of randomizations (default 50).
        randomization_multipliers: Per-sequence randomization multiplier (parallel to
            instruction_sequences).
        validate: If ``True`` (default), enforce the same validation checks as
            :meth:`replace` (co-replacement, length consistency, relations bounds).
    """

    def __init__(
        self,
        *,
        fidelity_model: FidelityModel | ModelGateSet | None = None,
        paths: list[Path] | None = None,
        instruction_sequences: list[InstructionSequence] | None = None,
        relations: set[tuple[int, int]] | None = None,
        shots: int = 20,
        randomizations: int = 50,
        randomization_multipliers: list[int] | None = None,
        validate: bool = True,
    ):
        if isinstance(fidelity_model, ModelGateSet):
            fidelity_model = CompleteFidelityModel(fidelity_model)

        self._fidelity_model = None
        self._paths = None
        self._instruction_sequences = None
        self._relations = None
        self._shots = 20
        self._randomizations = 50
        self._randomization_multipliers = None
        self._design_matrix_cache: IndexedMatrix | None = None

        result = self.replace(
            validate=validate,
            fidelity_model=fidelity_model,
            paths=paths,
            instruction_sequences=instruction_sequences,
            relations=relations,
            shots=shots,
            randomizations=randomizations,
            randomization_multipliers=randomization_multipliers,
        )
        self.__dict__.update(result.__dict__)

    @property
    def fidelity_model(self) -> FidelityModel | None:
        """The fidelity model."""
        return self._fidelity_model

    @property
    def gate_set(self) -> ModelGateSet | None:
        """The model gate set."""
        if self._fidelity_model is None:
            return None
        return self._fidelity_model.gate_set

    @property
    def paths(self) -> list[Path] | None:
        """The analysis paths."""
        return self._paths

    @property
    def instruction_sequences(self) -> list[InstructionSequence] | None:
        """The instruction sequences."""
        return self._instruction_sequences

    @property
    def relations(self) -> set[tuple[int, int]] | None:
        """The set of path and sequence relations."""
        return self._relations

    @property
    def shots(self) -> int:
        """Global number of shots."""
        return self._shots

    @property
    def randomizations(self) -> int:
        """Global number of randomizations."""
        return self._randomizations

    @property
    def randomization_multipliers(self) -> list[int] | None:
        """Per-sequence randomization multipliers."""
        return self._randomization_multipliers

    @property
    def design_matrix(self) -> IndexedMatrix:
        """The design matrix, lazily computed from the fidelity model and paths.

        Raises:
            ValueError: If ``fidelity_model`` or ``paths`` is ``None``.
        """
        if self._design_matrix_cache is None:
            if self._fidelity_model is None:
                raise ValueError("Cannot compute design matrix: fidelity_model is None.")
            if self._paths is None:
                raise ValueError("Cannot compute design matrix: paths is None.")

            self._design_matrix_cache = self._compute_design_matrix()

        return self._design_matrix_cache

    @property
    def is_executable(self) -> bool:
        """Whether this experiment has all the information required to be run.

        Requires: ``instruction_sequences`` is set, all sequences are bound and complete,
        and ``randomization_multipliers`` is set.
        """
        if self._instruction_sequences is None:
            return False
        if any(seq.is_unbound or not seq.is_complete for seq in self._instruction_sequences):
            return False
        if self._randomization_multipliers is None:
            return False
        return True

    def replace(self, *, validate: bool = True, **kwargs) -> Experiment:
        """Return a shallow copy with the given fields overridden.

        When ``validate=True`` (default), the following checks are enforced:

        - **Co-replacement**: ``instruction_sequences`` and ``randomization_multipliers`` must
          always both be ``None`` or both be non-``None``. Replacing one without the other is
          only allowed if it preserves this invariant.
        - **Length consistency**: ``randomization_multipliers`` must have the same length as
          ``instruction_sequences``.
        - **Relations bounds**: Setting ``relations`` requires ``paths`` and
          ``instruction_sequences`` to be present, and all indices must be in bounds.
        - **Soft invalidation**: Replacing ``paths`` or ``instruction_sequences`` without
          providing new ``relations`` will set ``relations`` to ``None`` with a warning.

        Args:
            validate: If ``True``, enforce the above checks. If ``False``, fields are set
                as-is with no validation.
            **kwargs: Field names and their new values.

        Raises:
            TypeError: If an unrecognized field name is given.
            ValueError: If a validation constraint is violated.
        """
        for key in kwargs:
            if not hasattr(self, f"_{key}"):
                raise TypeError(f"Experiment has no field '{key}'.")

        new = copy(self)
        for key, value in kwargs.items():
            setattr(new, f"_{key}", value)

        if validate:
            # Co-replacement invariant: both None or both non-None
            seqs_none = new.instruction_sequences is None
            mult_none = new.randomization_multipliers is None
            if seqs_none != mult_none:
                raise ValueError(
                    "instruction_sequences and randomization_multipliers must both be None "
                    "or both be non-None."
                )

            # Length consistency
            if not seqs_none and not mult_none:
                if len(new.randomization_multipliers) != len(new.instruction_sequences):
                    raise ValueError(
                        f"randomization_multipliers length "
                        f"({len(new.randomization_multipliers)}) does not match "
                        f"instruction_sequences length ({len(new.instruction_sequences)})."
                    )

            # Relations bounds check
            if "relations" in kwargs and new.relations is not None:
                if new.paths is None:
                    raise ValueError("Cannot set relations: paths is None.")
                if new.instruction_sequences is None:
                    raise ValueError("Cannot set relations: instruction_sequences is None.")
                n_paths = len(new.paths)
                n_seqs = len(new.instruction_sequences)
                for path_idx, seq_idx in new.relations:
                    if path_idx >= n_paths or seq_idx >= n_seqs:
                        raise ValueError(
                            f"Relation ({path_idx}, {seq_idx}) is out of bounds "
                            f"(paths={n_paths}, sequences={n_seqs})."
                        )

            # Soft invalidation: replacing paths or sequences invalidates relations
            if ("paths" in kwargs or "instruction_sequences" in kwargs) and (
                "relations" not in kwargs and new.relations is not None
            ):
                warnings.warn(
                    "Replacing paths or instruction_sequences invalidates relations. "
                    "Setting relations to None.",
                    stacklevel=2,
                )
                new._relations = None  # noqa: SLF001

        # Always invalidate design matrix cache
        if "paths" in kwargs or "fidelity_model" in kwargs:
            new._design_matrix_cache = None  # noqa: SLF001

        return new

    def __str__(self) -> str:
        lines = ["Experiment:"]

        if self._fidelity_model is not None:
            lines.append(f"  Fidelity model: {self._fidelity_model}")
        else:
            lines.append("  Fidelity model: None")

        if self._paths is not None:
            n_unbound = sum(1 for p in self._paths if p.is_unbound)
            n_bound = len(self._paths) - n_unbound
            lines.append(f"  Paths: {len(self._paths)} ({n_unbound} unbound, {n_bound} bound)")
        else:
            lines.append("  Paths: None")

        if self._instruction_sequences is not None:
            n_unbound = sum(1 for s in self._instruction_sequences if s.is_unbound)
            n_bound = len(self._instruction_sequences) - n_unbound
            lines.append(
                f"  Instruction sequences: {len(self._instruction_sequences)}"
                f" ({n_unbound} unbound, {n_bound} bound)"
            )
        else:
            lines.append("  Instruction sequences: None")

        if self._relations is not None:
            lines.append(f"  Relations: {len(self._relations)}")
        else:
            lines.append("  Relations: None")

        lines.append(f"  Shots: {self._shots}")
        lines.append(f"  Randomizations: {self._randomizations}")
        lines.append(f"  Randomization multipliers: {self._randomization_multipliers}")

        return "\n".join(lines)

    def __add__(self, other: Experiment) -> Experiment:
        """Compose two experiments by concatenating their list fields.

        Scalar fields (``fidelity_model``, ``shots``, ``randomizations``) must match or both
        be ``None``. List fields (``paths``, ``instruction_sequences``,
        ``randomization_multipliers``) are concatenated, with ``None`` treated as empty (unless
        both are ``None``, in which case the result is ``None``). Relation indices from
        ``other`` are offset to account for the concatenation.

        Raises:
            TypeError: If ``other`` is not an :class:`Experiment`.
            ValueError: If scalar fields do not match.
        """
        if not isinstance(other, Experiment):
            return NotImplemented

        # Scalar fields must match
        for field in ("fidelity_model", "shots", "randomizations"):
            self_val = getattr(self, f"_{field}")
            other_val = getattr(other, f"_{field}")
            if self_val != other_val:
                raise ValueError(
                    f"Cannot compose experiments: '{field}' does not match "
                    f"({self_val!r} vs {other_val!r})."
                )

        # Concatenate list fields (None treated as empty unless both None)
        paths = _optional_concat(self._paths, other._paths)
        instruction_sequences = _optional_concat(
            self._instruction_sequences, other._instruction_sequences
        )
        randomization_multipliers = _optional_concat(
            self._randomization_multipliers, other._randomization_multipliers
        )

        # Offset and merge relations
        relations = None
        if self._relations is not None or other._relations is not None:
            relations = set(self._relations) if self._relations else set()
            if other._relations is not None:
                n_paths_offset = len(self._paths) if self._paths else 0
                n_seqs_offset = (
                    len(self._instruction_sequences) if self._instruction_sequences else 0
                )
                for path_idx, seq_idx in other._relations:
                    relations.add((path_idx + n_paths_offset, seq_idx + n_seqs_offset))

        return Experiment(
            fidelity_model=self._fidelity_model,
            paths=paths,
            instruction_sequences=instruction_sequences,
            relations=relations,
            shots=self._shots,
            randomizations=self._randomizations,
            randomization_multipliers=randomization_multipliers,
        )

    def _compute_design_matrix(self) -> IndexedMatrix:
        matrix = IndexedMatrix()
        rows = []
        for path in self._paths:
            rows.append(self._fidelity_model.row_from_path(path))
        matrix.add_rows(row_indices=self._paths, rows=rows)
        return matrix


def _optional_concat(a: list | None, b: list | None) -> list | None:
    """Concatenate two optional lists. Returns None only if both are None."""
    if a is None and b is None:
        return None
    return (a or []) + (b or [])
