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

"""Process-independent digests of the objects a build produces.

An optimization is only interesting if it leaves the answer alone, so every benchmark run records
digests that can be compared across runs, machines, and library versions. ``__hash__`` cannot be
used for this: :class:`~.Path` and :class:`~.FidelityIndex` hash through :func:`hash` of strings,
which Python randomizes per process. Nor can ``repr``:
:class:`~qiskit.quantum_info.QubitSparsePauli` has no content-based ``repr`` on every version, and
an address that leaks into a digest silently makes it useless.

So content is serialized explicitly here, by :func:`_canonical`, which raises on any type it has
not been taught. A benchmark that stops fingerprinting something is much better than one that
fingerprints an address.
"""

import hashlib
from collections.abc import Iterable

import numpy as np

from qiskit_noise_learning.experiment_builder import Experiment
from qiskit_noise_learning.math import IndexedMatrix
from qiskit_noise_learning.sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionSequence,
    PartialPauliPermutation,
    Path,
)
from qiskit_noise_learning.sequences.instruction import Instruction


def _canonical(value: object) -> bytes:
    """Serialize a value to bytes deterministically, across processes and machines.

    Args:
        value: The value to serialize. Scalars, byte strings, numpy arrays and scalars, and
            (possibly nested) sequences, sets, and mappings of those are understood.

    Returns:
        A byte string that is equal for equal values and unequal for unequal ones.

    Raises:
        TypeError: If ``value`` contains a type this function has not been taught to serialize.
            Guessing would risk digesting a memory address, which would defeat the purpose.
    """
    if value is None:
        return b"n;"
    if isinstance(value, bool):
        return b"b1;" if value else b"b0;"
    if isinstance(value, int | np.integer):
        return b"i%d;" % int(value)
    if isinstance(value, float | np.floating):
        # repr of a float round-trips exactly, and is stable across platforms.
        return b"f" + repr(float(value)).encode() + b";"
    if isinstance(value, str):
        return b"s%d:%s;" % (len(value), value.encode())
    if isinstance(value, bytes | bytearray | memoryview):
        raw = bytes(value)
        return b"y%d:%s;" % (len(raw), raw)
    if isinstance(value, np.ndarray):
        # The dtype string and shape are included so that arrays differing only in either do not
        # collide, and C order is forced so a view's strides cannot matter.
        header = f"a{value.dtype.str}{value.shape}:".encode()
        return header + np.ascontiguousarray(value).tobytes() + b";"
    if isinstance(value, tuple | list):
        return b"t%d:" % len(value) + b"".join(_canonical(item) for item in value) + b";"
    if isinstance(value, frozenset | set):
        # Sets have no order, so serialize the sorted digests of the elements.
        return b"e%d:" % len(value) + b"".join(sorted(_canonical(item) for item in value)) + b";"
    if isinstance(value, dict):
        items = sorted((_canonical(key), _canonical(item)) for key, item in value.items())
        return b"d%d:" % len(items) + b"".join(key + item for key, item in items) + b";"
    raise TypeError(
        f"Refusing to fingerprint a {type(value).__name__}: no content-based serialization is "
        "defined for it, and falling back on repr risks digesting a memory address."
    )


def _digest(value: object) -> str:
    """A short hexadecimal digest of :func:`_canonical` of a value."""
    return hashlib.sha256(_canonical(value)).hexdigest()[:16]


def _fidelity_index_content(index: FidelityIndex) -> tuple:
    """The full content of a fidelity index, as canonicalizable values."""
    pauli = index.pauli
    return (
        index.gate_name,
        pauli.num_qubits,
        pauli.paulis,
        pauli.indices,
        index.in_z_idxs,
        index.out_z_idxs,
    )


def _path_content(path: Path) -> tuple:
    """The full content of a path, as canonicalizable values."""
    return (
        path.fragment_depth,
        tuple(_fidelity_index_content(index) for index in path.start_fragment),
        tuple(_fidelity_index_content(index) for index in path.repeatable_fragment),
        tuple(_fidelity_index_content(index) for index in path.end_fragment),
    )


def _instruction_content(instruction: Instruction) -> tuple:
    """The full content of an instruction, as canonicalizable values.

    Args:
        instruction: The instruction to serialize.

    Returns:
        Canonicalizable content.

    Raises:
        TypeError: If the instruction is of a type this module does not know how to serialize;
            see :func:`_canonical` for why no fallback is offered.
    """
    if isinstance(instruction, ApplyGate):
        return ("gate", instruction.gate_name)
    if isinstance(instruction, PartialPauliPermutation):
        return (
            "permutation",
            instruction.num_qubits,
            instruction.partial_permutation_indices,
        )
    raise TypeError(
        f"Refusing to fingerprint the {type(instruction).__name__} instruction: no content-based "
        "serialization is defined for it."
    )


def _sequence_content(sequence: InstructionSequence) -> tuple:
    """The full content of an instruction sequence, as canonicalizable values."""
    return (
        sequence.fragment_depth,
        tuple(_instruction_content(instr) for instr in sequence.start_fragment),
        tuple(_instruction_content(instr) for instr in sequence.repeatable_fragment),
        tuple(_instruction_content(instr) for instr in sequence.end_fragment),
    )


def paths_digest(paths: Iterable[Path]) -> str:
    """Digest the ordered list of paths.

    Order is included: it is what the rank reduction's "prefer earlier rows" rule acts on, so two
    builds that select the same paths in a different order are not interchangeable.

    Args:
        paths: The paths to digest.

    Returns:
        A hexadecimal digest.
    """
    return _digest(tuple(_path_content(path) for path in paths))


def path_set_digest(paths: Iterable[Path]) -> str:
    """Digest the paths as an unordered set.

    Compare this with :func:`paths_digest` to tell a reordering apart from a change of content.
    Digests of the paths are sorted rather than the contents themselves, since path content
    includes numpy arrays and so is not itself hashable or orderable.

    Args:
        paths: The paths to digest.

    Returns:
        A hexadecimal digest.
    """
    return _digest(tuple(sorted(_digest(_path_content(path)) for path in paths)))


def sequences_digest(sequences: Iterable[InstructionSequence]) -> str:
    """Digest instruction sequences as an unordered multiset.

    Sequence order out of the merge stage is an implementation detail of the grouping, but which
    sequences come out is not, so the multiset is the right invariant.

    Args:
        sequences: The sequences to digest.

    Returns:
        A hexadecimal digest.
    """
    return _digest(tuple(sorted(_digest(_sequence_content(seq)) for seq in sequences)))


def design_matrix_digest(matrix: IndexedMatrix) -> str:
    """Digest a design matrix in a way that is blind to row and column ordering.

    Row and column order comes from insertion order, which optimizations may legitimately change,
    while the linear-algebraic content may not. So what is digested is the shape, the nonzero
    count, and the sorted per-row and per-column nonzero counts together with sorted row and
    column absolute sums -- enough to catch a real change without failing on a permutation.

    Args:
        matrix: The matrix to digest.

    Returns:
        A hexadecimal digest.
    """
    data = np.asarray(matrix.data)
    nonzero = data != 0
    return _digest(
        (
            data.shape,
            int(nonzero.sum()),
            np.sort(nonzero.sum(axis=1)),
            np.sort(nonzero.sum(axis=0)),
            np.sort(np.round(np.abs(data).sum(axis=1), 9)),
            np.sort(np.round(np.abs(data).sum(axis=0), 9)),
        )
    )


def experiment_fingerprint(experiment: Experiment, *, design_matrix: bool = True) -> dict[str, str]:
    """Fingerprint every part of an experiment that an optimization must not change.

    Args:
        experiment: The experiment to fingerprint.
        design_matrix: Whether to include the design matrix. Building it is expensive, so pass
            ``False`` if it has not already been computed and cached.

    Returns:
        A mapping from part name to hexadecimal digest. Parts that the experiment does not have
        are absent.
    """
    out: dict[str, str] = {}
    if experiment.paths is not None:
        out["paths"] = paths_digest(experiment.paths)
        out["path_set"] = path_set_digest(experiment.paths)
    if experiment.instruction_sequences is not None:
        out["sequences"] = sequences_digest(experiment.instruction_sequences)
    if experiment.relations is not None:
        out["relations"] = _digest(frozenset(experiment.relations))
    if design_matrix and experiment.paths is not None:
        out["design_matrix"] = design_matrix_digest(experiment.design_matrix)
    return out
