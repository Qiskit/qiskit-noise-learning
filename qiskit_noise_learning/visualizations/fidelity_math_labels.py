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

"""Math-mode LaTeX labels for fidelity indices and paths."""

from collections import Counter
from collections.abc import Mapping
from typing import Literal

from qiskit.quantum_info import QubitSparsePauli

from qiskit_noise_learning.gate_sets import GateSet
from qiskit_noise_learning.sequences import FidelityIndex, Path


def fidelity_index_math_label(
    gate_set: GateSet,
    fidelity_index: FidelityIndex,
    style: Literal["transition", "formula"] = "transition",
    noise_site: Mapping[str, Literal["before", "after"]] | None = None,
) -> str:
    r"""Return a math-mode LaTeX label for a fidelity index.

    Args:
        gate_set: The gate set the fidelity index belongs to, used to look up the gate's
            :attr:`~.Gate.math_label`.
        fidelity_index: The fidelity index to label.
        style: Either ``"transition"`` (shows input :math:`\to` output Pauli) or ``"formula"``
            (shows index data ``pauli``, ``in_bit_indices``, ``out_bit_indices``).
        noise_site: An optional mapping from gate name to ``"before"`` or ``"after"``, indicating
            the gate noise is modelled as a Pauli-channel occuring either before or after the gate.
            This simplifies the ``style="formula"`` label to only display the Pauli passing through
            the Pauli channel.

    Returns:
        A math-mode LaTeX label.

    Raises:
        ValueError: If ``style`` is not ``"transition"`` or ``"formula"``.
    """
    gate_sym = gate_set[fidelity_index.gate_name].math_label

    if style == "transition":
        in_pauli, out_pauli = fidelity_index.transition
        in_str = _qubit_sparse_pauli_math_label(in_pauli)
        out_str = _qubit_sparse_pauli_math_label(out_pauli)
        return rf"{in_str} \xrightarrow{{{gate_sym}}} {out_str}"
    elif style == "formula":
        if noise_site is not None and fidelity_index.gate_name in noise_site:
            pauli = (
                fidelity_index.transition[0]
                if noise_site[fidelity_index.gate_name] == "before"
                else fidelity_index.transition[1]
            )
            pauli_str = _qubit_sparse_pauli_math_label(pauli)
            return rf"f^{{{gate_sym}}}_{{{pauli_str}}}"

        pauli_str = _qubit_sparse_pauli_math_label(fidelity_index.pauli)
        parts = [pauli_str]
        if fidelity_index.in_bit_indices:
            in_bits = (
                r"\{" + ",".join(str(i) for i in sorted(fidelity_index.in_bit_indices)) + r"\}"
            )
            parts.append(rf"b_{{in}}={in_bits}")
        if fidelity_index.out_bit_indices:
            out_bits = (
                r"\{" + ",".join(str(i) for i in sorted(fidelity_index.out_bit_indices)) + r"\}"
            )
            parts.append(rf"b_{{out}}={out_bits}")
        return rf"f^{{{gate_sym}}}(" + r",\, ".join(parts) + r")"
    else:
        raise ValueError(f"Invalid style: {style!r}. Must be 'transition' or 'formula'.")


def path_math_label(
    gate_set: GateSet,
    path: Path,
    style: Literal["transition", "formula"] = "transition",
    noise_site: Mapping[str, Literal["before", "after"]] | None = None,
    repeatable_only: bool = False,
) -> str:
    r"""Return a math-mode LaTeX label for a path.

    Args:
        gate_set: The gate set the path's fidelity indices belong to, used to look up each gate's
            :attr:`~.Gate.math_label`.
        path: The path to label.
        style: The style to use for each fidelity index label. ``"transition"`` displays the
            Pauli operator transitions induced by each gate, and ``"formula"`` displays the
            fidelity label formula associated with this path.
        noise_site: An optional noise-site mapping forwarded to :func:`fidelity_index_math_label`
            for the ``"formula"`` style (see that function for details).
        repeatable_only: If ``True``, only render the repeatable fragment without brackets
            or depth exponent.

    Returns:
        A math-mode LaTeX label.
    """
    if repeatable_only:
        return _fragment_math_label(gate_set, path.repeatable_fragment, style, noise_site)

    parts = []

    if path.start_fragment:
        parts.append(_fragment_math_label(gate_set, path.start_fragment, style, noise_site))

    if path.repeatable_fragment:
        rep_str = _fragment_math_label(gate_set, path.repeatable_fragment, style, noise_site)
        depth_str = str(path.depth) if path.depth is not None else "r"
        parts.append(f"[{rep_str}]^{{{depth_str}}}")

    if path.end_fragment:
        parts.append(_fragment_math_label(gate_set, path.end_fragment, style, noise_site))

    delimiter = r" \rightarrow " if style == "transition" else ""
    return delimiter.join(parts)


def _fragment_math_label(
    gate_set: GateSet,
    fragment: list[FidelityIndex],
    style: str,
    noise_site: Mapping[str, str] | None,
) -> str:
    """Return a math-mode LaTeX label for a single fragment of a path."""
    if style != "transition":
        counts = Counter(fragment)
        parts = []
        seen = set()
        for fi in fragment:
            if fi in seen:
                continue
            seen.add(fi)
            sym = fidelity_index_math_label(gate_set, fi, style=style, noise_site=noise_site)
            count = counts[fi]
            if count > 1:
                sym = f"{{{sym}}}^{{{count}}}"
            parts.append(sym)
        return "".join(parts)

    chains = []
    current_chain = []
    for fi in fragment:
        in_pauli, out_pauli = fi.transition
        gate_sym = gate_set[fi.gate_name].math_label
        if current_chain and current_chain[-1][1] == in_pauli:
            current_chain.append((in_pauli, out_pauli, gate_sym))
        else:
            if current_chain:
                chains.append(current_chain)
            current_chain = [(in_pauli, out_pauli, gate_sym)]
    if current_chain:
        chains.append(current_chain)

    chain_strs = []
    for arrow_chain in chains:
        parts = [_qubit_sparse_pauli_math_label(arrow_chain[0][0])]
        for _, out_pauli, gate_sym in arrow_chain:
            out_str = _qubit_sparse_pauli_math_label(out_pauli)
            parts.append(rf"\xrightarrow{{{gate_sym}}} {out_str}")
        chain_strs.append(" ".join(parts))

    return r" \rightarrow ".join(chain_strs)


_PAULI_LABELS = {1: "Z", 2: "X", 3: "Y"}


def _qubit_sparse_pauli_math_label(pauli: QubitSparsePauli) -> str:
    """Convert a QubitSparsePauli to a math-mode LaTeX label like ``X_{0} Z_{2}``."""
    if len(pauli.paulis) == 0:
        return "I"
    parts = []
    for p, idx in zip(pauli.paulis, pauli.indices):
        parts.append(f"{_PAULI_LABELS[int(p)]}_{{{int(idx)}}}")
    return " ".join(parts)
