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

"""LaTeX rendering for fidelity indices and paths."""

from collections import Counter

from qiskit.quantum_info import QubitSparsePauli

from qiskit_noise_learning.models import FidelityModel, PauliLindbladModel
from qiskit_noise_learning.sequences import FidelityIndex, Path

_PAULI_LABELS = {1: "Z", 2: "X", 3: "Y"}


def fidelity_index_latex_str(
    fidelity_index: FidelityIndex,
    model: FidelityModel,
    format: str = "transition",
) -> str:
    r"""Return a LaTeX string for a fidelity index of a model.

    For the ``"formula"`` format, if ``model`` is (or contains, in the case of a
    :class:`~.ComposedFidelityModel`) a :class:`~.PauliLindbladModel`, a simplified label
    :math:`f^{G}_{P}` is produced, where :math:`G` is the gate symbol and :math:`P` is the Pauli at
    the model's noise site. Otherwise the generic formula listing the index data is used.

    Args:
        fidelity_index: The fidelity index to label.
        model: The fidelity model the index belongs to.
        format: Either ``"transition"`` (shows input :math:`\to` output Pauli) or ``"formula"``
            (shows the fidelity label formula).

    Returns:
        A LaTeX string.

    Raises:
        ValueError: If ``format`` is not ``"transition"`` or ``"formula"``.
    """
    gate_name = fidelity_index.gate_name

    if format == "transition":
        gate_sym = model.gate_set[gate_name].latex_str
        in_pauli, out_pauli = fidelity_index.transition
        in_str = _qubit_sparse_pauli_to_latex(in_pauli)
        out_str = _qubit_sparse_pauli_to_latex(out_pauli)
        return rf"{in_str} \xrightarrow{{{gate_sym}}} {out_str}"

    if format == "formula":
        pauli_lindblad_model = _find_pauli_lindblad_model(model)
        if pauli_lindblad_model is not None:
            gate_sym = pauli_lindblad_model.gate_set[gate_name].latex_str
            pauli = (
                fidelity_index.transition[0]
                if pauli_lindblad_model.noise_site[gate_name] == "before"
                else fidelity_index.transition[1]
            )
            return rf"f^{{{gate_sym}}}_{{{_qubit_sparse_pauli_to_latex(pauli)}}}"

        gate_sym = model.gate_set[gate_name].latex_str
        pauli_str = _qubit_sparse_pauli_to_latex(fidelity_index.pauli)
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

    raise ValueError(f"Invalid format: {format!r}. Must be 'transition' or 'formula'.")


def path_latex_str(
    path: Path,
    model: FidelityModel,
    format: str = "transition",
    repeatable_only: bool = False,
) -> str:
    r"""Return a LaTeX string for a path.

    Args:
        path: The path to label.
        model: The fidelity model the path's indices belong to.
        format: The format to use for each fidelity index label. ``"transition"`` displays the
            Pauli operator transitions induced by each gate, and ``"formula"`` displays the
            fidelity label formula associated with this path.
        repeatable_only: If ``True``, only render the repeatable fragment without brackets
            or depth exponent.

    Returns:
        A LaTeX string.
    """
    if repeatable_only:
        return _fragment_latex_str(path.repeatable_fragment, model, format)

    parts = []

    if path.start_fragment:
        parts.append(_fragment_latex_str(path.start_fragment, model, format))

    if path.repeatable_fragment:
        rep_str = _fragment_latex_str(path.repeatable_fragment, model, format)
        depth_str = str(path.depth) if path.depth is not None else "r"
        parts.append(f"[{rep_str}]^{{{depth_str}}}")

    if path.end_fragment:
        parts.append(_fragment_latex_str(path.end_fragment, model, format))

    delimiter = r" \rightarrow " if format == "transition" else ""
    return delimiter.join(parts)


def _find_pauli_lindblad_model(model: FidelityModel) -> PauliLindbladModel | None:
    """Return a :class:`~.PauliLindbladModel` from ``model`` or its composition chain, if any."""
    if isinstance(model, PauliLindbladModel):
        return model
    for sub_map in getattr(model, "maps", []):
        found = _find_pauli_lindblad_model(sub_map)
        if found is not None:
            return found
    return None


def _fragment_latex_str(fragment: list[FidelityIndex], model: FidelityModel, format: str) -> str:
    """Return a LaTeX string for a single fragment of a path."""
    if format != "transition":
        counts = Counter(fragment)
        parts = []
        seen = set()
        for fi in fragment:
            if fi in seen:
                continue
            seen.add(fi)
            sym = fidelity_index_latex_str(fi, model, format=format)
            count = counts[fi]
            if count > 1:
                sym = f"{{{sym}}}^{{{count}}}"
            parts.append(sym)
        return "".join(parts)

    chains = []
    current_chain = []
    for fi in fragment:
        in_pauli, out_pauli = fi.transition
        gate_sym = model.gate_set[fi.gate_name].latex_str
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
        parts = [_qubit_sparse_pauli_to_latex(arrow_chain[0][0])]
        for _, out_pauli, gate_sym in arrow_chain:
            out_str = _qubit_sparse_pauli_to_latex(out_pauli)
            parts.append(rf"\xrightarrow{{{gate_sym}}} {out_str}")
        chain_strs.append(" ".join(parts))

    return r" \rightarrow ".join(chain_strs)


def _qubit_sparse_pauli_to_latex(pauli: QubitSparsePauli) -> str:
    """Convert a QubitSparsePauli to a LaTeX string like ``X_{0} Z_{2}``."""
    if len(pauli.paulis) == 0:
        return "I"
    parts = []
    for p, idx in zip(pauli.paulis, pauli.indices):
        parts.append(f"{_PAULI_LABELS[int(p)]}_{{{int(idx)}}}")
    return " ".join(parts)
