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

"""FidelityModel"""

from abc import abstractmethod
from collections import Counter
from collections.abc import Hashable
from copy import copy
from itertools import chain
from typing import Generic, TypeVar, overload

import numpy as np
from qiskit.quantum_info import QubitSparsePauli

from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.gate_sets import GateSet, ModelGateSet
from qiskit_noise_learning.math import ComposedLinearMap, IndexedVector, LinearMap, ParameterSpace
from qiskit_noise_learning.sequences import FidelityIndex, Path

from .fidelity_index_space import FidelityIndexSpace

ParameterIndex = TypeVar("ParameterIndex", bound=Hashable)


class FidelityModel(LinearMap[ParameterIndex, FidelityIndex], Generic[ParameterIndex]):
    r"""A linear parameterization of the log fidelities of a gate set.

    A :class:`FidelityModel` is a special kind of :class:`LinearMap`:
    - The output space is the :class:`FidelityIndexSpace` of a gate set; i.e. it represents linear
      map :math:`A` for which :math:`-\log(f) = Ar`, where :math:`f` is the vector of fidelities,
      and :math:`r` is the model parameter space.
    - The type is preserved under composition with other :class:`LinearMap`\s.

    The latter point allows concrete subclasses representing specific classes of fidelity models to
    maintain control over specialized behaviour (e.g. fidelity renderings in specific contexts).

    Args:
        gate_set: The gate set for which the fidelities are modelled, to be converted to a
            :class:`ModelGateSet`.
    """

    def __init__(self, gate_set: GateSet, input_space: ParameterSpace[ParameterIndex]):
        self._gate_set = gate_set.model_gate_set
        super().__init__(input_space=input_space, output_space=FidelityIndexSpace(self._gate_set))
        self._pre_map = None
        self._post_map = None

    @property
    def gate_set(self) -> ModelGateSet:
        return self._gate_set

    def row(self, output_index: FidelityIndex) -> IndexedVector[ParameterIndex]:
        """Get a row of the log fidelity parameterization matrix.

        Args:
            output_index: The fidelity index for the row.
        """
        if self._post_map is not None:
            core_result = self._core_left_multiply(self._post_map.row(output_index))
        else:
            core_result = self._core_row(output_index)

        if self._pre_map is not None:
            return self._pre_map.left_multiply(core_result)

        return core_result

    @abstractmethod
    def _core_row(self, fidelity_index: FidelityIndex) -> IndexedVector:
        """Return the row for a fidelity index without composition.

        Subclasses implement this to define the core linear map.
        """

    def _core_left_multiply(self, vector: IndexedVector[FidelityIndex]) -> IndexedVector:
        """Left-multiply a sparse vector against the core map."""
        result = IndexedVector()
        for idx, coeff in vector.items():
            result = result + coeff * self._core_row(idx)
        return result

    def compose(self, outer: "LinearMap[FidelityIndex, FidelityIndex]") -> "FidelityModel":
        """Post-compose with a fidelity transformation.

        Returns a new model of the same type with the transformation applied after the core map.

        Args:
            outer: A linear map on the fidelity index space.
        """
        new_model = copy(self)
        if self._post_map is not None:
            new_model._post_map = ComposedLinearMap(  # noqa: SLF001
                inner=self._post_map, outer=outer
            )
        else:
            new_model._post_map = outer  # noqa: SLF001
        return new_model

    def pre_compose(self, inner: "LinearMap") -> "FidelityModel":
        """Pre-compose with a parameter transformation.

        Returns a new model of the same type with the transformation applied before the core map.

        Args:
            inner: A linear map whose output space matches the model's native parameter space.
        """
        new_model = copy(self)
        if self._pre_map is not None:
            new_model._pre_map = ComposedLinearMap(  # noqa: SLF001
                inner=inner, outer=self._pre_map
            )
        else:
            new_model._pre_map = inner  # noqa: SLF001
        new_model._input_space = inner.input_space  # noqa: SLF001
        return new_model

    def row_from_fidelity(self, fidelity_index: FidelityIndex) -> IndexedVector[ParameterIndex]:
        """Get a row of the log fidelity parameterization matrix.

        Alias for :meth:`row`.

        Args:
            fidelity_index: The fidelity index for the row.
        """
        return self.row(fidelity_index)

    def row_from_path(self, path: Path) -> IndexedVector[ParameterIndex]:
        """Get the design matrix row generated by a path.

        If the path is unbound (``depth`` is ``None``), returns the row for the repeatable fragment
        only. If the path is bound (``depth`` is an integer), returns the full row including the
        start and end fragments scaled by the depth.

        Args:
            path: The path.
        """
        vector = IndexedVector[ParameterIndex](dict())

        for fidelity_index in path.repeatable_fragment:
            vector += self.row(fidelity_index)

        if path.depth is None:
            return vector

        vector = path.depth * vector

        for fidelity_index in chain(path.start_fragment, path.end_fragment):
            vector += self.row(fidelity_index)

        return vector

    @overload
    def log_fidelity_estimate(self, index: FidelityIndex, model_data: ModelData) -> float: ...

    @overload
    def log_fidelity_estimate(self, index: Path, model_data: ModelData) -> float: ...

    def log_fidelity_estimate(self, index: FidelityIndex | Path, model_data: ModelData) -> float:
        """Return the log-fidelity estimate for a fidelity index or path given model parameters.

        Computes the dot product ``row . params`` where ``row`` is the design matrix row for
        ``index``. For a :class:`~.Path`, the semantics follow :meth:`row_from_path`: unbound paths
        give the per-repetition log-fidelity (repeatable fragment only), bound paths give the total
        log-fidelity including start and end fragments.

        Args:
            index: A fidelity index or path.
            model_data: The fitted model parameters.

        Returns:
            The log-fidelity estimate.

        Raises:
            ValueError: If a parameter index from the design matrix row is not present in
                ``model_data``.
        """
        row = self.row_from_path(index) if isinstance(index, Path) else self.row(index)

        params = model_data.dataset["parameter_values"]
        param_labels = params.coords["parameter"].values
        param_index_map = {label: pos for pos, label in enumerate(param_labels)}
        param_values = params.values

        dot = 0.0
        for idx, coeff in row.items():
            if idx not in param_index_map:
                raise ValueError(f"Parameter index {idx} not found in ModelData.")
            dot += coeff * param_values[param_index_map[idx]]

        return dot

    @overload
    def fidelity_estimate(self, index: FidelityIndex, model_data: ModelData) -> float: ...

    @overload
    def fidelity_estimate(self, index: Path, model_data: ModelData) -> float: ...

    def fidelity_estimate(self, index: FidelityIndex | Path, model_data: ModelData) -> float:
        """Return the fidelity estimate for a fidelity index or path given model parameters.

        Computes ``exp(-log_fidelity_estimate(index, model_data))``. See
        :meth:`log_fidelity_estimate` for details on the semantics.

        Args:
            index: A fidelity index or path.
            model_data: The fitted model parameters.

        Returns:
            The fidelity estimate.
        """
        return float(np.exp(-self.log_fidelity_estimate(index, model_data)))

    def fidelity_index_latex_str(
        self,
        fidelity_index: FidelityIndex,
        format: str = "transition",
    ) -> str:
        r"""Return a LaTeX string for a fidelity index.

        Args:
            fidelity_index: The fidelity index to label.
            format: Either ``"transition"`` (shows input :math:`\to` output Pauli) or ``"formula"``
                (shows index data ``pauli``, ``in_bit_indices``, ``out_bit_indices``).

        Returns:
            A LaTeX string.

        Raises:
            ValueError: If ``format`` is not ``"transition"`` or ``"formula"``.
        """
        gate_sym = self._gate_set[fidelity_index.gate_name].latex_str

        if format == "transition":
            in_pauli, out_pauli = fidelity_index.transition
            in_str = _qubit_sparse_pauli_to_latex(in_pauli)
            out_str = _qubit_sparse_pauli_to_latex(out_pauli)
            return rf"{in_str} \xrightarrow{{{gate_sym}}} {out_str}"
        elif format == "formula":
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
        else:
            raise ValueError(f"Invalid format: {format!r}. Must be 'transition' or 'index'.")

    def path_latex_str(
        self, path: Path, format: str = "transition", repeatable_only: bool = False
    ) -> str:
        r"""Return a LaTeX string for a path.

        Args:
            path: The path to label.
            format: The format to use for each fidelity index label. ``"transition"`` displays the
                Pauli operator transitions induced by each gate, and ``"formula"`` displays the
                fidelity label formula associated with this path.
            repeatable_only: If ``True``, only render the repeatable fragment without brackets
                or depth exponent.

        Returns:
            A LaTeX string.
        """
        if repeatable_only:
            return self._fragment_latex_str(path.repeatable_fragment, format)

        parts = []

        if path.start_fragment:
            parts.append(self._fragment_latex_str(path.start_fragment, format))

        if path.repeatable_fragment:
            rep_str = self._fragment_latex_str(path.repeatable_fragment, format)
            depth_str = str(path.depth) if path.depth is not None else "r"
            parts.append(f"[{rep_str}]^{{{depth_str}}}")

        if path.end_fragment:
            parts.append(self._fragment_latex_str(path.end_fragment, format))

        delimiter = r" \rightarrow " if format == "transition" else ""
        return delimiter.join(parts)

    def _fragment_latex_str(self, fragment: list[FidelityIndex], format: str) -> str:
        """Return a LaTeX string for a single fragment of a path."""
        if format != "transition":
            counts = Counter(fragment)
            parts = []
            seen = set()
            for fi in fragment:
                if fi in seen:
                    continue
                seen.add(fi)
                sym = self.fidelity_index_latex_str(fi, format=format)
                count = counts[fi]
                if count > 1:
                    sym = f"{{{sym}}}^{{{count}}}"
                parts.append(sym)
            return "".join(parts)

        chains = []
        current_chain = []
        for fi in fragment:
            in_pauli, out_pauli = fi.transition
            gate_sym = self._gate_set[fi.gate_name].latex_str
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


_PAULI_LABELS = {1: "Z", 2: "X", 3: "Y"}


def _qubit_sparse_pauli_to_latex(pauli: QubitSparsePauli) -> str:
    """Convert a QubitSparsePauli to a LaTeX string like ``X_{0} Z_{2}``."""
    if len(pauli.paulis) == 0:
        return "I"
    parts = []
    for p, idx in zip(pauli.paulis, pauli.indices):
        parts.append(f"{_PAULI_LABELS[int(p)]}_{{{int(idx)}}}")
    return " ".join(parts)
