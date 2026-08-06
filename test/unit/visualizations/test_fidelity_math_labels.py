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

from itertools import chain

import pytest
from matplotlib.mathtext import MathTextParser
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import FidelityIndex, Path
from qiskit_noise_learning.visualizations import fidelity_index_math_label, path_math_label


@pytest.fixture()
def gate_set():
    model_gate_set = ModelGateSet(2)
    ident_2q = Clifford(QuantumCircuit(2))
    model_gate_set.add_gate(
        ModelGate("CZ", [((0, 1), ident_2q)], qubit_idxs=[0, 1], latex_str=r"\mathrm{CZ}")
    )
    return model_gate_set


@pytest.fixture()
def fidelity_index(gate_set):
    return FidelityIndex.from_gate(
        gate=gate_set["CZ"],
        pauli=QubitSparsePauli.from_sparse_label(("X", [0]), num_qubits=2),
        in_z_idxs=frozenset(),
        out_z_idxs=frozenset(),
    )


class TestFidelityIndexMathLabel:
    def test_transition_format(self, gate_set, fidelity_index):
        result = fidelity_index_math_label(gate_set, fidelity_index, style="transition")
        assert isinstance(result, str)

    def test_formula_format(self, gate_set, fidelity_index):
        result = fidelity_index_math_label(gate_set, fidelity_index, style="formula")
        assert isinstance(result, str)

    def test_invalid_format_raises(self, gate_set, fidelity_index):
        with pytest.raises(ValueError):
            fidelity_index_math_label(gate_set, fidelity_index, style="bad")

    def test_qubit_labels_remap_subscript(self, gate_set, fidelity_index):
        # The X on qubit 0 renders as X_{i} once qubit 0 is relabeled to "i".
        labeled = fidelity_index_math_label(
            gate_set, fidelity_index, style="formula", qubit_labels={0: "i"}
        )
        assert "X_{i}" in labeled
        assert "X_{0}" not in labeled

    def test_qubit_labels_partial_falls_back_to_index(self, gate_set, fidelity_index):
        # An index absent from the map renders as its integer value.
        labeled = fidelity_index_math_label(
            gate_set, fidelity_index, style="formula", qubit_labels={5: "z"}
        )
        assert "X_{0}" in labeled

    def test_formula_z_exponents_use_formalism_symbols(self, gate_set_cz):
        r"""The :math:`Z` exponents render as :math:`x` and :math:`y`, as in the formalism.

        A measurement carrying :math:`Z` on qubit 0 at its input and :math:`Z` on both qubits at its
        output labels as ``f^{M}(I,\, x=\{0\},\, y=\{0,1\})``, with the exponents shown as the qubit
        indices on which they are non-zero.
        """
        fidelity_index = FidelityIndex.from_gate(
            gate=gate_set_cz["M"],
            pauli=QubitSparsePauli("II"),
            in_z_idxs=frozenset([0]),
            out_z_idxs=frozenset([0, 1]),
        )

        label = fidelity_index_math_label(gate_set_cz, fidelity_index, style="formula")
        assert r"x=\{0\}" in label
        assert r"y=\{0,1\}" in label

        # the exponent index sets honor the qubit relabeling map
        labeled = fidelity_index_math_label(
            gate_set_cz, fidelity_index, style="formula", qubit_labels={0: "i", 1: "j"}
        )
        assert r"x=\{i\}" in labeled
        assert r"y=\{i,j\}" in labeled


class TestPathMathLabel:
    def test_transition_format(self, gate_set, fidelity_index):
        path = Path(
            start_fragment=[fidelity_index],
            repeatable_fragment=[fidelity_index, fidelity_index],
            end_fragment=[fidelity_index],
            fragment_depth=3,
        )
        result = path_math_label(gate_set, path, style="transition")
        assert isinstance(result, str)

    def test_formula_format(self, gate_set, fidelity_index):
        path = Path(
            start_fragment=[],
            repeatable_fragment=[fidelity_index, fidelity_index],
            end_fragment=[],
            fragment_depth=5,
        )
        result = path_math_label(gate_set, path, style="formula")
        assert isinstance(result, str)

    def test_qubit_labels_remap_subscript(self, gate_set, fidelity_index):
        path = Path(
            start_fragment=[],
            repeatable_fragment=[fidelity_index],
            end_fragment=[],
            fragment_depth=5,
        )
        labeled = path_math_label(gate_set, path, style="formula", qubit_labels={0: "i"})
        assert "X_{i}" in labeled
        assert "X_{0}" not in labeled

    def test_repeatable_only(self, gate_set, fidelity_index):
        path = Path(
            start_fragment=[fidelity_index],
            repeatable_fragment=[fidelity_index],
            end_fragment=[fidelity_index],
            fragment_depth=3,
        )
        result = path_math_label(gate_set, path, repeatable_only=True)
        assert isinstance(result, str)


_STYLES = ["transition", "formula"]
_NOISE_SITES = [None, {"CZ": "before"}, {"CZ": "after"}]
_QUBIT_LABELS = [None, {0: "i", 1: "j"}]


def _assert_renders(label):
    """Assert a label typesets, wrapped in ``$...$`` exactly as the plotting code wraps it.

    These labels are typeset by matplotlib's mathtext, whose macro coverage is narrower than a real
    LaTeX installation's, so a label can be valid LaTeX and still fail to render.
    """
    assert isinstance(label, str)
    try:
        MathTextParser("path").parse(f"${label}$")
    except Exception as exc:  # noqa: BLE001 - surface the offending label, whatever the failure
        pytest.fail(f"mathtext cannot render {label!r}: {exc}")


class TestMathtextRenderability:
    """Every label the package can generate must typeset under matplotlib's mathtext.

    The interactive figures and the static exports share one math engine, so an unsupported macro
    shows up as a broken legend in both. Before these labels were typeset in-process the only signal
    for that was a browser console warning, which nothing could check.
    """

    @pytest.fixture()
    def paths(self, make_cz_path):
        """Structurally varied real paths over the ``CZ`` orbit.

        ``CZ`` is an involution, so its orbits are at most two Paulis long and the interesting
        variation is elsewhere: weight-one vs weight-two Paulis, an orbit ``CZ`` leaves fixed (whose
        two transitions are identical, so the formula style collapses them to an exponent), the
        explicit single-node orbit form, and a path with no SPAM fragments.
        """
        return [
            make_cz_path("XI"),
            make_cz_path("XX"),
            make_cz_path("ZI"),
            make_cz_path(["ZZ"]),
            make_cz_path("XI", spam=False),
        ]

    @pytest.mark.parametrize("style", _STYLES)
    @pytest.mark.parametrize("noise_site", _NOISE_SITES)
    @pytest.mark.parametrize("qubit_labels", _QUBIT_LABELS)
    def test_fidelity_index_labels_render(
        self, gate_set_cz, paths, style, noise_site, qubit_labels
    ):
        indices = list(
            chain.from_iterable(
                chain(p.start_fragment, p.repeatable_fragment, p.end_fragment) for p in paths
            )
        )
        assert indices, "fixture produced no fidelity indices to check"
        for fidelity_index in indices:
            _assert_renders(
                fidelity_index_math_label(
                    gate_set_cz,
                    fidelity_index,
                    style=style,
                    noise_site=noise_site,
                    qubit_labels=qubit_labels,
                )
            )

    @pytest.mark.parametrize("style", _STYLES)
    @pytest.mark.parametrize("noise_site", _NOISE_SITES)
    @pytest.mark.parametrize("qubit_labels", _QUBIT_LABELS)
    @pytest.mark.parametrize("repeatable_only", [True, False])
    def test_path_labels_render(
        self, gate_set_cz, paths, style, noise_site, qubit_labels, repeatable_only
    ):
        for path in paths:
            _assert_renders(
                path_math_label(
                    gate_set_cz,
                    path,
                    style=style,
                    noise_site=noise_site,
                    repeatable_only=repeatable_only,
                    qubit_labels=qubit_labels,
                )
            )

    def test_bound_fragment_depth_exponent_renders(self, gate_set_cz, paths):
        """An unbound path labels its exponent ``r``; a bound one substitutes the integer depth."""
        for path in paths:
            for bound in (path, path.bind_at(3)):
                _assert_renders(path_math_label(gate_set_cz, bound, style="transition"))

    def test_latex_str_gate_symbol_renders(self, gate_set, fidelity_index):
        """A gate carrying an explicit ``latex_str`` (rather than the ``\\text{...}`` fallback)."""
        for style in _STYLES:
            _assert_renders(fidelity_index_math_label(gate_set, fidelity_index, style=style))


class TestNoiseSiteMathLabel:
    @pytest.mark.parametrize("site", ["before", "after"])
    def test_formula_format(self, gate_set, fidelity_index, site):
        result = fidelity_index_math_label(
            gate_set, fidelity_index, style="formula", noise_site={"CZ": site}
        )
        assert isinstance(result, str)

    def test_transition_format(self, gate_set, fidelity_index):
        result = fidelity_index_math_label(
            gate_set, fidelity_index, style="transition", noise_site={"CZ": "before"}
        )
        assert isinstance(result, str)
