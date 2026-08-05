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

import json
import re

import pytest
from qiskit.circuit import Measure, QuantumCircuit, Reset
from qiskit.circuit.library import CXGate
from qiskit.transpiler import InstructionProperties, Target

from qiskit_noise_learning.gate_sets import QiskitGateSet
from qiskit_noise_learning.visualizations import gate_set_topology
from qiskit_noise_learning.visualizations.interactive_svg import InteractiveFigure, axes_gid


def _make_5q_target() -> Target:
    """Build a minimal 5-qubit Target with CX, Measure, Reset on a star topology."""
    target = Target(num_qubits=5)
    cx_props = {
        (0, 1): InstructionProperties(),
        (1, 0): InstructionProperties(),
        (1, 2): InstructionProperties(),
        (2, 1): InstructionProperties(),
        (1, 3): InstructionProperties(),
        (3, 1): InstructionProperties(),
        (1, 4): InstructionProperties(),
        (4, 1): InstructionProperties(),
    }
    target.add_instruction(CXGate(), cx_props)
    single_props = {(q,): InstructionProperties() for q in range(5)}
    target.add_instruction(Measure(), single_props)
    target.add_instruction(Reset(), single_props)
    return target


def _svg_ids(figure, prefix):
    """The ``id`` attributes of the rendered SVG that start with ``prefix``, in document order."""
    return [
        found for found in re.findall(r'id="([^"]+)"', figure.to_svg()) if found.startswith(prefix)
    ]


def _sidecar(figure):
    """The figure's JSON sidecar, as the browser receives it."""
    payload = re.search(
        r'class="qnl-figure-data">(.*?)</script>', figure.to_html(), re.DOTALL
    ).group(1)
    return json.loads(payload)


def _key_tokens(figure, dimension):
    """The tokens the given legend offers, taken from the rendered SVG."""
    return {
        found.split("|")[2]
        for found in _svg_ids(figure, "key|")
        if found.split("|")[1] == dimension
    }


def _trace_tokens(figure):
    """The ``(gate_token, activity_token)`` pairs the rendered artists carry."""
    return {tuple(found.split("|")[2:4]) for found in _svg_ids(figure, "trace|")}


@pytest.fixture()
def gate_set_5q():
    target = _make_5q_target()
    gate_set = QiskitGateSet(target=target)

    circ1 = QuantumCircuit(5)
    circ1.cx(0, 1)
    circ1.cx(2, 1)
    gate_set.add_circuit_as_gate(circ1, range(5), name="L0")

    circ2 = QuantumCircuit(5)
    circ2.cx(3, 1)
    circ2.cx(4, 1)
    gate_set.add_circuit_as_gate(circ2, range(5), name="L1")

    return gate_set


@pytest.fixture()
def topology(gate_set_5q):
    return gate_set_topology(gate_set_5q)


def test_returns_interactive_figure(topology):
    assert isinstance(topology, InteractiveFigure)


def test_draw_method(gate_set_5q):
    assert isinstance(gate_set_5q.draw(), InteractiveFigure)


def test_no_target_raises():
    gate_set = QiskitGateSet(5)
    with pytest.raises(ValueError, match="target is None"):
        gate_set.draw()


def test_draws_marks_for_every_gate(topology, gate_set_5q):
    # One legend entry per gate that drew something, and nothing left unlabelled.
    assert len(_key_tokens(topology, "path")) == len(gate_set_5q)


def test_aspect_is_held(topology):
    # The coupling graph's proportions say which qubits are neighbours; a stretched one misleads.
    assert topology.figure.axes[0].get_aspect() == 1.0


def test_axes_patch_survives_hiding_the_frame(topology):
    # Ticks and spines are hidden one by one rather than with ``set_axis_off`` precisely so that the
    # patch stays: the browser measures it to place the hover readout.
    (cell,) = _sidecar(topology)["cells"]
    assert axes_gid(cell) in _svg_ids(topology, "axes|")


def test_traces_and_legends_agree(topology):
    """This figure builds its own ids, so it earns the same contract check as the grid: distinct
    ids, and legend keys that are exactly the keys something is drawn under."""
    ids = _svg_ids(topology, "trace|")
    assert len(ids) > 1
    assert len(set(ids)) == len(ids)

    drawn = _trace_tokens(topology)
    assert _key_tokens(topology, "path") == {tokens[0] for tokens in drawn}
    assert _key_tokens(topology, "layer") == {tokens[1] for tokens in drawn}


def test_hover_text_names_the_gate_and_the_qubits(topology, gate_set_5q):
    texts = [text for trace in _sidecar(topology)["traces"].values() for text in trace["texts"]]
    assert any("cx: 0-1" in text for text in texts)
    assert any("measure: 0" in text for text in texts)
    assert any("idle: 3" in text for text in texts)
    assert all(text.startswith(tuple(gate_set_5q)) for text in texts)


def test_hover_points_are_finite(topology):
    # The polylines are joined with ``nan`` to lift the pen between marks, and ``nan`` is not JSON;
    # a leaked one would make the whole sidecar unparseable and cost the readout entirely.
    for trace in _sidecar(topology)["traces"].values():
        for point in trace["points"]:
            assert all(value == value for value in point), "NaN reached the sidecar"


def test_spam_gates_open_hidden(topology, gate_set_5q):
    # Preparation and measurement touch every qubit at once and would bury the rest of the figure,
    # so they start switched off -- but drawn, and with a legend entry, which is the whole
    # difference from leaving them out.
    hidden = set(_sidecar(topology)["hidden"]["path"])
    spam = {name for name, gate in gate_set_5q.items() if gate.prep_idxs or gate.meas_idxs}
    assert len(hidden) == len(spam)
    assert hidden <= _key_tokens(topology, "path")
    assert hidden <= {token for token, _ in _trace_tokens(topology)}


def test_ordinary_gates_open_visible(topology, gate_set_5q):
    hidden = set(_sidecar(topology)["hidden"]["path"])
    layers = {name for name, gate in gate_set_5q.items() if not (gate.prep_idxs or gate.meas_idxs)}
    assert layers
    # Every gate that is neither a preparation nor a measurement still has a legend entry, and each
    # of those entries switches something that is on to begin with.
    assert len(_key_tokens(topology, "path") - hidden) == len(layers)
