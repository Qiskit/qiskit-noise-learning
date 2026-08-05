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
from qiskit_noise_learning.visualizations.interactive_figure import InteractiveFigure, axes_gid


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


def _make_line_target(num_qubits: int) -> Target:
    """Build a Target with CX along a line, Measure and Reset everywhere."""
    target = Target(num_qubits=num_qubits)
    cx_props = {}
    for qubit in range(num_qubits - 1):
        cx_props[(qubit, qubit + 1)] = InstructionProperties()
        cx_props[(qubit + 1, qubit)] = InstructionProperties()
    target.add_instruction(CXGate(), cx_props)
    single_props = {(q,): InstructionProperties() for q in range(num_qubits)}
    target.add_instruction(Measure(), single_props)
    target.add_instruction(Reset(), single_props)
    return target


def _make_unconstrained_target(num_qubits: int) -> Target:
    """Build a Target whose CX is globally applicable, so it constrains connectivity nowhere and
    ``Target.build_coupling_map()`` returns ``None``."""
    target = Target(num_qubits=num_qubits)
    target.add_instruction(CXGate(), None)
    single_props = {(q,): InstructionProperties() for q in range(num_qubits)}
    target.add_instruction(Measure(), single_props)
    target.add_instruction(Reset(), single_props)
    return target


def _gate_set_with_one_layer(target: Target) -> QiskitGateSet:
    """A gate set over ``target`` carrying one two-qubit layer, so something is drawn."""
    gate_set = QiskitGateSet(target=target)
    circuit = QuantumCircuit(target.num_qubits)
    circuit.cx(0, 1)
    gate_set.add_circuit_as_gate(circuit, range(target.num_qubits), name="L0")
    return gate_set


#: What this figure legends itself by, in the order a trace's id carries their tokens.
_GATE_DIMENSION, _ACTIVITY_DIMENSION = "gate", "activity"


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
    return {tuple(found.split("|")[2:-1]) for found in _svg_ids(figure, "trace|")}


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


def test_draws_the_device_topology_under_the_gates(topology):
    # The device's own edges are the one thing drawn that belongs to no gate, hence the only line
    # without a gid. The contrast is with an unconstrained target, which has no topology at all.
    assert any(line.get_gid() is None for line in topology.figure.axes[0].get_lines())


def test_unconstrained_target_draws_without_topology_edges():
    # A target whose two-qubit gate is globally applicable constrains connectivity nowhere, so there
    # is no coupling graph: an ideal device to draw without edges, not a failure. Its size still has
    # a conventional layout, so the qubits go where a reader expects them.
    figure = gate_set_topology(_gate_set_with_one_layer(_make_unconstrained_target(5)))
    axes = figure.figure.axes[0]
    assert len(axes.patches) == 5  # one node circle per qubit
    assert all(line.get_gid() for line in axes.get_lines())


def test_unconventional_size_is_laid_out_from_the_coupling_graph():
    # No conventional layout for 4 qubits, so the positions come from the coupling graph. Each qubit
    # still gets its own node, which is what the rest of the drawing indexes by qubit.
    figure = gate_set_topology(_gate_set_with_one_layer(_make_line_target(4)))
    axes = figure.figure.axes[0]
    assert len({tuple(patch.get_center()) for patch in axes.patches}) == 4
    assert any(line.get_gid() is None for line in axes.get_lines())


def test_unconventional_size_without_connectivity_raises():
    # Neither a conventional layout for this size nor a graph to derive one from.
    gate_set = _gate_set_with_one_layer(_make_unconstrained_target(4))
    with pytest.raises(ValueError, match="no coupling map"):
        gate_set.draw()


def test_draws_marks_for_every_gate(topology, gate_set_5q):
    # One legend entry per gate that drew something, and nothing left unlabelled.
    assert len(_key_tokens(topology, _GATE_DIMENSION)) == len(gate_set_5q)


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
    assert _key_tokens(topology, _GATE_DIMENSION) == {tokens[0] for tokens in drawn}
    assert _key_tokens(topology, _ACTIVITY_DIMENSION) == {tokens[1] for tokens in drawn}


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
    hidden = set(_sidecar(topology)["hidden"][_GATE_DIMENSION])
    spam = {name for name, gate in gate_set_5q.items() if gate.prep_idxs or gate.meas_idxs}
    assert len(hidden) == len(spam)
    assert hidden <= _key_tokens(topology, _GATE_DIMENSION)
    assert hidden <= {token for token, _ in _trace_tokens(topology)}


def test_ordinary_gates_open_visible(topology, gate_set_5q):
    hidden = set(_sidecar(topology)["hidden"][_GATE_DIMENSION])
    ordinary = {
        name for name, gate in gate_set_5q.items() if not (gate.prep_idxs or gate.meas_idxs)
    }
    assert ordinary
    # Every gate that is neither a preparation nor a measurement still has a legend entry, and each
    # of those entries switches something that is on to begin with.
    assert len(_key_tokens(topology, _GATE_DIMENSION) - hidden) == len(ordinary)
