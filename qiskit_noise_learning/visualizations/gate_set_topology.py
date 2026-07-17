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

"""Gate set topology visualization."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..optionals import HAS_PLOTLY

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ..gate_sets.gate import Gate
    from ..gate_sets.gate_set import GateSet

_NODE_RADIUS = 0.20  # data-unit radius of qubit node circles
_ARC_RADIUS = 0.24  # data-unit radius of gate arcs (must be > _NODE_RADIUS)
_ARC_NPTS = 50  # number of points per arc segment
_ARC_GAP = 0.08  # fraction of sector span to leave as a visual gap on each side
_EDGE_OFFSET = 0.14  # data-unit perpendicular offset between parallel edges

_COLOR_BG_EDGE = "rgb(180,180,180)"  # unoccupied topology edges
_COLOR_NODE_ACTIVE = "rgb(60,60,60)"  # nodes within qubit_subset
_COLOR_NODE_INACTIVE = "rgb(180,180,180)"  # nodes outside qubit_subset
_COLOR_LABEL = "rgb(210,210,220)"  # qubit index labels inside nodes
_COLOR_FIGURE_BG = "white"  # figure and plot background


def _arc_segments(
    qubits: frozenset[int],
    xs: list[float],
    ys: list[float],
    qubit_to_all_gates: dict[int, list[str]],
    gate_name: str,
    label_fn: Callable[[int], str],
) -> tuple[list[float | None], list[float | None], list[str | None]]:
    """Return (arc_x, arc_y, hover) for arc segments drawn around the given qubits.

    Sector sector_idx of num_sectors is centred at angle 2π·sector_idx/num_sectors
    (sector_idx=0 → right side), proceeding ccw.
    """
    arc_x: list[float | None] = []
    arc_y: list[float | None] = []
    hover: list[str | None] = []
    for qubit in sorted(qubits):
        gates_on_qubit = qubit_to_all_gates[qubit]
        sector_idx = gates_on_qubit.index(gate_name)
        num_sectors = len(gates_on_qubit)
        center = 2 * math.pi * sector_idx / num_sectors
        half = math.pi / num_sectors * (1 - _ARC_GAP)
        angle_start, angle_end = center - half, center + half
        angles = [
            angle_start + (angle_end - angle_start) * idx / (_ARC_NPTS - 1)
            for idx in range(_ARC_NPTS)
        ]
        arc_x.extend(xs[qubit] + _ARC_RADIUS * math.cos(angle) for angle in angles)
        arc_y.extend(ys[qubit] + _ARC_RADIUS * math.sin(angle) for angle in angles)
        arc_x.append(None)
        arc_y.append(None)
        label = label_fn(qubit)
        hover.extend(label for _ in angles)
        hover.append(None)
    return arc_x, arc_y, hover


@HAS_PLOTLY.require_in_call
def gate_set_topology(gate_set: GateSet[Gate]) -> go.Figure:
    """Draw the device topology with per-gate coloring.

    Gates with 2-qubit interactions are drawn as colored edges on the device
    coupling graph. Gates that act only on individual qubits (such as preparation
    and measurement) are shown as colored arcs around the relevant nodes, with one
    arc sector per gate per qubit. The arc sectors are arranged so that the first
    gate in the set occupies the right-hand side of the circle (angle 0), and
    subsequent gates proceed counter-clockwise. Qubits that are idling in a given
    gate receive a slightly transparent arc.

    Args:
        gate_set: The gate set to visualize. Must have a non-``None``:attr:`~.GateSet.target` so
            that qubit coordinates and the device topology can be determined.

    Returns:
        A plotly Figure.

    Raises:
        ValueError: If ``gate_set.target`` is ``None``.
        ImportError: If ``plotly`` or ``qiskit-ibm-runtime`` is not installed.
    """
    import plotly.colors as pc
    import plotly.graph_objects as go

    if gate_set.target is None:
        raise ValueError(
            "Cannot draw gate set topology: gate_set.target is None. "
            "A Target is required to determine qubit coordinates and device connectivity."
        )

    try:
        from qiskit_ibm_runtime.visualization.embeddings import _get_qubits_coordinates
    except ImportError as exc:
        raise ImportError(
            "qiskit-ibm-runtime is required for gate set topology visualization."
        ) from exc

    raw_coords = _get_qubits_coordinates(gate_set.num_qubits)
    xs = [float(col) for _, col in raw_coords]
    ys = [float(-row) for row, _ in raw_coords]

    topo_edges = {
        (min(q1, q2), max(q1, q2)) for q1, q2 in gate_set.target.build_coupling_map().get_edges()
    }

    palette = pc.qualitative.Plotly
    gate_names = list(gate_set)
    gate_colors = {name: palette[idx % len(palette)] for idx, name in enumerate(gate_names)}

    gate_labels = {name: gate.label for name, gate in gate_set.items()}

    # for each gate: edge_type_pairs holds 2-qubit pairs (drawn as colored edges);
    # arc_type_active holds single-qubit non-idling qubits not in a multi-qubit op
    # (drawn as colored arcs). a mixed gate participates in both.
    edge_type_pairs: dict[str, list[tuple[int, int]]] = {}
    arc_type_active: dict[str, frozenset[int]] = {}
    hidden_by_default: set[str] = set()
    gate_op_names: dict[str, dict[frozenset[int], str]] = {}

    for gate_name in gate_names:
        gate = gate_set[gate_name]
        qubit_set = frozenset(gate.qubit_idxs)
        op_names: dict[frozenset[int], str] = {}
        if hasattr(gate, "iter_ops"):
            for idxs, op in gate.iter_ops():
                op_names[frozenset(idxs)] = op.name
        gate_op_names[gate_name] = op_names
        pairs: list[tuple[int, int]] = []
        qubits_in_multi: set[int] = set()
        for idxs in gate.constituent_gate_idxs:
            if len(idxs) >= 2:
                qubits_in_multi.update(idxs)
                for idx in range(len(idxs)):
                    for jdx in range(idx + 1, len(idxs)):
                        pairs.append((min(idxs[idx], idxs[jdx]), max(idxs[idx], idxs[jdx])))
        if pairs:
            edge_type_pairs[gate_name] = pairs
        arc_qubits = frozenset(set(gate.qubit_idxs) - gate.idling_idxs - qubits_in_multi)
        if arc_qubits:
            arc_type_active[gate_name] = arc_qubits
            if gate.prep_idxs == qubit_set or gate.meas_idxs == qubit_set:
                hidden_by_default.add(gate_name)

    edge_to_gates: dict[tuple[int, int], list[str]] = defaultdict(list)
    for gate_name in gate_names:
        if gate_name in edge_type_pairs:
            for pair in edge_type_pairs[gate_name]:
                if gate_name not in edge_to_gates[pair]:
                    edge_to_gates[pair].append(gate_name)

    # used by _arc_segments to assign consistent sector positions across active and idling arcs
    qubit_to_all_gates: dict[int, list[str]] = {qubit: [] for qubit in range(gate_set.num_qubits)}
    for gate_name in gate_names:
        gate = gate_set[gate_name]
        arc_active = arc_type_active.get(gate_name, frozenset())
        idling = frozenset(gate.idling_idxs)
        for qubit in arc_active | idling:
            qubit_to_all_gates[qubit].append(gate_name)

    traces: list = []
    shown_in_legend: set[str] = set()

    # add device topology edges
    bg_x: list[float | None] = []
    bg_y: list[float | None] = []
    for q1, q2 in sorted(topo_edges):
        bg_x += [xs[q1], xs[q2], None]
        bg_y += [ys[q1], ys[q2], None]
    if bg_x:
        traces.append(
            go.Scatter(
                x=bg_x,
                y=bg_y,
                mode="lines",
                line={"width": 2, "color": _COLOR_BG_EDGE},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # colored edges for 2-qubit gates, offset when multiple gates share an edge
    for gate_name in gate_names:
        if gate_name not in edge_type_pairs:
            continue
        color = gate_colors[gate_name]
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edge_hover: list[str | None] = []
        for q1, q2 in edge_type_pairs[gate_name]:
            gates_on_edge = edge_to_gates[(q1, q2)]
            gate_idx = gates_on_edge.index(gate_name)
            num_gates_on_edge = len(gates_on_edge)
            delta_x, delta_y = xs[q2] - xs[q1], ys[q2] - ys[q1]
            length = math.hypot(delta_x, delta_y)
            perp_x, perp_y = (-delta_y / length, delta_x / length) if length > 0 else (0.0, 0.0)
            offset = _EDGE_OFFSET * (gate_idx - (num_gates_on_edge - 1) / 2)
            offset_x, offset_y = offset * perp_x, offset * perp_y
            mid_x = (xs[q1] + xs[q2]) / 2 + offset_x
            mid_y = (ys[q1] + ys[q2]) / 2 + offset_y
            edge_x += [xs[q1] + offset_x, mid_x, xs[q2] + offset_x, None]
            edge_y += [ys[q1] + offset_y, mid_y, ys[q2] + offset_y, None]
            op = gate_op_names[gate_name].get(frozenset({q1, q2}), gate_name)
            edge_hover += [None, f"{gate_name}<br>{op}: {q1}-{q2}", None, None]
        traces.append(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 4, "color": color},
                name=gate_labels[gate_name],
                legendgroup=gate_name,
                showlegend=True,
                hoverinfo="text",
                text=edge_hover,
            )
        )
        shown_in_legend.add(gate_name)

    # colored arcs around nodes for single-qubit gate activity
    for gate_name in gate_names:
        if gate_name not in arc_type_active:
            continue
        color = gate_colors[gate_name]

        def _active_label(qubit: int, gn: str = gate_name) -> str:
            gate = gate_set[gn]
            if qubit in gate.prep_idxs:
                op = "prepare"
            elif qubit in gate.meas_idxs:
                op = "measure"
            else:
                op = gate_op_names[gn].get(frozenset({qubit}), gn)
            return f"{gn}<br>{op}: {qubit}"

        arc_x, arc_y, hover = _arc_segments(
            arc_type_active[gate_name], xs, ys, qubit_to_all_gates, gate_name, _active_label
        )
        visible = "legendonly" if gate_name in hidden_by_default else True
        traces.append(
            go.Scatter(
                x=arc_x,
                y=arc_y,
                mode="lines",
                line={"width": 5, "color": color},
                name=gate_labels[gate_name],
                legendgroup=gate_name,
                legendgrouptitle=None,
                showlegend=gate_name not in shown_in_legend,
                visible=visible,
                hoverinfo="text",
                text=hover,
            )
        )
        shown_in_legend.add(gate_name)

    # faint arcs for idling qubits
    for gate_name in gate_names:
        gate = gate_set[gate_name]
        idling = frozenset(gate.idling_idxs)
        if not idling:
            continue
        color = gate_colors[gate_name]
        arc_x, arc_y, hover = _arc_segments(
            idling,
            xs,
            ys,
            qubit_to_all_gates,
            gate_name,
            lambda qubit, gn=gate_name: f"{gn}<br>idle: {qubit}",
        )
        visible = "legendonly" if gate_name in hidden_by_default else True
        traces.append(
            go.Scatter(
                x=arc_x,
                y=arc_y,
                mode="lines",
                line={"width": 5, "color": color},
                opacity=0.35,
                name=gate_labels[gate_name],
                legendgroup=gate_name,
                showlegend=gate_name not in shown_in_legend,
                visible=visible,
                hoverinfo="text",
                text=hover,
            )
        )
        shown_in_legend.add(gate_name)

    # node circles (data-coordinate shapes so they scale with zoom like the arcs)
    active_qubits = gate_set.qubit_subset
    node_shapes = []
    for qubit in range(gate_set.num_qubits):
        fill = _COLOR_NODE_ACTIVE if qubit in active_qubits else _COLOR_NODE_INACTIVE
        node_shapes.append(
            go.layout.Shape(
                type="circle",
                x0=xs[qubit] - _NODE_RADIUS,
                y0=ys[qubit] - _NODE_RADIUS,
                x1=xs[qubit] + _NODE_RADIUS,
                y1=ys[qubit] + _NODE_RADIUS,
                fillcolor=fill,
                line={"color": fill, "width": 0},
                layer="above",
            )
        )

    # hover-only scatter at node centres (labels are annotations rendered above shapes)
    traces.append(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker={"size": 1, "opacity": 0},
            hoverinfo="text",
            hovertext=[f"{qubit}" for qubit in range(gate_set.num_qubits)],
            showlegend=False,
        )
    )

    # qubit-index annotations sit above the node shapes
    label_annotations = [
        go.layout.Annotation(
            x=xs[qubit],
            y=ys[qubit],
            text=str(qubit),
            showarrow=False,
            font={"color": _COLOR_LABEL, "size": 8},
            xanchor="center",
            yanchor="middle",
        )
        for qubit in range(gate_set.num_qubits)
    ]

    x_span = max(xs) - min(xs) if len(xs) > 1 else 2.0
    y_span = max(ys) - min(ys) if len(ys) > 1 else 2.0
    fig_size = max(500, int(45 * max(x_span, y_span)))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{gate_set.name} on {len(gate_set.qubit_subset)} Qubits",
        showlegend=True,
        legend={"yanchor": "middle", "y": 0.5},
        paper_bgcolor=_COLOR_FIGURE_BG,
        plot_bgcolor=_COLOR_FIGURE_BG,
        shapes=node_shapes,
        annotations=label_annotations,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "scaleanchor": "x",
        },
        width=fig_size,
        height=fig_size,
    )
    return fig
