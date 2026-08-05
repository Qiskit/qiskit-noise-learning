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

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..optionals import HAS_MATPLOTLIB
from ._qubit_coordinates import qubit_coordinates
from .interactive_figure import InteractiveFigure, TokenMap, cell_token, tag_legend, trace_gid

if TYPE_CHECKING:
    from ..gate_sets.gate import Gate
    from ..gate_sets.gate_set import GateSet

_NODE_RADIUS = 0.20  # data-unit radius of qubit node circles
_ARC_RADIUS = 0.24  # data-unit radius of gate arcs (must be > _NODE_RADIUS)
_ARC_NPTS = 50  # number of points per arc segment
_ARC_GAP = 0.08  # fraction of sector span to leave as a visual gap on each side
_EDGE_OFFSET = 0.14  # data-unit perpendicular offset between parallel edges

_COLOR_BG_EDGE = "0.71"  # unoccupied topology edges
_COLOR_NODE_ACTIVE = "0.24"  # nodes within qubit_subset
_COLOR_NODE_INACTIVE = "0.71"  # nodes outside qubit_subset
_COLOR_LABEL = "#d2d2dc"  # qubit index labels inside nodes
_COLOR_FIGURE_BG = "white"  # figure and plot background
_COLOR_ACTIVITY_PROXY = "0.35"  # activity legend handles, which stand for no gate in particular

_BG_LINEWIDTH = 2
_EDGE_LINEWIDTH = 4
_ARC_LINEWIDTH = 5
_IDLING_ALPHA = 0.35  # idling arcs, and the legend handle standing for them
_LABEL_FONT_SIZE = 8

# The dimensions this figure resolves visibility along, one legend each: which gate a mark belongs
# to, and what kind of activity it shows. The order is the order the tokens appear in a trace's
# ``gid``. They are named for what this figure is about rather than reusing the decay plots' "path"
# and "layer", since the figure declares its own dimensions and nothing outside it has to recognize
# the names.
_DIMENSIONS = ("gate", "activity")
_GATE_DIMENSION, _ACTIVITY_DIMENSION = _DIMENSIONS

# Names of the three kinds of mark, which are both the activity keys and the labels of the legend
# that switches them.
_INTERACTION_ACTIVITY = "Interactions"
_ACTIVE_ACTIVITY = "Single-qubit"
_IDLING_ACTIVITY = "Idling"

# Layout. The canvas is sized from the extent of the device so that the aspect ratio matplotlib is
# asked to hold is the one it is given room for; the legends sit outside it and are added to the
# crop by ``bbox_inches="tight"``.
_INCHES_PER_UNIT = 0.45
_MIN_FIG_INCHES = 3.0
_PADDING = _ARC_RADIUS + 0.3  # data units of margin, enough that no arc touches the edge

#: One drawn mark: the polyline to draw, the single point its hover readout hangs off, and the text
#: of that readout.  The readout attaches to one point rather than every vertex because a mark is
#: never larger than the pointer's catch radius, and a per-vertex payload on a full device would
#: outweigh the figure.
_Mark = tuple[np.ndarray, np.ndarray, tuple[float, float], str]


def _arc_marks(
    qubits: frozenset[int],
    xs: list[float],
    ys: list[float],
    qubit_to_all_gates: dict[int, list[str]],
    gate_name: str,
    label_fn: Callable[[int], str],
) -> list[_Mark]:
    r"""Return one arc mark per qubit, drawn around that qubit's node.

    Sector ``sector_idx`` of ``num_sectors`` is centred at angle
    :math:`2\pi \cdot \mathrm{sector\_idx} / \mathrm{num\_sectors}` (``sector_idx=0``
    :math:`\to` right side), proceeding counter-clockwise.
    """
    marks: list[_Mark] = []
    for qubit in sorted(qubits):
        gates_on_qubit = qubit_to_all_gates[qubit]
        sector_idx = gates_on_qubit.index(gate_name)
        num_sectors = len(gates_on_qubit)
        center = 2 * math.pi * sector_idx / num_sectors
        half = math.pi / num_sectors * (1 - _ARC_GAP)
        angles = np.linspace(center - half, center + half, _ARC_NPTS)
        arc_x = xs[qubit] + _ARC_RADIUS * np.cos(angles)
        arc_y = ys[qubit] + _ARC_RADIUS * np.sin(angles)
        middle = _ARC_NPTS // 2
        marks.append((arc_x, arc_y, (arc_x[middle], arc_y[middle]), label_fn(qubit)))
    return marks


def _joined(pieces: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[list[float], list[float]]:
    """Concatenate several polylines into one, so that a gate's marks are a single artist.

    A ``nan`` between them lifts the pen, which is what keeps them separate strokes; one artist is
    what lets the whole set be switched by a single ``gid``.
    """
    line_x: list[float] = []
    line_y: list[float] = []
    for piece_x, piece_y in pieces:
        line_x.extend(float(value) for value in piece_x)
        line_y.extend(float(value) for value in piece_y)
        line_x.append(math.nan)
        line_y.append(math.nan)
    return line_x, line_y


@HAS_MATPLOTLIB.require_in_call
def gate_set_topology(gate_set: "GateSet[Gate]") -> InteractiveFigure:
    """Draw the device topology with per-gate coloring.

    Gates with 2-qubit interactions are drawn as colored edges on the device
    coupling graph. Gates that act only on individual qubits (such as preparation
    and measurement) are shown as colored arcs around the relevant nodes, with one
    arc sector per gate per qubit. The arc sectors are arranged so that the first
    gate in the set occupies the right-hand side of the circle (angle 0), and
    subsequent gates proceed counter-clockwise. Qubits that are idling in a given
    gate receive a slightly transparent arc.

    Two legends switch what is shown: one per gate, and one per kind of mark. Gates that only
    prepare or only measure open switched off, since they touch every qubit at once and would
    otherwise bury the rest; their legend entry brings them back.

    Qubits are placed in the device's conventional layout where it has one, and laid out from its
    coupling graph otherwise. A target that constrains connectivity nowhere -- an ideal or
    all-to-all device -- has no topology to draw and no graph to lay out from, so its qubits are
    drawn without edges, in the conventional layout if its size has one.

    Args:
        gate_set: The gate set to visualize. Must have a non-``None`` :attr:`~.GateSet.target` so
            that qubit coordinates and the device topology can be determined.

    Returns:
        The figure.

    Raises:
        ValueError: If ``gate_set.target`` is ``None``, or if the device's size is not one of the
            conventionally drawn ones and its target constrains connectivity nowhere, leaving no
            coupling graph to derive a layout from.
        ImportError: If ``matplotlib`` is not installed.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    if gate_set.target is None:
        raise ValueError(
            "Cannot draw gate set topology: gate_set.target is None. "
            "A Target is required to determine qubit coordinates and device connectivity."
        )

    # ``None`` when the target constrains connectivity nowhere -- an ideal or all-to-all device, or
    # one mixing constrained and globally defined two-qubit operations. There is then no topology to
    # draw beneath the gates, which is a device to draw without edges rather than an error.
    coupling_map = gate_set.target.build_coupling_map()
    coords = qubit_coordinates(gate_set.num_qubits, coupling_map)
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]

    topo_edges = (
        set()
        if coupling_map is None
        else {(min(q1, q2), max(q1, q2)) for q1, q2 in coupling_map.get_edges()}
    )

    gate_names = list(gate_set)
    # ``CN`` names the Nth color of the active style's property cycle, wrapping, so the palette
    # follows whatever style the figure is drawn under instead of being fixed here.
    gate_colors = {name: f"C{idx % 10}" for idx, name in enumerate(gate_names)}

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

    # used by _arc_marks to assign consistent sector positions across active and idling arcs
    qubit_to_all_gates: dict[int, list[str]] = {qubit: [] for qubit in range(gate_set.num_qubits)}
    for gate_name in gate_names:
        gate = gate_set[gate_name]
        arc_active = arc_type_active.get(gate_name, frozenset())
        idling = frozenset(gate.idling_idxs)
        for qubit in arc_active | idling:
            qubit_to_all_gates[qubit].append(gate_name)

    x_span = max(xs) - min(xs) if len(xs) > 1 else 2.0
    y_span = max(ys) - min(ys) if len(ys) > 1 else 2.0
    fig = Figure(
        figsize=(
            max(_MIN_FIG_INCHES, _INCHES_PER_UNIT * (x_span + 2 * _PADDING)),
            max(_MIN_FIG_INCHES, _INCHES_PER_UNIT * (y_span + 2 * _PADDING)),
        ),
        layout="constrained",
        facecolor=_COLOR_FIGURE_BG,
    )
    ax = fig.subplots()
    ax.set_facecolor(_COLOR_FIGURE_BG)
    # The coupling graph's proportions carry information -- which qubits are neighbours -- so both
    # axes have to keep the same scale. Ticks and spines are hidden individually rather than with
    # ``set_axis_off``, which would take the background patch with them, and the patch is what the
    # browser measures to place the hover readout.
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(min(xs) - _PADDING, max(xs) + _PADDING)
    ax.set_ylim(min(ys) - _PADDING, max(ys) + _PADDING)

    cell = cell_token(0)
    gate_tokens, activity_tokens = TokenMap("g"), TokenMap("a")
    traces: dict[str, dict[str, Any]] = {}

    def draw(marks: Sequence[_Mark], gate_name: str, activity: str, **line_kwargs: Any) -> None:
        """Draw one gate's marks of one kind, as a single artist the browser can switch."""
        if not marks:
            return
        line_x, line_y = _joined([(mark_x, mark_y) for mark_x, mark_y, _, _ in marks])
        (line,) = ax.plot(line_x, line_y, color=gate_colors[gate_name], **line_kwargs)
        # Both tokens are taken here rather than up front, so that a gate or a kind of mark only
        # reaches a legend once it has actually drawn something. Their order is _DIMENSIONS'.
        gid = trace_gid(cell, (gate_tokens.token(gate_name), activity_tokens.token(activity)))
        line.set_gid(gid)
        traces[gid] = {
            "label": gate_labels[gate_name],
            "points": [[float(point[0]), float(point[1])] for _, _, point, _ in marks],
            "texts": [text for _, _, _, text in marks],
        }

    # device topology edges, under everything else and part of no gate
    if topo_edges:
        background = [
            (np.array([xs[q1], xs[q2]]), np.array([ys[q1], ys[q2]]))
            for q1, q2 in sorted(topo_edges)
        ]
        ax.plot(*_joined(background), color=_COLOR_BG_EDGE, linewidth=_BG_LINEWIDTH, zorder=1)

    # colored edges for 2-qubit gates, offset when multiple gates share an edge
    for gate_name in gate_names:
        edge_marks: list[_Mark] = []
        for q1, q2 in edge_type_pairs.get(gate_name, []):
            gates_on_edge = edge_to_gates[(q1, q2)]
            gate_idx = gates_on_edge.index(gate_name)
            num_gates_on_edge = len(gates_on_edge)
            delta_x, delta_y = xs[q2] - xs[q1], ys[q2] - ys[q1]
            length = math.hypot(delta_x, delta_y)
            perp_x, perp_y = (-delta_y / length, delta_x / length) if length > 0 else (0.0, 0.0)
            offset = _EDGE_OFFSET * (gate_idx - (num_gates_on_edge - 1) / 2)
            offset_x, offset_y = offset * perp_x, offset * perp_y
            edge_x = np.array([xs[q1] + offset_x, xs[q2] + offset_x])
            edge_y = np.array([ys[q1] + offset_y, ys[q2] + offset_y])
            op = gate_op_names[gate_name].get(frozenset({q1, q2}), gate_name)
            edge_marks.append(
                (
                    edge_x,
                    edge_y,
                    (float(edge_x.mean()), float(edge_y.mean())),
                    f"{gate_name}\n{op}: {q1}-{q2}",
                )
            )
        draw(edge_marks, gate_name, _INTERACTION_ACTIVITY, linewidth=_EDGE_LINEWIDTH, zorder=2)

    # colored arcs around nodes for single-qubit gate activity
    for gate_name in gate_names:
        if gate_name not in arc_type_active:
            continue

        def _active_label(qubit: int, gn: str = gate_name) -> str:
            gate = gate_set[gn]
            if qubit in gate.prep_idxs:
                op = "prepare"
            elif qubit in gate.meas_idxs:
                op = "measure"
            else:
                op = gate_op_names[gn].get(frozenset({qubit}), gn)
            return f"{gn}\n{op}: {qubit}"

        draw(
            _arc_marks(
                arc_type_active[gate_name], xs, ys, qubit_to_all_gates, gate_name, _active_label
            ),
            gate_name,
            _ACTIVE_ACTIVITY,
            linewidth=_ARC_LINEWIDTH,
            zorder=2,
        )

    # faint arcs for idling qubits
    for gate_name in gate_names:
        idling = frozenset(gate_set[gate_name].idling_idxs)
        if not idling:
            continue
        draw(
            _arc_marks(
                idling,
                xs,
                ys,
                qubit_to_all_gates,
                gate_name,
                lambda qubit, gn=gate_name: f"{gn}\nidle: {qubit}",
            ),
            gate_name,
            _IDLING_ACTIVITY,
            linewidth=_ARC_LINEWIDTH,
            alpha=_IDLING_ALPHA,
            zorder=2,
        )

    # node circles, in data coordinates so that they keep their place among the arcs, with the qubit
    # index over the top
    active_qubits = gate_set.qubit_subset
    for qubit in range(gate_set.num_qubits):
        fill = _COLOR_NODE_ACTIVE if qubit in active_qubits else _COLOR_NODE_INACTIVE
        ax.add_patch(
            Circle((xs[qubit], ys[qubit]), _NODE_RADIUS, facecolor=fill, edgecolor="none", zorder=3)
        )
        ax.text(
            xs[qubit],
            ys[qubit],
            str(qubit),
            color=_COLOR_LABEL,
            fontsize=_LABEL_FONT_SIZE,
            ha="center",
            va="center",
            zorder=4,
        )

    fig.suptitle(f"{gate_set.name} on {len(gate_set.qubit_subset)} Qubits")

    gate_entries = list(gate_tokens.items())
    if gate_entries:
        legend = fig.legend(
            [Line2D([], [], color=gate_colors[name], linewidth=2) for name, _ in gate_entries],
            [gate_labels[name] for name, _ in gate_entries],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            title="Gate",
            frameon=False,
        )
        tag_legend(legend, _GATE_DIMENSION, [token for _, token in gate_entries])

    activity_entries = list(activity_tokens.items())
    if activity_entries:
        # The handles say nothing the labels do not -- an arc cannot be drawn in a legend, and a
        # straight sample line for all three would only look like a mistake -- so they are uniform,
        # apart from the fading that distinguishes idling in the figure itself. Their job here is to
        # be a second thing to click.
        legend = fig.legend(
            [
                Line2D(
                    [],
                    [],
                    color=_COLOR_ACTIVITY_PROXY,
                    linewidth=3,
                    alpha=_IDLING_ALPHA if key == _IDLING_ACTIVITY else 1.0,
                )
                for key, _ in activity_entries
            ],
            [key for key, _ in activity_entries],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncols=len(activity_entries),
            title="Activity",
            frameon=False,
        )
        tag_legend(legend, _ACTIVITY_DIMENSION, [token for _, token in activity_entries])

    hidden = {
        _GATE_DIMENSION: [
            gate_tokens.token(name)
            for name in gate_names
            if name in hidden_by_default and name in gate_tokens
        ]
    }
    return InteractiveFigure(
        fig, dimensions=_DIMENSIONS, cells={cell: ax}, traces=traces, hidden=hidden
    )
