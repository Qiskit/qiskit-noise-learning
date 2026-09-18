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

"""Samplex surgery for injecting state-preparation noise at the front of a circuit.

Noise-learning and mitigation circuits no longer carry a dedicated preparation box (an empty
twirled single-qubit layer) at their start.  Instead, state-preparation noise is represented as a
:class:`~qiskit.quantum_info.PauliLindbladMap` and injected directly into a built samplex so that it
acts *before any gate* on the prepared qubits -- earlier than the box ``"before"`` injection site,
which sits after single-qubit gates absorbed into the first dressing.
"""

from collections.abc import Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from samplomatic import build
from samplomatic.samplex import Samplex
from samplomatic.samplex.nodes import (
    CollectTemplateValues,
    CollectZ2ToOutputNode,
    CombineRegistersNode,
    InjectNoiseNode,
    Node,
)
from samplomatic.tensor_interface import PauliLindbladMapSpecification, TensorSpecification
from samplomatic.virtual_registers import VirtualType

_PAULI_SIGNS_OUTPUT = "pauli_signs"
_PAULI_SIGNS_DESCRIPTION = "Signs from sampled Pauli Lindblad maps."


def _written_registers(node: Node) -> dict:
    """Registers a node either writes to or instantiates, keyed by register name."""
    registers = {}
    registers.update(node.writes_to())
    registers.update(node.instantiates())
    return registers


def inject_prep_noise(
    samplex: Samplex,
    qubits: Sequence[int],
    noise_ref: str,
    *,
    modifier_ref: str = "",
) -> Samplex:
    """Inject state-preparation Pauli-Lindblad noise at the physical front of a built samplex.

    The noise is spliced into the *earliest* single-qubit dressing of the circuit (the
    :class:`~samplomatic.samplex.nodes.CollectTemplateValues` with the smallest template index) as
    the last operand of that dressing's combine node.  Because the injection is a virtual
    Pauli-frame operation folded into the existing dressing, it adds no physical layer and acts
    before every gate on ``qubits`` -- including a leading single-qubit gate absorbed into that
    dressing.  This reproduces the transpiler's native ``inject_noise_site="before"`` result exactly
    when nothing precedes the dressing, and correctly precedes leading single-qubit gates when they
    are present (which the ``"before"`` annotation site does not).

    The concrete :class:`~qiskit.quantum_info.PauliLindbladMap` is supplied at sampling time via the
    ``samplex.sample`` input ``pauli_lindblad_maps.{noise_ref}``, and the sampled Pauli signs are
    appended as a new column of the ``pauli_signs`` output.

    .. code-block:: python

        from samplomatic import build
        from qiskit_noise_learning.utils import inject_prep_noise

        template, samplex = build(boxed_circuit)
        inject_prep_noise(samplex, qubits=[0, 1], noise_ref="prep")
        out = samplex.sample(
            {"pauli_lindblad_maps.prep": prep_map}, num_randomizations=1000, rng=0
        )

    Args:
        samplex: A samplex produced by :func:`samplomatic.build`.  Mutated in place.
        qubits: The prepared qubits the noise acts on.  The map supplied at sampling time is
            interpreted over these qubits in ascending physical order, and they must coincide with
            the qubits spanned by the circuit's earliest single-qubit dressing.
        noise_ref: The reference name keying the map at sampling time
            (``pauli_lindblad_maps.{noise_ref}``) and naming the created registers.
        modifier_ref: Optional reference for per-sample rate modifiers.  When set, the optional
            sampling inputs ``noise_scales.{modifier_ref}`` (scalar) and
            ``local_scales.{modifier_ref}`` (per-term) scale the map's rates.

    Returns:
        The mutated, re-finalized samplex (also modified in place).
    """
    graph = samplex.graph
    num_subsystems = len(qubits)

    # 1. The earliest dressing is the CollectTemplateValues with the smallest template index; it is
    #    the initial single-qubit (twirl) layer that serves as the preparation-noise site.
    dressings = [
        (int(graph[idx].template_idxs.min()), idx)
        for idx in graph.node_indices()
        if isinstance(graph[idx], CollectTemplateValues)
    ]
    if not dressings:
        raise ValueError(
            "Cannot inject preparation noise: the circuit has no initial single-qubit-gate layer "
            "to serve as the preparation-noise site."
        )
    collect_idx = min(dressings)[1]
    collect = graph[collect_idx]
    dressing_register = collect._register_name  # noqa: SLF001
    dressing_type = collect._register_type  # noqa: SLF001

    # The node currently producing the dressing register (edge feeder -> collect).
    feeder_idx = next(
        pred
        for pred in graph.predecessor_indices(collect_idx)
        if dressing_register in _written_registers(graph[pred])
    )

    # 2. The noise-injection node (produces a Pauli register plus a length-1 sign register).
    inject_register = f"prep_inject_{noise_ref}"
    sign_register = f"prep_sign_{noise_ref}"
    inject_idx = samplex.add_node(
        InjectNoiseNode(inject_register, sign_register, noise_ref, num_subsystems, modifier_ref)
    )

    # 3. A new combine node: existing dressing register first, injected register LAST so that the
    #    injected Pauli acts on the earliest (input) side of the dressing.
    identity = np.arange(num_subsystems)
    combined_register = f"prep_combined_{noise_ref}"
    operands = {
        dressing_register: (identity, identity, dressing_type),
        inject_register: (identity, identity, VirtualType.PAULI),
    }
    combine_idx = samplex.add_node(
        CombineRegistersNode(dressing_type, combined_register, num_subsystems, operands)
    )
    samplex.add_edge(feeder_idx, combine_idx)
    samplex.add_edge(inject_idx, combine_idx)

    # 4. Repoint the dressing's collect node to read the combined register.
    graph[collect_idx] = CollectTemplateValues(
        collect._template_params_name,  # noqa: SLF001
        collect.template_idxs,
        combined_register,
        dressing_type,
        collect._subsystem_idxs,  # noqa: SLF001
        collect._synth,  # noqa: SLF001
    )
    graph.remove_edge(feeder_idx, collect_idx)
    samplex.add_edge(combine_idx, collect_idx)

    # 5. Route the sign register into the pauli_signs output, growing it by one column.
    if _PAULI_SIGNS_OUTPUT in samplex._output_specifications:  # noqa: SLF001
        previous = samplex._output_specifications.pop(_PAULI_SIGNS_OUTPUT)  # noqa: SLF001
        column = previous.shape[-1]
        description = previous.description
    else:
        column = 0
        description = _PAULI_SIGNS_DESCRIPTION
    sign_collect_idx = samplex.add_node(
        CollectZ2ToOutputNode(sign_register, [0], _PAULI_SIGNS_OUTPUT, [column])
    )
    samplex.add_edge(inject_idx, sign_collect_idx)
    samplex.add_output(
        TensorSpecification(
            _PAULI_SIGNS_OUTPUT,
            ("num_randomizations", column + 1),
            np.dtype(np.bool_),
            description,
        )
    )

    # 6. Register the map input (and optional per-sample rate modifiers).
    samplex.add_input(
        PauliLindbladMapSpecification(
            f"pauli_lindblad_maps.{noise_ref}", num_subsystems, f"num_terms_{noise_ref}"
        )
    )
    if modifier_ref:
        samplex.add_input(
            TensorSpecification(
                f"noise_scales.{modifier_ref}", (), np.dtype(np.float64), "", optional=True
            )
        )
        samplex.add_input(
            TensorSpecification(
                f"local_scales.{modifier_ref}",
                (f"num_terms_{noise_ref}",),
                np.dtype(np.float64),
                "",
                optional=True,
            )
        )

    return samplex.finalize()


def build_with_prep_noise(
    boxed_circuit: QuantumCircuit,
    qubits: Sequence[int],
    noise_ref: str,
    *,
    modifier_ref: str = "",
) -> tuple[QuantumCircuit, Samplex]:
    """Build a boxed circuit and inject preparation noise at the circuit's front.

    This is the entry point for assembling a prep-layer-free mitigation program: it runs
    :func:`samplomatic.build` and then applies :func:`inject_prep_noise` once for the preparation
    noise. The concrete map is supplied at sampling time as the samplex input
    ``pauli_lindblad_maps.{noise_ref}``; noise for the ordinary boxes is applied by their own
    :class:`~samplomatic.annotations.InjectNoise` annotations (during sampling) or by
    :class:`~.InsertNoisePass` (during Aer simulation), unchanged.

    The ``qubits``, ``noise_ref``, and ``modifier_ref`` arguments mirror :func:`inject_prep_noise`.

    Args:
        boxed_circuit: A circuit whose ordinary layers are annotated boxes.
        qubits: The prepared qubits the noise acts on (see :func:`inject_prep_noise`).
        noise_ref: The reference keying the map at sampling time.
        modifier_ref: Optional reference for per-sample rate modifiers.

    Returns:
        The built template circuit and the samplex with preparation noise injected.
    """
    template, samplex = build(boxed_circuit)
    inject_prep_noise(samplex, qubits, noise_ref, modifier_ref=modifier_ref)
    return template, samplex
