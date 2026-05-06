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

"""Functions for resolving SamplexItems into CircuitItems in a QuantumProgram."""

import numpy as np
from qiskit_ibm_runtime.quantum_program import QuantumProgram
from qiskit_ibm_runtime.quantum_program.quantum_program import CircuitItem, SamplexItem

from .broadcast_sample import broadcast_sample


def inline_samplex_item(
    item: SamplexItem,
    rng: np.random.Generator | None = None,
) -> tuple[CircuitItem, dict[str, np.ndarray]]:
    """Inline a :class:`SamplexItem` into a :class:`CircuitItem` and passthrough data.

    Calls ``item.samplex.sample()`` for every configuration in the item's extrinsic shape,
    assembles the results into arrays of shape ``(*item.shape, *intrinsic)``, then constructs
    a :class:`CircuitItem` whose ``circuit_arguments`` are the sampled ``parameter_values``.
    All other arrays returned by the samplex are collected into a passthrough dict.

    Args:
        item: The samplex item to resolve.
        rng: Random number generator. If ``None``, ``np.random.default_rng()`` is used.

    Returns:
        A tuple ``(circuit_item, passthrough_data)`` where ``passthrough_data`` maps output
        names (excluding ``"parameter_values"``) to arrays of shape
        ``(*item.shape, *intrinsic)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    output = broadcast_sample(item.samplex, item.samplex_arguments, item.shape, rng)
    circuit_arguments = output.pop("parameter_values")
    circuit_item = CircuitItem(
        item.circuit, circuit_arguments=circuit_arguments, chunk_size=item.chunk_size
    )
    return circuit_item, output


def inline_samplexes(
    program: QuantumProgram,
    omit: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> QuantumProgram:
    """Return a new :class:`QuantumProgram` with :class:`SamplexItem`\\s resolved in-place.

    Each :class:`SamplexItem` whose index is not in ``omit`` is replaced by a
    :class:`CircuitItem` via :func:`inline_samplex_item`.  The passthrough data
    produced by sampling (e.g. ``measurement_flips.*`` arrays) is stored in the
    returned program's ``passthrough_data`` under the key ``"inlined"``, keyed by
    the original item index (as a string).  Any existing ``passthrough_data`` on the
    input program is preserved alongside the new ``"inlined"`` entry.

    Args:
        program: The quantum program to process.
        omit: Indices of items to leave as :class:`SamplexItem`\\s.  All other
            :class:`SamplexItem`\\s are inlined.  ``CircuitItem``\\s are always
            left unchanged regardless of this list.
        rng: Random number generator passed to :func:`inline_samplex_item`.
            If ``None``, ``np.random.default_rng()`` is used.

    Returns:
        A new :class:`QuantumProgram` with the same shots, noise maps, and
        measurement level as the input, but with the selected
        :class:`SamplexItem`\\s replaced by :class:`CircuitItem`\\s.
    """
    if rng is None:
        rng = np.random.default_rng()

    omit_set = set(omit or [])
    new_items = []
    inlined_passthrough: dict[str, dict[str, np.ndarray]] = {}

    for idx, item in enumerate(program.items):
        if isinstance(item, SamplexItem) and idx not in omit_set:
            circuit_item, passthrough = inline_samplex_item(item, rng=rng)
            new_items.append(circuit_item)
            inlined_passthrough[str(idx)] = passthrough
        else:
            new_items.append(item)

    if isinstance(program.passthrough_data, dict):
        combined_passthrough = {**program.passthrough_data, "inlined": inlined_passthrough}
    elif program.passthrough_data is None:
        combined_passthrough = {"inlined": inlined_passthrough}
    else:
        combined_passthrough = {
            "inlined": inlined_passthrough,
            "original": program.passthrough_data,
        }

    return QuantumProgram(
        shots=program.shots,
        items=new_items,
        noise_maps=program.noise_maps,
        meas_level=program.meas_level,
        passthrough_data=combined_passthrough,
    )
