# This code is a Qiskit project.
#
# (C) Copyright IBM 2025, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for running a QuantumProgram on a local Aer simulator."""

import numpy as np
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.quantum_info import PauliLindbladMap
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_ibm_runtime.quantum_program.quantum_program import (
    CircuitItem,
    QuantumProgram,
    SamplexItem,
)
from qiskit_ibm_runtime.results import QuantumProgramResult

from ._seeding import next_seed
from .broadcast_sample import broadcast_sample
from .insert_noise_pass import InsertNoisePass


def _round_to_clifford(values: np.ndarray, decimals: int) -> np.ndarray:
    """Round angles to the nearest multiple of π/2 at ``decimals`` decimal places.

    This prevents floating-point drift from disqualifying nominally-Clifford circuits
    from the stabilizer simulation method.
    """
    return np.round(values / (np.pi / 2), decimals=decimals) * (np.pi / 2)


def get_aer_sampler(aer_simulator: AerSimulator, seed: int | None = None) -> AerSamplerV2:
    """Return an :class:`~qiskit_aer.primitives.SamplerV2` that runs on ``aer_simulator``.

    The simulator is used as given — neither copied nor mutated — so several samplers can
    share one and any configuration the caller made survives.

    Args:
        aer_simulator: The simulator the sampler runs on.
        seed: Seed for the sampler's random number generator.  If ``None``, the sampler
            seeds itself nondeterministically.

    Returns:
        A sampler that runs on ``aer_simulator``.
    """
    return AerSamplerV2.from_backend(aer_simulator, seed=seed)


def run_quantum_program(
    qasm_simulator: AerSimulator,
    program: QuantumProgram,
    noise_dict: dict[str, PauliLindbladMap] | None = None,
    angle_decimals: int = 5,
    warn_absent: bool = True,
    seed: int | None = None,
) -> QuantumProgramResult:
    """Run a quantum program on a simulator.

    Args:
        qasm_simulator: The simulator to use.
        program: The program to run.
        noise_dict: A map from barrier label refs to noise maps.
        angle_decimals: Gate angles are rounded to the nearest multiple of π/2 at this
            decimal precision before simulation.  See :func:`AerExecutor` for details.
        warn_absent: Passed to :class:`InsertNoisePass`; see :class:`AerExecutor`.
        seed: Root seed for this run.  Independent seeds are derived from it for the twirl
            sampling and for each item's shot sampling, so a fixed value reproduces the run
            without correlating the items with each other.  If ``None``, the root seed is
            drawn nondeterministically.  See :class:`AerExecutor`.

    Returns:
        Results of simulation.
    """
    seed_sequence = np.random.SeedSequence(seed)
    rng = np.random.default_rng(next_seed(seed_sequence))

    result_list = []
    metadata_list = []

    for prog_item in program.items:
        # A fresh seed per item, so that items sharing a circuit are still sampled
        # independently rather than returning identical shots.
        aer_sampler = get_aer_sampler(qasm_simulator, seed=next_seed(seed_sequence))

        if noise_dict is not None:
            circuit = PassManager(
                [InsertNoisePass(noise_dict=noise_dict, warn_absent=warn_absent)]
            ).run(prog_item.circuit)
        else:
            circuit = prog_item.circuit

        if isinstance(prog_item, CircuitItem):
            if prog_item.circuit_arguments is not None:
                bindings_array = BindingsArray(
                    {tuple(prog_item.circuit.parameters): prog_item.circuit_arguments}
                )
                for k, v in bindings_array._data.items():  # noqa: SLF001
                    bindings_array._data[k] = _round_to_clifford(v, angle_decimals)  # noqa: SLF001
            else:
                bindings_array = None
            sampler_res = aer_sampler.run(
                [
                    SamplerPub(
                        circuit=circuit,
                        parameter_values=bindings_array,
                        shots=program.shots,
                    )  # type: ignore
                ]
            ).result()
            metadata_list.append(sampler_res[0].metadata)
            bit_array = sampler_res[0].data
            data = {key: ba.to_bool_array(order="little") for key, ba in dict(bit_array).items()}
            result_list.append(data)

        elif isinstance(prog_item, SamplexItem):
            samplex_data = broadcast_sample(
                prog_item.samplex,
                prog_item.samplex_arguments,
                prog_item.shape,
                rng,
            )
            bindings_array = BindingsArray(
                {tuple(prog_item.circuit.parameters): samplex_data.pop("parameter_values")}
            )
            for k, v in bindings_array._data.items():  # noqa: SLF001
                bindings_array._data[k] = _round_to_clifford(v, angle_decimals)  # noqa: SLF001
            sampler_res = aer_sampler.run(
                [
                    SamplerPub(
                        circuit=circuit,
                        parameter_values=bindings_array,
                        shots=program.shots,
                    )  # type: ignore
                ]
            ).result()
            metadata_list.append(sampler_res[0].metadata)
            bit_array = sampler_res[0].data
            bool_arrays = {
                key: ba.to_bool_array(order="little") for key, ba in dict(bit_array).items()
            }
            data = {**samplex_data, **bool_arrays}
            result_list.append(data)

        else:
            raise TypeError(f"Unsupported QuantumProgramItem type: {type(prog_item)}")

    return QuantumProgramResult(
        data=result_list,
        metadata=dict(enumerate(metadata_list)),
        passthrough_data=program.passthrough_data,
    )
