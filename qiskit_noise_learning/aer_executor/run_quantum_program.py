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

from copy import deepcopy

import numpy as np
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.quantum_info import PauliLindbladMap
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_ibm_runtime.quantum_program import QuantumProgram, QuantumProgramResult
from qiskit_ibm_runtime.quantum_program.quantum_program import CircuitItem, SamplexItem

from .broadcast_sample import broadcast_sample
from .insert_noise_pass import InsertNoisePass


def get_aer_sampler(aer_simulator: AerSimulator) -> AerSamplerV2:
    aer_simulator.set_max_qubits(10000)
    # Pass a deepcopy so the sampler's backend is independent of the input simulator.
    return AerSamplerV2.from_backend(deepcopy(aer_simulator))


def run_quantum_program(
    qasm_simulator: AerSimulator,
    program: QuantumProgram,
    noise_dict: dict[str, PauliLindbladMap] | None = None,
    angle_decimals: int = 5,
) -> QuantumProgramResult:
    """Run a quantum program on a simulator.

    Args:
        qasm_simulator: The simulator to use.
        program: The program to run.
        noise_dict: A map from barrier label refs to noise maps.
        angle_decimals: How accurately to resolve angles.

    Returns:
        Results of simulation.
    """
    aer_sampler = get_aer_sampler(qasm_simulator)
    rng = np.random.default_rng(aer_sampler._seed)  # noqa: SLF001

    result_list = []
    metadata_list = []

    for prog_item in program.items:
        if noise_dict is not None:
            circuit = PassManager([InsertNoisePass(noise_dict=noise_dict)]).run(prog_item.circuit)
        else:
            circuit = prog_item.circuit

        if isinstance(prog_item, CircuitItem):
            if prog_item.circuit_arguments is not None:
                bindings_array = BindingsArray(
                    {tuple(prog_item.circuit.parameters): prog_item.circuit_arguments}
                )
                for k, v in bindings_array._data.items():  # noqa: SLF001
                    bindings_array._data[k] = np.round(  # noqa: SLF001
                        v / (np.pi / 2), decimals=angle_decimals
                    ) * (np.pi / 2)
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
                bindings_array._data[k] = np.round(v / (np.pi / 2), decimals=angle_decimals) * (  # noqa: SLF001
                    np.pi / 2
                )
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
