#%%
# Standard library
import pickle
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Qiskit
from qiskit.quantum_info import PauliLindbladMap
from qiskit_ibm_runtime import QuantumProgram
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_aer import QasmSimulator


# AerExecutor
from qiskit_noise_learning.aer_executor import AerExecutor

# Qiskit Noise Learning
from qiskit_noise_learning.gate_sets import QiskitGateSet
from qiskit_noise_learning.models import PauliLindbladModel
from qiskit_noise_learning.experiment_builder import (
    ExperimentBuilder,
    standard_vanilla_pattern_generator,
)
from qiskit_noise_learning.circuit_generator import ExecutorCircuitGenerator
from qiskit_noise_learning.analysis import (
    AnalysisPipeline,
    ComputeObservables,
    CurveFitObservables,
    NNLSSolve,
    AverageObservables,
    Fit,
    SymmetrizeFidelities,
    SymmetrizeGenerators,
    SymmetrizeFidelities3
)
from qiskit_noise_learning.analysis.legacy import LegacySolve

from qiskit_noise_learning.data import RawData,AveragedData,ModelData
from qiskit_noise_learning.sequences import Path
from qiskit.quantum_info import QubitSparsePauli, QubitSparsePauliList
from qiskit_aer import QasmSimulator

from qiskit_noise_learning.qnl_test_utils import get_fid_pairs,make_gateset
from qiskit_noise_learning.models import PauliLindbladModel
from qiskit_noise_learning.circuit_generator import ExecutorCircuitGenerator

# %%
backend = FakeFez()
# %% [markdown]
# ## Load Noise Maps and Configure Program Options
# %%
# ## Function Definitions
def load_noise_maps():
    """Load the noise maps from the pickle file."""
    with open('noise_maps_nlv3_ole.pkl','rb') as f:
        noise_maps_true = pickle.load(f)
    return noise_maps_true
def load_instructions():
    """Load the noise maps from the pickle file."""
    with open('instructions_info.pkl','rb') as f:
        instructions_info = pickle.load(f)
    return instructions_info


class LSQLinearSolve(NNLSSolve):
    def _linear_solve(self, a_mat: np.ndarray, b_vec: np.ndarray) -> tuple[np.ndarray, dict]:
        opt_res = opt.lsq_linear(a_mat, b_vec, verbose=2,bounds=(0, np.inf),method='bvls')
        return opt_res.x, {"opt_res": opt_res}

noise_maps_true = load_noise_maps()
instructions_info = load_instructions()
instructions_to_learn = instructions_info['instructions']
layer_name = instructions_info['layer_name']
qubit_subset = instructions_info['qubit_subset']
gate_set = make_gateset(instructions_to_learn[layer_name], backend.target, layer_name, qubit_subset)
pauli_lindblad_model = PauliLindbladModel.k_local(gate_set, k=2)
gate_set.draw()


# %% [markdown]
# ## Setup AerExecutor with Noise
noise_maps_true['M'] = PauliLindbladMap.from_sparse_list([('X',(0,),0.0)],num_qubits=56)
noise_maps_true['P'] = PauliLindbladMap.from_sparse_list([('X',(0,),0.0)],num_qubits=56)
# %%
qasm_simulator = QasmSimulator(
    coupling_map=backend.coupling_map, seed_simulator=111
)
executor = AerExecutor(
    qasm_simulator=qasm_simulator,
    noise_dict=noise_maps_true,
    annotation_key="tag",  # Use "tag" to match the Tag annotations
    angle_decimals=3
)

# %% [markdown]
# ## Build Experiment
# %%
experiment_builder = ExperimentBuilder(fidelity_model=pauli_lindblad_model)
experiment_builder.add_path_patterns(
    standard_vanilla_pattern_generator(
        experiment_builder.gate_set["P"],
        experiment_builder.gate_set["M"],
        experiment_builder.gate_set[layer_name],
        pauli_lindblad_model.generators[layer_name],
        experiment_builder.gate_set.coupling_map,
    ),
    attempt_instruction_merge=True,
)
experiment_builder.complete()

print(f"Number of path patterns: {len(experiment_builder.path_patterns)}")
print(f"Number of instruction patterns: {len(experiment_builder.instruction_patterns)}")

# %% [markdown]
# ## Generate Instruction Sequences
# %%
shots = 100  # Reduced from config for faster testing
num_randomizations = 10  # Reduced from config
layer_pair_depths = [0, 6, 48, 64]  # Reduced from config
instruction_sequences = experiment_builder.generate_instruction_sequences(layer_pair_depths)    
print(f"Number of instruction sequences: {len(instruction_sequences)}")

# %% [markdown]
# ## Generate Circuits
num_rand = num_randomizations
circuit_generator = ExecutorCircuitGenerator(gate_set, num_randomizations=num_rand)
samplex_items, data_mapper = circuit_generator.generate(instruction_sequences)

print(f"Number of template circuits: {len(samplex_items)}")
print("Depth 2 template circuit:")
samplex_items[0].circuit.draw("mpl", idle_wires=False, fold=False)

#%% [markdown]
## Run Program and Collect Data
# %%
# I AM RUNNING THIS SIM ONLY TO HAVE A RAW DATA OUTPUT ETC THAT I CAN USE AS AN EXAMPLE
# NONE OF THE PLOTS/DATA ACTUALLY USE THE SIMULATED SHOTS FROM THIS RUN

qp = QuantumProgram(shots, samplex_items)
job = executor.run(qp)
raw_data = circuit_generator.collect(job.result(), data_mapper)

#%%
fit = Fit(
    model=pauli_lindblad_model,
    paths=[Path(p, d) for p in experiment_builder.path_patterns for d in layer_pair_depths],
)
fit[RawData] = raw_data

fidelity_analysis = AnalysisPipeline(ComputeObservables(), CurveFitObservables())

fit = fidelity_analysis.run(fit)
#%%
fid_ps_1, fid_ps_2 = get_fid_pairs(fit)
#%%
###################################
######## FITTING COMPARISON #######
###################################

def make_true_avg_data(avg_data,true_fidelity_pairs):
    """given a true fidelity pairs list, and some 'host' averaged data, make the true averaged data for
    use with testing
    """
    from qiskit_noise_learning.data import AveragedData, ObservableData
    
    obs_path_patterns = avg_data.dataset.path_pattern.data
    obs_depths = avg_data.dataset.depth.data
    obs_means = true_fidelity_pairs
    obs_stds = avg_data.dataset.to_pandas()['std'].values
    obs_time_lbs = avg_data.dataset.time_lbs.data
    obs_time_ubs = avg_data.dataset.time_ubs.data
    
    true_avg_data = AveragedData.from_arrays(
            path_patterns=np.array(obs_path_patterns, dtype=object),
            depths=np.array(obs_depths, dtype=int),
            observables=np.array(obs_means, dtype=float),
            std=np.array(obs_stds, dtype=float),
            time_lbs=np.array(obs_time_lbs,dtype="datetime64[us]"),
            time_ubs=np.array(obs_time_ubs,dtype="datetime64[us]"),
            )
    return true_avg_data
#%%
from qiskit_noise_learning.qnl_test_utils import get_fids_df,make_noise_model_comp_df,plot_model_comparison

# get the true fidelity data from the input noise model
fids_df_true = get_fids_df(noise_maps_true[layer_name].apply_layout(qubit_subset, num_qubits=backend.num_qubits),fid_ps_1, fid_ps_2)
fids_df_true['name'] = 'true'
fid_pairs_true = fids_df_true['fid_pair'].values

#%%
fitter_configs = [
                                    [True,None,None,'legacy'],
                                    # [True,None,None,'legacy2'],
                                    # [True,None,'generators','nnls'],
                                    # [True,None,'generators','lsqlinear'],
                                    [True,None,'fidelities','nnls'],
                                    [True,None,'fidelities','lsqlinear'],
                                    [True,None,'fidelities3','nnls'],
                                    [True,None,'fidelities3','lsqlinear'],
                                ]
plot_types = ['fidelities','noise model'] # or model coefficients

plt.style.use('default')

nrow = len(fitter_configs)
ncol = len(plot_types)
fig, axs = plt.subplots(nrow,ncol,figsize=(5*ncol, 5*nrow))

for axrow, fit_config in zip(axs,fitter_configs):
    no_shot_noise, input_symmetry,output_symmetry,solver = fit_config

    if no_shot_noise:
        
        true_avg_data = make_true_avg_data(fit.averaged_data,fid_pairs_true)
    
    
        solver_func = {'nnls': NNLSSolve(), 'lsqlinear': LSQLinearSolve(), 'legacy': LegacySolve()}[solver]
        print(f'{solver}, symmetrize={output_symmetry}')
        
        if output_symmetry == 'fidelities':
            # model_solver = AnalysisPipeline(NNLSSolve(), SymmetrizeFidelities(),)
            model_solver = AnalysisPipeline(solver_func, SymmetrizeFidelities(),)
        elif output_symmetry == 'generators':
            model_solver = AnalysisPipeline(solver_func, SymmetrizeGenerators(),)
        elif output_symmetry == 'fidelities3':
            model_solver = AnalysisPipeline(solver_func, SymmetrizeFidelities3(),)
        else:
            model_solver = solver_func

        fit_true= Fit(
            model=fit.model,
            paths=[Path(p, d) for p in experiment_builder.path_patterns for d in layer_pair_depths],
        )
        fit_true[AveragedData] = true_avg_data

        out = model_solver.run(fit_true)

        model_from_exact_fids = pauli_lindblad_model.to_pauli_lindblad_maps(out.model_data)
        # print('model gamma fit from exact fids, true gamma ')
        # print(model_from_exact_fids[layer_name].inverse().gamma(), noise_maps_true_sym[layer_name].inverse().gamma())
        # print(model_from_exact_fids[layer_name].inverse().gamma() - noise_maps_true_sym[layer_name].inverse().gamma())
        noise_map_fit = model_from_exact_fids
    else:
        NotImplementedError('shot noise TBD!')

    noise_maps_comp = {
                "model": noise_map_fit[layer_name],
                "actual": noise_maps_true[layer_name].apply_layout(qubit_subset, num_qubits=backend.num_qubits),
            }
    
    for ax, plot_type in zip(axrow, plot_types):
        plot_ttl_str = f'{plot_type} {solver} \n shot noise false, symmetry out {output_symmetry}'
        if plot_type == 'fidelities':
            fids_df_model = get_fids_df(noise_maps_comp['model'],fid_ps_1,fid_ps_2)
            fids_df_model['name'] = 'model'
            fids_comp_df = pd.concat([fids_df_true,fids_df_model]).pivot(index=['ps','sup','ps_conj','sup_conj'],columns='name',values=['fid','fid_conj','fid_pair'])
            fids_comp_df

            # fig, ax = plt.subplots(figsize=(5,5))
            ax = fids_comp_df['fid'].plot(x='true', y='model', alpha=0.25, ls='', marker='o', ax=ax,label='pauli')
            ax = fids_comp_df['fid_conj'].plot(x='true', y='model', alpha=0.25, ls='', marker='o', ax=ax,label='pauli conj')
            ax = fids_comp_df['fid_pair'].plot(x='true', y='model', alpha=0.25, ls='', marker='o', ax=ax, label='pauli pair')
            ax.plot(np.linspace(0.96, 1, 10), np.linspace(0.96, 1, 10), ls='--', color='black', alpha=0.5)
            ax.set_xlabel('True Fidelity')
            ax.set_ylabel('Model Fidelity')
            # ax.annotate(plot_ttl_str,(0.75,0.25),xycoords='axes fraction',fontsize=9)
            ax.set_title(plot_ttl_str)

        elif plot_type == 'noise model':
            df_comp = make_noise_model_comp_df(noise_maps_comp)
            plot_model_comparison(df_comp,ax=ax)
            ax.set_title(plot_ttl_str)
            rmse = np.sqrt(np.mean((df_comp['rate']['model'] - df_comp['rate']['actual'])**2))
            ax.annotate(f'RMS error: {rmse:.3e}',xy=(0.05,0.75),xycoords='axes fraction',fontsize=12)
plt.tight_layout()
fig
# %%
fit.averaged_data.dataset.observables


fidelities_canonical = make_canonical_fid_dict(fid_ps_1.to_pauli_list().to_labels(),fid_ps_2.to_pauli_list().to_labels(),fid_pairs_true)
conj_paulis = make_conj_pauli_list(list(fidelities_canonical.keys()),fid_ps_1.to_pauli_list().to_labels(),fid_ps_2.to_pauli_list().to_labels())
len(conj_paulis)

    
noise_map_legacy = fit_noise_model_legacy2(basis_paulis=PauliList(list(fidelities_canonical.keys())), 
                            conjugated_basis_paulis=PauliList(conj_paulis), layer_name=layer_name,num_qubits=backend.num_qubits,
                            pauli_fidelities=np.array(list(fidelities_canonical.values())))
# %%
noise_map_legacy[layer_name].inverse().gamma()
# %%
out[ModelData].dataset.parameter
# %%
type(fit_true.averaged_data.dataset.path_pattern[0])
fit_true.averaged_data.dataset.path_pattern[0].item().repeatable_fragment[0].gate.name
# %%
