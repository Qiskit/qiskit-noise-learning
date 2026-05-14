import pickle
from pathlib import Path
from copy import deepcopy

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Samplomatic
from samplomatic import Twirl
from samplomatic.annotations import InjectNoise, Tag

# Qiskit Noise Learning
from qiskit_noise_learning.gate_sets import QiskitGateSet

from qiskit_noise_learning.sequences import InstructionPattern
from qiskit.quantum_info import QubitSparsePauli, QubitSparsePauliList


def validate_tag_inject_agreement(instructions_to_learn):
    """Validate that the instructions to learn tags and inject noise match"""
    
    for key, instr in instructions_to_learn.items():
        if _tags := [annot for annot in instr.operation.annotations if isinstance(annot, Tag)]:
            print(f"Found tags: {_tags}")
            if len(_tags) > 1:
                raise ValueError(f"Too many tags: {_tags}")
            tag = _tags[0]
        else:
            print(f"Found no tags")
            tag = None
        
        if _injects := [annot for annot in instr.operation.annotations if isinstance(annot, InjectNoise)]:
            print(f"Found InjectNoise: {_injects}")
            if len(_injects) > 1:
                raise ValueError(f"Too many InjectNoise: {_injects}")
            inject = _injects[0]
        else:
            print(f"Found no injects")
            inject = None
        
        if tag and not inject:
            print('tag and not inject')
            instr.operation.annotations += [InjectNoise(ref=tag.ref)]

def validate_noise_maps_instructions_keys(instructions_to_learn,noise_maps_true):
    print("\nVerifying noise map keys match instruction refs...")
    noise_map_keys = set(noise_maps_true.keys())
    instruction_refs = set(instructions_to_learn.keys())

    print(f"  Noise map keys: {sorted(noise_map_keys)}")
    print(f"  Instruction refs: {sorted(instruction_refs)}")

    print(f"  Noise map keys: {sorted(noise_map_keys)}")
    if noise_map_keys != instruction_refs:
        missing_in_noise = instruction_refs - noise_map_keys
        extra_in_noise = noise_map_keys - instruction_refs
        if missing_in_noise:
            print(f"  ⚠ Missing in noise maps: {sorted(missing_in_noise)}")
        if extra_in_noise:
            print(f"  ⚠ Extra in noise maps: {sorted(extra_in_noise)}")
        raise ValueError("Noise map keys do not match instruction refs!")
    else:
        print("  ✓ All keys match!")
    print()

def vanilla_basis(instruction_pattern: InstructionPattern) -> QubitSparsePauli:
    prep_gate = instruction_pattern.start_fragment[0].gate
    permutation = instruction_pattern.start_fragment[1]
    prepped_qs = list(prep_gate.prep_idxs)
    pauli = QubitSparsePauli.from_sparse_label(("Z" * len(prepped_qs), prepped_qs), num_qubits=permutation.num_qubits)
    pauli = permutation.propagate(pauli)
    return pauli

def vanilla_bases(instruction_patterns: list[InstructionPattern]) -> QubitSparsePauliList:
    return QubitSparsePauliList.from_qubit_sparse_paulis([vanilla_basis(x) for x in instruction_patterns])

def print_vanilla_bases(instruction_patterns):
    return [p.replace('I','') for p in vanilla_bases(instruction_patterns).to_pauli_list().to_labels()]


def get_fid_pairs(fit):
    """Get the fits to the fidelity pairs as a Dataframe"""
    fid_pairs = []

    for pp in fit.averaged_data.dataset.observables.path_pattern.data:
        fid_pairs.append([rf.pauli for rf in pp.repeatable_fragment])

    # Convert to QubitSparsePauliList
    fid_ps_1 = QubitSparsePauliList(np.array(fid_pairs)[:,0].tolist())
    fid_ps_2 = QubitSparsePauliList(np.array(fid_pairs)[:,1].tolist())
    return fid_ps_1, fid_ps_2

def get_fid_pair_fits(fit):
    
    fid_ps_1, fid_ps_2 = get_fid_pairs(fit)
    fid_pair_fid_fits = fit.averaged_data.dataset.observables.data

    df_fid_pair_fits = pd.concat([
        pd.DataFrame(fid_ps_1.to_sparse_list(), columns=['ps','sup']),
        pd.DataFrame(fid_ps_2.to_sparse_list(), columns=['ps_conj','sup_conj'])
    ], axis=1)
    df_fid_pair_fits['fid_pair'] = fid_pair_fid_fits
    df_fid_pair_fits['sup'] = df_fid_pair_fits['sup'].apply(tuple)
    df_fid_pair_fits['sup_conj'] = df_fid_pair_fits['sup_conj'].apply(tuple)
    df_fid_pair_fits['name'] = 'fit'
    return df_fid_pair_fits

def get_fids_df(noise_map, fid_ps_1, fid_ps_2):
    fids_df = pd.concat([
        pd.DataFrame(fid_ps_1.to_sparse_list(), columns=['ps','sup']),
        pd.DataFrame(fid_ps_2.to_sparse_list(), columns=['ps_conj','sup_conj'])
    ], axis=1)
    fids_df['sup'] = fids_df['sup'].apply(tuple)
    fids_df['sup_conj'] = fids_df['sup_conj'].apply(tuple)
    # fids_df['name'] = name
    fids = [noise_map.pauli_fidelity(p) for p in fid_ps_1]
    fids_df['fid'] = fids
    fids_conj = [noise_map.pauli_fidelity(p) for p in fid_ps_2]
    fids_df['fid_conj'] = fids_conj
    fids_df['fid_pair'] = np.array(fids) * np.array(fids_conj)
    fids_df['fid_diff'] = fids_df['fid'] - fids_df['fid_conj']
    return fids_df

def make_gateset(circuit_instruction, target, layer_name, qubit_list):
    """Create a QiskitGateSet from a circuit instruction."""
    circuit_instruction_no_noise = deepcopy(circuit_instruction)
    circuit_instruction_no_noise.operation.annotations = [
        annot
        for annot in circuit_instruction_no_noise.operation.annotations
        if not isinstance(annot, InjectNoise)
    ]

    _twirl_annots = [
        annot for annot in circuit_instruction.operation.annotations
        if isinstance(annot, Twirl)
    ]
    if len(_twirl_annots) != 1:
        raise ValueError(f"Unexpected number of twirling annotations: {_twirl_annots=}")
    twirl = _twirl_annots[0]

    gateset = QiskitGateSet(
        target=target, 
        qubit_subset=qubit_list,
        add_default_spam=False,
    )
    gateset.add_box_as_gate(
        box_instr=circuit_instruction_no_noise,
        name=layer_name,
    )
    gateset.add_measurement(name="M", qubit_idxs=qubit_list, annotations=[twirl])
    gateset.add_preparation(name="P", qubit_idxs=qubit_list, annotations=[twirl])
    return gateset


def _sorted_data(averaged_data):
    """Sort averaged data by path pattern and depth."""
    dataset = averaged_data.dataset

    sorted_data = {}
    for path_pattern, depth, val in zip(
        dataset["path_pattern"].data, dataset["depth"].data, dataset["observables"].data
    ):
        if depth < 0:
            continue
        this_data = sorted_data.setdefault(path_pattern, ([], []))
        this_data[0].append(val)
        this_data[1].append(depth)

    return {k: v for k, v in sorted_data.items() if len(v[0]) > 1}  # filter on decays


def fidelity_label(path_pattern, subset):
    """Generate fidelity label for plotting."""
    return (
        "$f_{"
        + "} f_{".join(
            "".join(str(fid.pauli.to_pauli()[subset])) for fid in path_pattern.repeatable_fragment
        )
        + "} $"
    )

def plot(averaged_data, subsets, num_cols=3):
    """Plot empirical decays."""
    sorted_data = _sorted_data(averaged_data)

    plots = {}
    for subset in subsets:
        plots[subset] = []
        for path_pattern in sorted_data.keys():
            if set(subset).issuperset(path_pattern.start_fragment[0].out_bit_indices):
                plots[subset].append(path_pattern)

    num_figs = len(plots)
    num_rows = num_figs // num_cols

    fig, axs = plt.subplots(num_rows, num_cols)
    flag = True
    for ax, (subset, path_patterns) in zip(axs.flat, plots.items()):
        ax.set_title(f"{subset}")
        for path_pattern in path_patterns:
            this_data = sorted_data[path_pattern]
            label = fidelity_label(path_pattern, [s for s in subset]) if flag else None
            ax.plot(
                this_data[1],
                this_data[0],
                "--",
                label=label,
            )
        flag = False

    fig.legend(bbox_to_anchor=(1.15, 0), loc="lower right")
    plt.tight_layout()


def plot_with_model_decay(
    pauli_lindblad_map, observable_data, subsets, num_cols=3, start_idx=None, end_idx=None
):
    """Plot decays vs model prediction."""
    sorted_data = _sorted_data(observable_data)

    plots = {}
    for subset in subsets:
        plots[subset] = []
        for path_pattern in sorted_data.keys():
            if set(subset).issuperset(path_pattern.start_fragment[0].out_bit_indices):
                plots[subset].append(path_pattern)

    num_figs = len(plots)
    num_rows = num_figs // num_cols

    fig, axs = plt.subplots(num_rows, num_cols)
    flag = True
    for ax, (subset, path_patterns) in zip(axs.flat, plots.items()):
        ax.set_title(f"{subset}")
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(path_patterns)
        for color_code, path_pattern in list(zip(mcolors.TABLEAU_COLORS.values(), path_patterns))[
            start_idx:end_idx
        ]:
            this_data = sorted_data[path_pattern]
            label = fidelity_label(path_pattern, [s for s in subset]) if flag else None
            ax.plot(
                this_data[1],
                this_data[0],
                ".",
                label=label,
                color=color_code,
            )
            pauli_in = path_pattern.repeatable_fragment[0].transition[0]
            fidelity_in = pauli_lindblad_map.pauli_fidelity(pauli_in)
            pauli_out = path_pattern.repeatable_fragment[0].transition[1]
            fidelity_out = pauli_lindblad_map.pauli_fidelity(pauli_out)
            ax.plot(
                np.arange(max(this_data[1])),
                (fidelity_in * fidelity_out) ** (np.arange(max(this_data[1]))),
                "-",
                label=label,
                color=color_code,
            )
        flag = False

    fig.legend(bbox_to_anchor=(1.15, 0), loc="lower right")
    plt.tight_layout()

def make_noise_model_comp_df(noise_maps_comp):
    
    _data_comp = []
    for name, plm in noise_maps_comp.items():
        for ps, sup, rate in plm.to_sparse_list():
            sup = tuple(sup)
            _data_comp.append({
                "name": name,
                "ps": ps,
                "sup": sup,
                "rate": rate,
            })
    df_comp = pd.DataFrame(_data_comp)
    df_comp = df_comp.pivot(columns=["name"], index=["sup", "ps"], values=["rate"]).reset_index()

    return df_comp

def plot_model_comparison(df_comp,ax=None):
    # Separate zero and non-zero values
    df_nonzero = df_comp[(df_comp['rate']['actual'] > 0) & (df_comp['rate']['model'] > 0)]
    df_zero_actual = df_comp[df_comp['rate']['actual'] <= 0]
    df_zero_model = df_comp[df_comp['rate']['model'] <= 0]

    # Create the main scatter plot with non-zero values
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    df_nonzero['rate'].plot(x='actual', y='model', alpha=0.25, ls='', marker='o', ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-7, 1e-2)
    ax.set_ylim(1e-7, 1e-2)
    ax.set_xlabel('True rate')
    ax.set_ylabel('Learned rate')
    ax.plot(np.linspace(1e-5, 1e-1, 50), np.linspace(1e-5, 1e-1, 50), ls='--', color='black', alpha=0.5)

    # Add markers for zero values on the axes
    # For points where actual is zero but model is not, place on y-axis
    if len(df_zero_actual) > 0:
        y_vals = df_zero_actual['rate']['model'].values
        y_vals_nonzero = y_vals[y_vals > 0]
        if len(y_vals_nonzero) > 0:
            ax.scatter([1e-7] * len(y_vals_nonzero), y_vals_nonzero,
                    marker='<', s=50, color='red', alpha=0.5, label='Actual<=0')

    # For points where model is zero but actual is not, place on x-axis
    if len(df_zero_model) > 0:
        x_vals = df_zero_model['rate']['actual'].values
        x_vals_nonzero = x_vals[x_vals > 0]
        if len(x_vals_nonzero) > 0:
            ax.scatter(x_vals_nonzero, [1e-7] * len(x_vals_nonzero),
                    marker='v', s=50, color='blue', alpha=0.5, label='Model<=0')

    # Add legend if there are zero values
    if len(df_zero_actual) > 0 or len(df_zero_model) > 0:
        ax.legend()
    return ax 

def model_diff_qq_plot(nm_diff):
    fig, ax = plt.subplots()
    import scipy
    scipy.stats.probplot(nm_diff, dist="norm", plot=ax)
    plt.title("Q-Q Plot")
    plt.ylim(-3e-3,3e-3)
    plt.show()

def fidelities_comparison(noise_maps_comp, fid_ps_1, fid_ps_2):
    fig, axs = plt.subplots(1,2,figsize=(2*5,5))
    fids_dfs = []

    for name in ["model", "actual"]:
        
        fids_df = pd.DataFrame(fid_ps_1.to_sparse_list(), columns=['ps','sup'])
        fids_df['sup'] = fids_df['sup'].apply(tuple)
        fids = [noise_maps_comp[name].pauli_fidelity(p) for p in fid_ps_1]
        fids_df['name'] = name
        fids_df['fid'] = fids
        fids_dfs.append(fids_df)

    df_fids = pd.concat(fids_dfs)
    df_fids = df_fids.pivot(columns=["name"], index=["sup", "ps"], values=["fid"])
    df_fids = df_fids.reset_index()

    ax = df_fids['fid'].plot(x='actual', y='model', alpha=0.25, ls='', marker='o', ax=axs[0])
    ax.plot(np.linspace(0.97, 1, 10), np.linspace(0.97, 1, 10), ls='--', color='black', alpha=0.5)
    ax.set_xlabel('True Fidelity')
    ax.set_ylabel('Model Fidelity')
    fids_dfs = []
    for name in ["model", "actual"]:
        
        fids_df = pd.DataFrame(fid_ps_2.to_sparse_list(), columns=['ps','sup'])
        fids_df['sup'] = fids_df['sup'].apply(tuple)
        fids = [noise_maps_comp[name].pauli_fidelity(p) for p in fid_ps_2]
        fids_df['name'] = name
        fids_df['fid_conj'] = fids
        fids_dfs.append(fids_df)

    df_fids_conj = pd.concat(fids_dfs)
    df_fids_conj = df_fids_conj.pivot(columns=["name"], index=["sup", "ps"], values=["fid_conj"])
    df_fids_conj = df_fids_conj.reset_index()

    ax = df_fids_conj['fid_conj'].plot(x='actual', y='model', alpha=0.25, ls='', marker='o', ax=axs[1])
    ax.plot(np.linspace(0.97, 1, 10), np.linspace(0.97, 1, 10), ls='--', color='black', alpha=0.5)
    ax.set_xlabel('True Fidelity Conj.')
    ax.set_ylabel('Model Fidelity Conj.')
    return df_fids, df_fids_conj

def fid_pairs_comparison(df_fid_pair_fits,noise_maps_comp,fid_ps_1,fid_ps_2):

    fids_dfs = []
    for name in ["model", "actual"]:
        
        fids_df = pd.concat([
            pd.DataFrame(fid_ps_1.to_sparse_list(), columns=['ps','sup']),
            pd.DataFrame(fid_ps_2.to_sparse_list(), columns=['ps_conj','sup_conj'])
        ], axis=1)
        fids_df['sup'] = fids_df['sup'].apply(tuple)
        fids_df['sup_conj'] = fids_df['sup_conj'].apply(tuple)
        fids_df['name'] = name
        fids = [noise_maps_comp[name].pauli_fidelity(p) for p in fid_ps_1]
        fids_df['fid'] = fids
        fids_conj = [noise_maps_comp[name].pauli_fidelity(p) for p in fid_ps_2]
        fids_df['fid_conj'] = fids_conj
        fids_df['fid_pair'] = np.array(fids) * np.array(fids_conj)
        fids_dfs.append(fids_df)

    if df_fid_pair_fits is not None:
        df_fid_pairs = pd.concat(fids_dfs + [df_fid_pair_fits])
    else:
        df_fid_pairs = pd.concat(fids_dfs)
    df_fid_pairs = df_fid_pairs.pivot(columns=["name"], index=["sup", "ps",'sup_conj','ps_conj'], values=["fid_pair"])
    df_fid_pairs = df_fid_pairs.reset_index()

    fig, axs = plt.subplots(1, 3, figsize=(5*3, 5))

    for ax, x, y in zip(axs, ['actual','actual','fit'], ['fit','model','model']):
        df_fid_pairs['fid_pair'].plot(x=x, y=y, alpha=0.25, ls='', marker='o', ax=ax)
        ax.plot(np.linspace(0.95, 1, 10), np.linspace(0.95, 1, 10), ls='--', color='black', alpha=0.5)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    fig.suptitle('Fidelities Agreement')
    return df_fid_pairs

