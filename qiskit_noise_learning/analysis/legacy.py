from collections import defaultdict
from typing import Literal
import numpy as np
import scipy.optimize as opt
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList
from scipy.sparse import csr_array

from qiskit.quantum_info import QubitSparsePauliList,PauliLindbladMap

from qiskit_noise_learning.analysis import Fit
from qiskit_noise_learning.data import AveragedData, ModelData

from qiskit_noise_learning.analysis.model_solve import ModelSolve
from qiskit_noise_learning.data.xarray_utils import time_bound

OptimizerLiteral = Literal["nnls", "lsq_linear_sparse", "cvxpy"]
NoiseAssumptionLiteral = Literal["symmetric_fidelities", "symmetric_generators"]

def get_fid_pairs(fit):
    """extracts the first and second paulis from the 'repeatable fragments' of the path patterns (i.e. pauli and conjugate pauli)"""
    fid_pairs = []

    for pp in fit.averaged_data.dataset.observables.path_pattern.data:
        fid_pairs.append([rf.pauli for rf in pp.repeatable_fragment])

    # Convert to QubitSparsePauliList
    fid_ps_1 = QubitSparsePauliList(np.array(fid_pairs)[:,0].tolist())
    fid_ps_2 = QubitSparsePauliList(np.array(fid_pairs)[:,1].tolist())
    return fid_ps_1, fid_ps_2


def make_canonical_fid_dict(fid_ps_1,fid_ps_2,fid_pairs_data):
    """
    Args: fid_ps_1, fid_ps2 are lists of dense Pauli strings, representing the first and second elements
            of the repeatable fragments in the path patterns
    Returns:
            dict with keys being the canoncial fidelities (unique fidelities with weight < 3)
            values being the pair fidelity. This is the input to the legacy fitter
    """
    from collections import defaultdict
    # this converts the the data of fidelity pairs to a data set of fidelities (with values of fidelity decay per pair depth) 
    fid_pair_p_dict = defaultdict(list)
    for i,p in enumerate(fid_ps_1):
        if len(p.replace('I','')) < 3:
            fid_pair_p_dict[p].append(fid_pairs_data[i])
    for i,p in enumerate(fid_ps_2):
        if len(p.replace('I','')) < 3:
            fid_pair_p_dict[p].append(fid_pairs_data[i])
    len(fid_pair_p_dict)

    fid_pair_p_dict_mean = {}
    for p, evlist in fid_pair_p_dict.items():
        fid_pair_p_dict_mean[p] = np.mean(evlist)
    return fid_pair_p_dict_mean

def make_conj_pauli_list(pauli_list, fid_ps_1, fid_ps_2):
    """Given the input pauli list, return the conjugate pauli list"""
    pconj_list = []
    for p in pauli_list:
        if p in fid_ps_1:
            p_conj = fid_ps_2[fid_ps_1.index(p)]
            pconj_list.append(p_conj)
        elif p in fid_ps_2:
            p_conj = fid_ps_1[fid_ps_2.index(p)]
            pconj_list.append(p_conj)
    return pconj_list

def fit_noise_model_legacy(fit,
                            noise_assumption = 'symmetric_fidelities',
                            decimals: int | None = None,
                            optimizer_name = "nnls",
                            constrained: bool = True):
    """Do the legacy noise model fit"""
    fid_ps_1, fid_ps_2 = get_fid_pairs(fit)
    fid_pair_data = fit.averaged_data.dataset.observables
    fidelities_canonical = make_canonical_fid_dict(fid_ps_1.to_pauli_list().to_labels(),
                                                                  fid_ps_2.to_pauli_list().to_labels(),
                                                                  fid_pair_data)
    pauli_fidelities = np.array(list(fidelities_canonical.values()))
    basis_paulis=PauliList(list(fidelities_canonical.keys()))
    conjugated_basis_paulis = PauliList(make_conj_pauli_list(list(fidelities_canonical.keys()),
                                                        fid_ps_1.to_pauli_list().to_labels(),
                                                        fid_ps_2.to_pauli_list().to_labels()))

    
    sparse_model_paulis = basis_paulis.copy()
    # Form the non-commuting array (``M`` as in the PEC paper)
    nc_array_basis = np.logical_not([p.commutes(sparse_model_paulis) for p in basis_paulis]).astype(
        int
    )
    nc_array_conj_basis = np.logical_not(
        [p.commutes(sparse_model_paulis) for p in conjugated_basis_paulis]
    ).astype(int)
    nc_array = nc_array_basis + nc_array_conj_basis

    # Set up the least-squares problem according to the selected assumption:
    if noise_assumption == "symmetric_fidelities":
        # assumption lets us compute the layer fidelities:
        layer_fidelities = np.tile(np.sqrt(pauli_fidelities), 2)
        nc_array_shaped = np.concatenate([nc_array_basis, nc_array_conj_basis])
        fit_vector = -np.log(np.abs(layer_fidelities)) / 2
    elif noise_assumption == "symmetric_generators":
        # identify which generators are conjugate pairs:
        conjugated_generators = conjugated_basis_paulis.copy()

        conj_pairs = []
        for i, p in enumerate(sparse_model_paulis):
            j = np.where(conjugated_generators[i + 1 :].equiv(p))[0]
            if len(j):
                conj_pairs.append([i, j[0] + i + 1])
        conj_pairs = np.array(conj_pairs)
        # We will solve Ax = B. x is generator rates ("lambda").
        # For conj. pair (P1 P2) we fix x_1 = x_2.
        # Instead of solving for x_1 and x_2, we solve for (x_1 + x_2).
        # This makes the x vector shorter:
        #     replace x_1 with (x_1 + x_2), and delete x_2,
        # and removes columns from `nc_array_shaped`:
        #     replace col 1 with (col 1 + col 2), and delete col 2.
        # Solving Ax = b will now include the value of (x_1 + x_2).
        # Then later, we will use the symmetry assumption
        # x_1 = x_2 = (x_1 + x_2)/2
        # to construct the full list of generator rates.
        nc_array_shaped = nc_array.copy()
        nc_array_shaped[:, conj_pairs[:, 0]] += nc_array_shaped[:, conj_pairs[:, 1]]
        nc_array_shaped = np.delete(nc_array_shaped, conj_pairs[:, 1], axis=1)
        fit_vector = -np.log(np.abs(pauli_fidelities)) / 2
    else:
        raise ValueError(f"Noise assumption {noise_assumption} not recognized")

    # Perform the least squares fitting:
    if optimizer_name == "lsq_linear_sparse":
        nc_array_shaped = csr_array(nc_array_shaped)
        sparse_model_coeffs = opt.lsq_linear(
            nc_array_shaped, fit_vector, bounds=(0, np.inf) if constrained else (-np.inf,np.inf)
        ).x
    elif optimizer_name == "nnls":
        if not constrained:
            raise ValueError("...")
        sparse_model_coeffs, _ = opt.nnls(nc_array_shaped, fit_vector)
    elif optimizer_name == "cvxpy":
        try:
            import cvxpy
        except ImportError as err:
            raise ValueError(
                "Attempted to use `cvxpy` for sparse model fitting, but it is not installed."
            ) from err

        nc_array_shaped = csr_array(nc_array_shaped)
        cvxpy_fit_var = cvxpy.Variable(nc_array_shaped.shape[1])
        cost = cvxpy.sum_squares(nc_array_shaped @ cvxpy_fit_var - fit_vector)
        prob = cvxpy.Problem(
            cvxpy.Minimize(cost),
            constraints=[cvxpy_fit_var >= 0] if constrained else None,
        )
        prob.solve()
        sparse_model_coeffs = cvxpy_fit_var.value
    else:
        raise ValueError(f"Optimizer name {optimizer_name} not recognized.")

    if noise_assumption == "symmetric_generators":
        indices_to_restore = np.sort(conj_pairs[:, 1])
        sparse_model_coeffs = np.insert(
            sparse_model_coeffs,
            indices_to_restore - np.arange(len(indices_to_restore)),
            values=0,
            axis=0,
        )
        sparse_model_coeffs[conj_pairs[:, 1]] = sparse_model_coeffs[conj_pairs[:, 0]]

    # Discard small terms
    if decimals is not None:
        sparse_model_coeffs = np.round(sparse_model_coeffs, decimals)

    sparse_model_coeffs_residuals = (nc_array @ sparse_model_coeffs) - (
        -np.log(np.abs(pauli_fidelities)) / 2
    )
    sparse_model_noncommuting = nc_array

    # Check that the arrays have the appropriate shape
    assert len(sparse_model_paulis) == len(sparse_model_coeffs)
    assert sparse_model_noncommuting.shape[1] == len(sparse_model_paulis)

    noise_map_pecr = PauliLindbladMap.from_list(list(zip(sparse_model_paulis.to_labels(),sparse_model_coeffs)))
    return noise_map_pecr

class LegacySolve(ModelSolve):
    def _linear_solve(self, a_mat: np.ndarray, b_vec: np.ndarray) -> tuple[np.ndarray, dict]:
            # this is required but I don't use it (should I use it somehow?)
            return True
            # opt_res = opt.lsq_linear(a_mat, b_vec, verbose=2,bounds=(0, np.inf),method='bvls')
            # return opt_res.x, {"opt_res": opt_res}
    def _run(self, fit: Fit):

        averaged_data = fit[AveragedData]

        noise_map = fit_noise_model_legacy(fit,
                            
                            noise_assumption = 'symmetric_fidelities',
                            decimals = None,
                            optimizer_name = "nnls",
                            constrained = True)

        from qiskit_noise_learning.models.pauli_lindblad_model import GeneratorIndex
        layer_name = fit.averaged_data.dataset.path_pattern[0].item().repeatable_fragment[0].gate.name

        param_labels = [GeneratorIndex(gate_name=layer_name,generator=g) for g in noise_map.generators()]
        x = np.array(noise_map.rates)
        cov_x = np.zeros(shape=(len(x),len(x)))
        metadata = {}
        # Filter to decay data (depth == -1)
        decay_mask = averaged_data.dataset["depth"].data == -1
        decay_dataset = averaged_data.dataset.sel({"observable": decay_mask})

        time_lb = time_bound(decay_dataset["time_lbs"].data, "min")
        time_ub = time_bound(decay_dataset["time_ubs"].data, "max")

        
        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=param_labels,
            parameter_values=x,
            covariance=cov_x,
            time_lbs=np.full(len(x), time_lb, dtype="datetime64[us]"),
            time_ubs=np.full(len(x), time_ub, dtype="datetime64[us]"),
            metadata=metadata,
        )
