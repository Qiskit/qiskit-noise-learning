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

"""Per-layer legacy noise-model fitter.  Extends the single-gate legacy method to gate sets
with multiple unitary layers by solving each layer independently and combining the results.
"""

import numpy as np

from qiskit_noise_learning.analysis import AnalysisStage, Fit
from qiskit_noise_learning.data import AggregatedObservableData, ModelData
from qiskit_noise_learning.data.xarray_utils import time_bound
from qiskit_noise_learning.models import GeneratorIndex
from qiskit_noise_learning.sequences import Path

from .legacy import fit_noise_model_legacy


def _row_gate_name(path: Path) -> str:
    if len(path.repeatable_fragment) == 0:
        raise ValueError(
            "PerLayerLegacySolve requires every observable to have a non-empty "
            "repeatable_fragment to determine its layer; encountered a path with an empty "
            "repeatable_fragment."
        )
    return path.repeatable_fragment[0].gate_name


class PerLayerLegacySolve(AnalysisStage):
    """Solves for the :class:`~.ModelData` using the legacy pair-fidelity method, applied
    independently to each gate layer.

    For each distinct gate name found in the :class:`~.AggregatedObservableData`, this stage
    partitions the observable rows by that gate name, runs :func:`~.fit_noise_model_legacy`
    with ``noise_assumption="symmetric_fidelities"``, ``optimizer_name="nnls"``, and
    ``constrained=True``, then concatenates all per-layer results into a single
    :class:`~.ModelData`.

    This is a strict generalization of :class:`~.LegacySolve`: on a single-layer gate set
    it produces the same output.  On a multi-layer gate set it assigns each
    :class:`~.GeneratorIndex` to its correct gate layer, whereas :class:`~.LegacySolve`
    would mislabel all generators with the first-seen gate name.

    If any layer violates the legacy-learner assumptions (wrong repeatable-fragment length,
    single-qubit Cliffords required, or inconsistent conjugate fidelities), the entire solve
    raises.  There is no per-layer skip or warning.

    Layer order in the output :class:`~.ModelData` follows first-seen order in the observable
    dataset, which is deterministic for a given :class:`~.AggregatedObservableData`.
    """

    input_level = AggregatedObservableData
    output_level = ModelData

    def _run(self, fit: Fit) -> None:
        aggregated_data = fit[AggregatedObservableData]
        dataset = aggregated_data.dataset

        paths = dataset["unbound_path"].data
        gate_names = np.array([_row_gate_name(p) for p in paths], dtype=object)

        all_labels: list[GeneratorIndex] = []
        all_rates: list[float] = []
        all_time_lbs: list[np.datetime64] = []
        all_time_ubs: list[np.datetime64] = []

        for name in dict.fromkeys(gate_names):
            mask = gate_names == name
            layer_data = AggregatedObservableData(dataset.sel({"observable": mask}))

            noise_map = fit_noise_model_legacy(
                layer_data,
                noise_assumption="symmetric_fidelities",
                decimals=None,
                optimizer_name="nnls",
                constrained=True,
            )

            layer_labels = [
                GeneratorIndex(gate_name=name, generator=g) for g in noise_map.generators()
            ]
            layer_rates = list(noise_map.rates)

            decay_mask = layer_data.dataset["fragment_depth"].data == -1
            decay_ds = layer_data.dataset.sel({"observable": decay_mask})
            # time_bound returns NaT when no decay rows exist for this layer — acceptable.
            time_lb = time_bound(decay_ds["time_lbs"].data, "min")
            time_ub = time_bound(decay_ds["time_ubs"].data, "max")

            all_labels.extend(layer_labels)
            all_rates.extend(layer_rates)
            all_time_lbs.extend([time_lb] * len(layer_labels))
            all_time_ubs.extend([time_ub] * len(layer_labels))

        # If the fit carries a full model, pad any generators not covered by the solved layers
        # (e.g. P/M gates absent from vanilla paths) with rate 0.0 and NaT time bounds so that
        # predicted_path_decays can build a complete parameter vector for model-prediction plots.
        if fit.model is not None:
            solved = set(all_labels)
            nat = np.datetime64("NaT")
            for gate_name, paulis in fit.model.generators.items():
                for g in paulis:
                    idx = GeneratorIndex(gate_name=gate_name, generator=g)
                    if idx not in solved:
                        all_labels.append(idx)
                        all_rates.append(0.0)
                        all_time_lbs.append(nat)
                        all_time_ubs.append(nat)

        x = np.array(all_rates)
        cov_x = np.zeros((len(x), len(x)))
        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=all_labels,
            parameter_values=x,
            covariance=cov_x,
            time_lbs=np.array(all_time_lbs, dtype="datetime64[us]"),
            time_ubs=np.array(all_time_ubs, dtype="datetime64[us]"),
            metadata={},
        )
