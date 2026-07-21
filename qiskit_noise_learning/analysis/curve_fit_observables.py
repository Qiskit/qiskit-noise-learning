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

import warnings

import numpy as np
import scipy.optimize as opt

from qiskit_noise_learning.analysis import AnalysisStage
from qiskit_noise_learning.analysis.average_observables import average_observables
from qiskit_noise_learning.data import AveragedData, ObservableData
from qiskit_noise_learning.data.xarray_utils import time_bound


class CurveFitObservables(AnalysisStage):
    """Fit observable data to exponential decays of the form ``a * f**fragment_depth``, and average
    any remaining observables over randomizations.

    This stage will curve fit data for any unbound paths in ``fit.paths``. If ``fit.paths is None``,
    any path in the data with multiple fragment depths will be curve fit. In both cases, any
    remaining paths will be averaged.
    """

    @property
    def input_level(self):
        return ObservableData

    @property
    def output_level(self):
        return AveragedData

    def _run(self, fit):
        observable_data = fit.observable_data
        dataset = observable_data.dataset
        unique_unbound_paths = list(set(dataset["unbound_path"].data))

        # Determine which paths should be curve-fit vs averaged
        curve_fit_paths = {p for p in fit.paths if p.is_unbound} if fit.paths else set()

        # for accumulating exponential fits
        decay_paths = []
        decay_fidelities = []
        decay_fidelity_stds = []
        spam_fidelities = []
        spam_fidelity_stds = []
        chi_squareds = []
        reduced_chi_squareds = []
        decay_time_lbs_out = []
        decay_time_ubs_out = []

        # for accumulating single-fragment-depth paths
        single_fragment_depth_paths = set()

        for path in unique_unbound_paths:
            path_mask = dataset["unbound_path"].data == path
            path_dataset = dataset.sel({"observable": path_mask})

            unique_fragment_depths = sorted(set(path_dataset["fragment_depth"].data))

            if curve_fit_paths:
                if path not in curve_fit_paths:
                    single_fragment_depth_paths.add(path)
                    continue
                if len(unique_fragment_depths) <= 1:
                    raise ValueError(
                        f"Cannot curve-fit path with data at {len(unique_fragment_depths)} "
                        "fragment depth(s). At least 2 fragment depths are required."
                    )
            elif len(unique_fragment_depths) <= 1:
                single_fragment_depth_paths.add(path)
                continue

            fragment_depths_list = []
            means_list = []
            stds_list = []

            for fragment_depth in unique_fragment_depths:
                fragment_depth_mask = path_dataset["fragment_depth"].data == fragment_depth
                values = path_dataset["observables"].data[fragment_depth_mask].flatten()
                values = values[~np.isnan(values)]

                mean = float(np.mean(values))
                if values.size == 1:
                    p = (mean + 1) / 2
                    std = np.sqrt(p * (1 - p))
                else:
                    std = np.std(values, ddof=1) / np.sqrt(values.size)

                fragment_depths_list.append(fragment_depth)
                means_list.append(mean)
                stds_list.append(std)

            fragment_depths_arr = np.array(fragment_depths_list, dtype=float)
            means_arr = np.array(means_list, dtype=float)
            stds_arr = np.array(stds_list, dtype=float)

            a, f, a_std, f_std, chisq = fit_exponential(fragment_depths_arr, means_arr, stds_arr)

            # Reduced chi-squared normalizes by the degrees of freedom (M fragment depths minus the
            # 2 fit parameters a, f). It is nan when the raw value is undefined or there are no
            # degrees of freedom to normalize by.
            dof = len(unique_fragment_depths) - 2
            reduced_chisq = chisq / dof if dof > 0 else float("nan")

            decay_paths.append(path)
            spam_fidelities.append(a)
            decay_fidelities.append(f)
            spam_fidelity_stds.append(a_std)
            decay_fidelity_stds.append(f_std)
            chi_squareds.append(chisq)
            reduced_chi_squareds.append(reduced_chisq)

            decay_time_lbs_out.append(time_bound(path_dataset["time_lbs"].data, "min"))
            decay_time_ubs_out.append(time_bound(path_dataset["time_ubs"].data, "max"))

        decay_data = AveragedData.from_arrays(
            unbound_paths=decay_paths,
            fragment_depths=np.array([-1] * len(decay_paths), dtype=int),
            observables=np.array(decay_fidelities),
            std=np.array(decay_fidelity_stds),
            time_lbs=np.array(decay_time_lbs_out, dtype="datetime64[us]"),
            time_ubs=np.array(decay_time_ubs_out, dtype="datetime64[us]"),
            metadata=[
                {
                    "spam_fidelity": sf,
                    "spam_fidelity_std": sf_std,
                    "chi_squared": chi_squared,
                    "reduced_chi_squared": reduced_chi_squared,
                }
                for sf, sf_std, chi_squared, reduced_chi_squared in zip(
                    spam_fidelities, spam_fidelity_stds, chi_squareds, reduced_chi_squareds
                )
            ],
        )

        single_fragment_depth_data = average_observables(
            observable_data=observable_data, unique_unbound_paths=single_fragment_depth_paths
        )

        fit[AveragedData] = decay_data.merge(single_fragment_depth_data)


def fit_exponential(
    fragment_depths: np.ndarray, y_data: np.ndarray, y_err: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Fit ``y = a * f**fragment_depth`` and return ``(a, f, a_std, f_std, chi_sq)``.

    Falls back to fitting without uncertainty weights if ``curve_fit`` fails.
    """

    def _decay_fn(fragment_depth, a, f):
        return a * (f**fragment_depth)

    # Initial parameter guesses
    min_fragment_depth = fragment_depths.min()
    mask = (y_data > 0) & (fragment_depths > 0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        spam_guess = (
            np.median(y_data[fragment_depths == min_fragment_depth])
            if (fragment_depths == min_fragment_depth).any()
            else 0.9
        )
        decay_guess = (
            np.nanmedian(y_data[mask] ** (1.0 / fragment_depths[mask])) if mask.any() else 0.9
        )
    spam_guess = float(np.nan_to_num(spam_guess, nan=0.9))
    decay_guess = float(np.nan_to_num(decay_guess, nan=0.9))
    spam_guess = float(np.clip(spam_guess, 1e-10, 1 - 1e-10))
    decay_guess = float(np.clip(decay_guess, 1e-10, 1 - 1e-10))
    p0 = [spam_guess, decay_guess]

    def _clip_sigma(sigma):
        atol = 1e-10
        if np.allclose(sigma, 0, atol=atol):
            return None
        return np.clip(sigma, atol, None)

    def _run_curve_fit(sigma):
        return opt.curve_fit(
            f=_decay_fn,
            xdata=fragment_depths,
            ydata=y_data,
            p0=p0,
            bounds=(0, 1),
            sigma=sigma,
            absolute_sigma=True,
            full_output=True,
            check_finite=True,
        )

    clipped_err = _clip_sigma(y_err)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
            popt, pcov, _, _, _ = _run_curve_fit(clipped_err)
    except RuntimeError:
        popt, pcov, _, _, _ = _run_curve_fit(None)
        clipped_err = None

    perr = np.sqrt(np.diag(pcov))
    residuals = y_data - _decay_fn(fragment_depths, *popt)
    if clipped_err is None:
        chi_sq = float("nan")
    else:
        chi_sq = float(np.sum((residuals / clipped_err) ** 2))

    return float(popt[0]), float(popt[1]), float(perr[0]), float(perr[1]), chi_sq
