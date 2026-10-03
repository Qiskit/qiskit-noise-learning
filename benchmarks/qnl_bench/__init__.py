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

"""Performance benchmarks for ``qiskit-noise-learning``.

This package is not part of the shipped library and is not imported by it. See ``README.md`` in
the ``benchmarks`` directory for how to run the suite.
"""

from .build import BuildResult, timed_build
from .cases import SUITE, BenchmarkCase, build_gate_set, build_model, case_by_name
from .fingerprint import experiment_fingerprint
from .instrument import CallRecord, ComponentTimers, StageTimeline
from .lattices import TOPOLOGIES, grid_coupling_map, heavy_hex_coupling_map, layer_couplings

__all__ = [
    "SUITE",
    "TOPOLOGIES",
    "BenchmarkCase",
    "BuildResult",
    "CallRecord",
    "ComponentTimers",
    "StageTimeline",
    "build_gate_set",
    "build_model",
    "case_by_name",
    "experiment_fingerprint",
    "grid_coupling_map",
    "heavy_hex_coupling_map",
    "layer_couplings",
    "timed_build",
]
