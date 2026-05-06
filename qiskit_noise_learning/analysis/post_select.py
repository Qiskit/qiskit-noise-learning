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

import numpy as np

from qiskit_noise_learning.analysis import AnalysisStage
from qiskit_noise_learning.data import RawData
from qiskit_noise_learning.data.xarray_utils import time_bound
from qiskit_noise_learning.sequences import PathPattern


class PostSelect(AnalysisStage):
    """Apply a mask to raw data based on post selection metadata."""

    @property
    def input_level(self):
        return RawData

    @property
    def output_level(self):
        return RawData

    def _run(self, fit):
        fit[RawData] = post_select(fit.raw_data)


def post_select(raw_data: RawData) -> RawData:
    """Post select the raw data...

    Notes:
        - Currently hacking it based on metadata

    Args:
        raw_data: The un-post-selected raw data.
    """

    pass