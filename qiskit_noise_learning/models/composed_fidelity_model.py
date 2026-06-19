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

"""ComposedFidelityModel"""

from qiskit_noise_learning.math import ComposedLinearMap, LinearMap

from .fidelity_model import FidelityModel


class ComposedFidelityModel(ComposedLinearMap, FidelityModel):
    """A fidelity model defined by a flat chain of composed linear maps.

    Inherits composition mechanics (flattening, ``row`` via ``left_multiply`` folding) from
    :class:`~.ComposedLinearMap` and domain convenience methods (``row_from_path``,
    ``fidelity_estimate``, etc.) from :class:`~.FidelityModel`.

    Because :class:`~.ComposedLinearMap` precedes :class:`~.FidelityModel` in the MRO,
    ``row``/``compose``/``pre_compose`` resolve to the composition-aware versions, while
    :class:`~.FidelityModel` (which defines no ``__init__``) lets ``ComposedLinearMap``'s
    cooperative ``super().__init__`` reach :class:`~.LinearMap` directly.

    Args:
        maps: Ordered sequence of maps in application order (``maps[0]`` applied first).
    """

    def __init__(self, maps: list[LinearMap]):
        super().__init__(maps)
        self._gate_set = self._maps[-1].output_space.gate_set
