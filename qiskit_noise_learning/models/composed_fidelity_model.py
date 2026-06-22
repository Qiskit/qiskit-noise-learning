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

from .fidelity_model import FidelityModel, _composed_fidelity_map


class ComposedFidelityModel(ComposedLinearMap, FidelityModel):
    """A fidelity model defined by a flat chain of composed linear maps.

    Args:
        maps: Ordered sequence of maps in application order (``maps[0]`` applied first).
    """

    def compose(self, outer: "LinearMap") -> "LinearMap":
        """Post-compose with a linear map on the output space, flattening the chain.

        Args:
            outer: A linear map applied after this one.

        Returns:
            A :class:`~.ComposedFidelityModel` if the composed chain still maps into a
            :class:`~.LogFidelitySpace`, otherwise a plain :class:`~.ComposedLinearMap`.
        """
        outer_maps = outer.maps if isinstance(outer, ComposedLinearMap) else [outer]
        return _composed_fidelity_map([*self._maps, *outer_maps])
