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

from qiskit_noise_learning.analysis.average_observables import average_observables


class TestAverageObservables:
    """Tests for the ``average_observables`` function."""

    def test_output_order_follows_first_seen_paths(self, make_cz_path, make_observable_data):
        """Observables are ordered by first appearance of their path, then by fragment depth.

        The fragment depths are supplied in descending order, so an ascending output order
        can only have come from this function.
        """
        paths = [make_cz_path(label) for label in ("XI", "IX", "XX", "YI", "IY")]
        obs = make_observable_data([(path, 0.9, 0.8, [2, 1]) for path in paths])

        dataset = average_observables(obs).dataset

        assert list(dataset["unbound_path"].data) == [path for path in paths for _ in range(2)]
        assert list(dataset["fragment_depth"].data) == [1, 2] * len(paths)

    def test_requested_path_order_is_respected(self, make_cz_path, make_observable_data):
        """The output follows the order of the requested paths, not that of the data."""
        paths = [make_cz_path(label) for label in ("XI", "IX", "XX")]
        obs = make_observable_data([(path, 0.9, 0.8, [1]) for path in paths])

        requested = [paths[2], paths[0]]
        dataset = average_observables(obs, unique_unbound_paths=requested).dataset

        assert list(dataset["unbound_path"].data) == requested
