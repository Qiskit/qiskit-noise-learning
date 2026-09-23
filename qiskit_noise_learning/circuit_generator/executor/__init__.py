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

"""Circuit generation and result mapping for the runtime executor.

Generating a program loses the caller's ordering: instruction sequences with the same structure are
built from one template circuit with differing samplex arguments, so results come back grouped by
template rather than in the order the sequences were given. The data mapper records what is needed
to undo that, and carries the experiment it describes alongside it.

- :mod:`~.executor_circuit_generator`: builds a quantum program from an experiment.
- :mod:`~.executor_data_mapper`: the layout and experiment record that interprets its results.
"""

from .executor_circuit_generator import ExecutorCircuitGenerator
from .executor_data_mapper import ExecutorDataMapper

__all__ = ["ExecutorCircuitGenerator", "ExecutorDataMapper"]
