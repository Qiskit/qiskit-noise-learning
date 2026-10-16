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

"""The vanilla learning protocol.

:func:`prepare_vanilla_program` builds the program to submit and :func:`process_vanilla_result`
analyzes the result it comes back as. Everything the analysis needs in order to interpret a result
travels in the program's passthrough data, because the two functions share nothing else: the
program is submitted, executed and returned by machinery that knows nothing about either of them.
The serialized experiment is written there by the circuit generator; what this protocol adds is the
record of which gate each submitted instruction became, which nothing in the experiment itself
preserves. That record is a plain, versioned entry of the reserved protocol data, written and read
by this subpackage alone.
"""

from .prepare_vanilla_program import prepare_vanilla_program
from .process_vanilla_result import process_vanilla_result
