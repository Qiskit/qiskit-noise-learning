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

"""Pre-packaged noise learning procedures.

One subpackage per protocol, each holding one pair of functions: one to build the program to
submit, one to analyze the result it comes back as. Every protocol's pair is re-exported here, so
a protocol is chosen by which pair you call rather than by an argument, and the two functions of a
pair communicate only through the program's passthrough data.
"""

from .vanilla import VanillaResults, prepare_vanilla_program, process_vanilla_result
