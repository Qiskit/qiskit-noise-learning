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

A *protocol* is one complete noise learning procedure, from a description of the gates to
characterize through to a fitted noise model. The word is used in its experimental sense --- a
procedure for characterizing a device --- rather than in the sense of :class:`typing.Protocol`.

Each protocol is exposed as a pair of functions, one per side of execution:

* a ``prepare_`` function turns a description of what to learn into a
  :class:`~qiskit_ibm_runtime.quantum_program.QuantumProgram` ready to submit, and
* a ``process_`` function turns that program's results into a :class:`~.Fit`.

The two communicate only through the program's passthrough data, which the prepare function writes
and the process function reads back. Nothing else is shared between them, so a program may be
submitted by one process and its results analyzed by another, with no state carried across.

Choosing a level
----------------

These functions are deliberately not as configurable as the library beneath them. Every argument is
either a quantity the protocol cannot guess --- how many randomizations, how many shots --- or a
library object extending one layer of the procedure. None of them is a string naming a preset.

The extension points sit at the two ends of the procedure: a pass manager applied to the generated
circuits, and an analysis stage applied to the raw data before the standard pipeline. Those are the
points where a concern of the caller's own, such as a post-selection scheme, legitimately enters.
Everything between them --- the noise model, the paths, the solver --- is what makes a protocol the
procedure it is, and is not adjustable.

A caller who needs to change the middle should build the experiment from the library directly rather
than reach for a protocol. The workflow tutorial walks through doing so, and every stage a protocol
assembles internally is public.
"""

from .learning import prepare_learning_program, process_learning_results
