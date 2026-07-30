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

"""Derivation of independent seeds from a root seed.

Randomness in this subpackage is drawn from several places — Aer's shot sampling, and the
twirl and noise sampling done here — and a run may need many streams of it.  Each stream
gets its own seed drawn from a shared ``numpy.random.SeedSequence``, so that one root seed
reproduces a whole run without any two streams sharing randomness.
"""

import numpy as np


def next_seed(seed_sequence: np.random.SeedSequence) -> int:
    """Draw a fresh seed from ``seed_sequence``, advancing it.

    Each call returns a seed for a stream independent of every other stream drawn from the
    same sequence.  Aer takes a plain integer rather than a seed sequence, and rejects
    values above ``2 ** 63 - 1``, so the drawn child is reduced to 63 bits.  That is wide
    enough that two draws colliding — which would silently correlate the two streams — is
    not a practical concern, as a 32-bit reduction would be for a program of many items.

    Args:
        seed_sequence: The sequence to draw from.  It is mutated, so successive calls
            return different seeds.

    Returns:
        A seed for one independent stream of randomness.
    """
    return int(seed_sequence.spawn(1)[0].generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))
