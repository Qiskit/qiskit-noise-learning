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

"""Regression tests pinning the grouping quality of the heuristic algorithm.

Checks number of groups for several known examples.
"""

from typing import get_args

import pytest

from qiskit_noise_learning.sequences import (
    ApplyGate,
    GroupingStrategy,
    InstructionSequence,
    PartialPauliPermutation,
    group_mergeable_instruction_sequences,
)

_ALL_GROUPING_STRATEGIES = list(
    (order, merging_strategy)
    for order in get_args(get_args(GroupingStrategy)[0])
    for merging_strategy in get_args(get_args(GroupingStrategy)[1])
)
"""Every pairing of a documented instruction sequence order with a documented merging strategy."""

_PERMUTATION_SETS = (
    {("X", "X"), ("Y", "Y"), ("Z", "Z")},
    {("X", "Z"), ("Y", "Y"), ("Z", "X")},
    {("X", "Y"), ("Y", "Z"), ("Z", "X")},
    {("X", "X"), ("Y", "Z"), ("Z", "Y")},
    {("X", "Z"), ("Y", "X"), ("Z", "Y")},
    {("X", "Y"), ("Y", "X"), ("Z", "Z")},
    {("Z", "Z")},
    {("Z", "X")},
    {("Z", "Y")},
    {("X", "Z")},
    {("X", "X")},
    {("X", "Y")},
    {("Y", "Z")},
    {("Y", "X")},
    {("Y", "Y")},
    set(),
)
"""The partial permutations the indices in the recorded instances refer to."""

_INSTANCES = {
    "dense_coupling": (
        14,
        """
        6fffffffffff6fffffffffff6fffffffffff6fffffffffff
        7fffffffffffa6ffffffffffafffffffffff9fffffffffff
        7fffffffffffb6ffffffffffdfffffffffff9fffffffffff
        8fffffffffffd6ffffffffffbfffffffffffcfffffffffff
        f6fffffffffff6fffffffffff6fffffffffff6ffffffffff
        f7ffffffffff6afffffffffffafffffffffff9ffffffffff
        f7ffffffffff6bfffffffffffdfffffffffff9ffffffffff
        f8ffffffffff6dfffffffffffbfffffffffffcffffffffff
        ff6fffffffffff6fffffffffff6fffffffffff6fffffffff
        ff7fffffffffffa6ffffffffffafffffffffff9fffffffff
        ff7fffffffffffb6ffffffffffdfffffffffff9fffffffff
        ff8fffffffffffd6ffffffffffbfffffffffffcfffffffff
        fff6fffffffffff6fffffffffff6fffffffffff6ffffffff
        fff7ffffffffff6afffffffffffafffffffffff9ffffffff
        fff7ffffffffff6bfffffffffffdfffffffffff9ffffffff
        fff8ffffffffff6dfffffffffffbfffffffffffcffffffff
        ffff6fffffffffff6fffffffffff6fffffffffff6fffffff
        ffff7fffffffffffa6ffffffffffafffffffffff9fffffff
        ffff7fffffffffffb6ffffffffffdfffffffffff9fffffff
        ffff8fffffffffffd6ffffffffffbfffffffffffcfffffff
        fffff6fffffffffff6fffffffffff6fffffffffff6ffffff
        fffff7ffffffffff6afffffffffffafffffffffff9ffffff
        fffff7ffffffffff6bfffffffffffdfffffffffff9ffffff
        fffff8ffffffffff6dfffffffffffbfffffffffffcffffff
        ffffff6fffffffffff6fffffffffff6fffffffffff6fffff
        ffffff7fffffffffffa6ffffffffffafffffffffff9fffff
        ffffff7fffffffffffb6ffffffffffdfffffffffff9fffff
        ffffff8fffffffffffd6ffffffffffbfffffffffffcfffff
        fffffff6fffffffffff6fffffffffff6fffffffffff6ffff
        fffffff7ffffffffff6afffffffffffafffffffffff9ffff
        fffffff7ffffffffff6bfffffffffffdfffffffffff9ffff
        fffffff8ffffffffff6dfffffffffffbfffffffffffcffff
        ffffffff6fffffffffff6fffffffffff6fffffffffff6fff
        ffffffff7fffffffffffa6ffffffffffafffffffffff9fff
        ffffffff7fffffffffffb6ffffffffffdfffffffffff9fff
        ffffffff8fffffffffffd6ffffffffffbfffffffffffcfff
        fffffffff6fffffffffff6fffffffffff6fffffffffff6ff
        fffffffff7ffffffffff6afffffffffffafffffffffff9ff
        fffffffff7ffffffffff6bfffffffffffdfffffffffff9ff
        fffffffff8ffffffffff6dfffffffffffbfffffffffffcff
        ffffffffff6fffffffffff6fffffffffff6fffffffffff6f
        ffffffffff6fffffffffff7fffffffffff9fffffffffff6f
        ffffffffff6fffffffffff8fffffffffffcfffffffffff6f
        fffffffffff6fffffffffff6fffffffffff6fffffffffff6
        fffffffffff6fffffffffff7fffffffffff9fffffffffff6
        fffffffffff6fffffffffff8fffffffffffcfffffffffff6
        66ffffffffff66ffffffffff66ffffffffff66ffffffffff
        66ffffffffff88ffffffffff99ffffffffff66ffffffffff
        66ffffffffff78ffffffffffc9ffffffffff66ffffffffff
        66ffffffffff87ffffffffff9cffffffffff66ffffffffff
        66ffffffffff77ffffffffffccffffffffff66ffffffffff
        6fff6fffffff6fff6fffffff6fff6fffffff6fff6fffffff
        7fff6fffffffa6ff6fffffffafff6fffffff9fff6fffffff
        7fff6fffffffb6ff6fffffffdfff6fffffff9fff6fffffff
        6fff7fffffff6fffa6ffffff6fffafffffff6fff9fffffff
        6fff7fffffff6fffb6ffffff6fffdfffffff6fff9fffffff
        7fff7fffffffa6ffa6ffffffafffafffffff9fff9fffffff
        7fff7fffffffb6ffa6ffffffdfffafffffff9fff9fffffff
        7fff7fffffffa6ffb6ffffffafffdfffffff9fff9fffffff
        7fff7fffffffb6ffb6ffffffdfffdfffffff9fff9fffffff
        f66ffffffffff66ffffffffff66ffffffffff66fffffffff
        f76fffffffff6a6ffffffffffa6ffffffffff96fffffffff
        f76fffffffff6b6ffffffffffd6ffffffffff96fffffffff
        f67ffffffffff6a6fffffffff6affffffffff69fffffffff
        f67ffffffffff6b6fffffffff6dffffffffff69fffffffff
        f77fffffffff6aa6fffffffffaaffffffffff99fffffffff
        f77fffffffff6ba6fffffffffdaffffffffff99fffffffff
        f77fffffffff6ab6fffffffffadffffffffff99fffffffff
        f77fffffffff6bb6fffffffffddffffffffff99fffffffff
        f6fff6fffffff6fff6fffffff6fff6fffffff6fff6ffffff
        f7fff6ffffff6afff6fffffffafff6fffffff9fff6ffffff
        f7fff6ffffff6bfff6fffffffdfff6fffffff9fff6ffffff
        f6fff7fffffff6ff6afffffff6fffafffffff6fff9ffffff
        f6fff7fffffff6ff6bfffffff6fffdfffffff6fff9ffffff
        f7fff7ffffff6aff6afffffffafffafffffff9fff9ffffff
        f7fff7ffffff6bff6afffffffdfffafffffff9fff9ffffff
        f7fff7ffffff6aff6bfffffffafffdfffffff9fff9ffffff
        f7fff7ffffff6bff6bfffffffdfffdfffffff9fff9ffffff
        ff66ffffffffff66ffffffffff66ffffffffff66ffffffff
        ff66ffffffffff88ffffffffff99ffffffffff66ffffffff
        ff66ffffffffff78ffffffffffc9ffffffffff66ffffffff
        ff66ffffffffff87ffffffffff9cffffffffff66ffffffff
        ff66ffffffffff77ffffffffffccffffffffff66ffffffff
        ff6fff6fffffff6fff6fffffff6fff6fffffff6fff6fffff
        ff7fff6fffffffa6ff6fffffffafff6fffffff9fff6fffff
        ff7fff6fffffffb6ff6fffffffdfff6fffffff9fff6fffff
        ff6fff7fffffff6fffa6ffffff6fffafffffff6fff9fffff
        ff6fff7fffffff6fffb6ffffff6fffdfffffff6fff9fffff
        ff7fff7fffffffa6ffa6ffffffafffafffffff9fff9fffff
        ff7fff7fffffffb6ffa6ffffffdfffafffffff9fff9fffff
        ff7fff7fffffffa6ffb6ffffffafffdfffffff9fff9fffff
        ff7fff7fffffffb6ffb6ffffffdfffdfffffff9fff9fffff
        fff6fff6fffffff6fff6fffffff6fff6fffffff6fff6ffff
        fff7fff6ffffff6afff6fffffffafff6fffffff9fff6ffff
        fff7fff6ffffff6bfff6fffffffdfff6fffffff9fff6ffff
        fff6fff7fffffff6ff6afffffff6fffafffffff6fff9ffff
        fff6fff7fffffff6ff6bfffffff6fffdfffffff6fff9ffff
        fff7fff7ffffff6aff6afffffffafffafffffff9fff9ffff
        fff7fff7ffffff6bff6afffffffdfffafffffff9fff9ffff
        fff7fff7ffffff6aff6bfffffffafffdfffffff9fff9ffff
        fff7fff7ffffff6bff6bfffffffdfffdfffffff9fff9ffff
        ffff66ffffffffff66ffffffffff66ffffffffff66ffffff
        ffff66ffffffffff88ffffffffff99ffffffffff66ffffff
        ffff66ffffffffff78ffffffffffc9ffffffffff66ffffff
        ffff66ffffffffff87ffffffffff9cffffffffff66ffffff
        ffff66ffffffffff77ffffffffffccffffffffff66ffffff
        ffff6fff6fffffff6fff6fffffff6fff6fffffff6fff6fff
        ffff7fff6fffffffa6ff6fffffffafff6fffffff9fff6fff
        ffff7fff6fffffffb6ff6fffffffdfff6fffffff9fff6fff
        ffff6fff7fffffff6fffa6ffffff6fffafffffff6fff9fff
        ffff6fff7fffffff6fffb6ffffff6fffdfffffff6fff9fff
        ffff7fff7fffffffa6ffa6ffffffafffafffffff9fff9fff
        ffff7fff7fffffffb6ffa6ffffffdfffafffffff9fff9fff
        ffff7fff7fffffffa6ffb6ffffffafffdfffffff9fff9fff
        ffff7fff7fffffffb6ffb6ffffffdfffdfffffff9fff9fff
        fffff66ffffffffff66ffffffffff66ffffffffff66fffff
        fffff76fffffffff6a6ffffffffffa6ffffffffff96fffff
        fffff76fffffffff6b6ffffffffffd6ffffffffff96fffff
        fffff67ffffffffff6a6fffffffff6affffffffff69fffff
        fffff67ffffffffff6b6fffffffff6dffffffffff69fffff
        fffff77fffffffff6aa6fffffffffaaffffffffff99fffff
        fffff77fffffffff6ba6fffffffffdaffffffffff99fffff
        fffff77fffffffff6ab6fffffffffadffffffffff99fffff
        fffff77fffffffff6bb6fffffffffddffffffffff99fffff
        fffff6fff6fffffff6fff6fffffff6fff6fffffff6fff6ff
        fffff7fff6ffffff6afff6fffffffafff6fffffff9fff6ff
        fffff7fff6ffffff6bfff6fffffffdfff6fffffff9fff6ff
        fffff6fff7fffffff6ff6afffffff6fffafffffff6fff9ff
        fffff6fff7fffffff6ff6bfffffff6fffdfffffff6fff9ff
        fffff7fff7ffffff6aff6afffffffafffafffffff9fff9ff
        fffff7fff7ffffff6bff6afffffffdfffafffffff9fff9ff
        fffff7fff7ffffff6aff6bfffffffafffdfffffff9fff9ff
        fffff7fff7ffffff6bff6bfffffffdfffdfffffff9fff9ff
        ffffff66ffffffffff66ffffffffff66ffffffffff66ffff
        ffffff66ffffffffff88ffffffffff99ffffffffff66ffff
        ffffff66ffffffffff78ffffffffffc9ffffffffff66ffff
        ffffff66ffffffffff87ffffffffff9cffffffffff66ffff
        ffffff66ffffffffff77ffffffffffccffffffffff66ffff
        ffffff6fff6fffffff6fff6fffffff6fff6fffffff6fff6f
        ffffff6fff6fffffff6fff7fffffff6fff9fffffff6fff6f
        ffffff6fff6fffffff6fff8fffffff6fffcfffffff6fff6f
        ffffff7fff6fffffffa6ff6fffffffafff6fffffff9fff6f
        ffffff7fff6fffffffb6ff6fffffffdfff6fffffff9fff6f
        ffffff7fff6fffffffa6ff7fffffffafff9fffffff9fff6f
        ffffff7fff6fffffffb6ff7fffffffdfff9fffffff9fff6f
        ffffff7fff6fffffffa6ff8fffffffafffcfffffff9fff6f
        ffffff7fff6fffffffb6ff8fffffffdfffcfffffff9fff6f
        ffffffff66ffffffffff66ffffffffff66ffffffffff66ff
        ffffffff66ffffffffff88ffffffffff99ffffffffff66ff
        ffffffff66ffffffffff78ffffffffffc9ffffffffff66ff
        ffffffff66ffffffffff87ffffffffff9cffffffffff66ff
        ffffffff66ffffffffff77ffffffffffccffffffffff66ff
        fffffffff66ffffffffff66ffffffffff66ffffffffff66f
        fffffffff66ffffffffff67ffffffffff69ffffffffff66f
        fffffffff66ffffffffff68ffffffffff6cffffffffff66f
        fffffffff76fffffffff6a6ffffffffffa6ffffffffff96f
        fffffffff76fffffffff6b6ffffffffffd6ffffffffff96f
        fffffffff76fffffffff6a7ffffffffffa9ffffffffff96f
        fffffffff76fffffffff6b7ffffffffffd9ffffffffff96f
        fffffffff76fffffffff6a8ffffffffffacffffffffff96f
        fffffffff76fffffffff6b8ffffffffffdcffffffffff96f
        fffffffff6f6fffffffff6f6fffffffff6f6fffffffff6f6
        fffffffff6f6fffffffff6f7fffffffff6f9fffffffff6f6
        fffffffff6f6fffffffff6f8fffffffff6fcfffffffff6f6
        fffffffff7f6ffffffff6af6fffffffffaf6fffffffff9f6
        fffffffff7f6ffffffff6bf6fffffffffdf6fffffffff9f6
        fffffffff7f6ffffffff6af7fffffffffaf9fffffffff9f6
        fffffffff7f6ffffffff6bf7fffffffffdf9fffffffff9f6
        fffffffff7f6ffffffff6af8fffffffffafcfffffffff9f6
        fffffffff7f6ffffffff6bf8fffffffffdfcfffffffff9f6
        """,
    ),
    "dense_coupling2": (
        2,
        """
        6fffffffffff6fffffffffff
        7fffffffffff96ffffffffff
        f6fffffffffff6ffffffffff
        f7ffffffffff69ffffffffff
        ff6fffffffffff6fffffffff
        ff7fffffffffff96ffffffff
        fff6fffffffffff6ffffffff
        fff7ffffffffff69ffffffff
        ffff6fffffffffff6fffffff
        ffff7fffffffffff96ffffff
        fffff6fffffffffff6ffffff
        fffff7ffffffffff69ffffff
        ffffff6fffffffffff6fffff
        ffffff7fffffffffff96ffff
        fffffff6fffffffffff6ffff
        fffffff7ffffffffff69ffff
        ffffffff6fffffffffff6fff
        ffffffff7fffffffffff96ff
        fffffffff6fffffffffff6ff
        fffffffff7ffffffffff69ff
        ffffffffff6fffffffffff6f
        fffffffffff6fffffffffff6
        """,
    ),
    "even_depth_spam": (
        14,
        """
        6fff6fff6fff6fff
        7fffa6ffafff9fff
        7fffb6ffdfff9fff
        8fffd6ffbfffcfff
        f6fff6fff6fff6ff
        f7ff6afffafff9ff
        f7ff6bfffdfff9ff
        f8ff6dfffbfffcff
        ff6fff6fff6fff6f
        ff7fffa6ffafff9f
        ff7fffb6ffdfff9f
        ff8fffd6ffbfffcf
        fff6fff6fff6fff6
        fff7ff6afffafff9
        fff7ff6bfffdfff9
        fff8ff6dfffbfffc
        66ff66ff66ff66ff
        66ff88ff99ff66ff
        66ff78ffc9ff66ff
        66ff87ff9cff66ff
        66ff77ffccff66ff
        f66ff66ff66ff66f
        f76f6a6ffa6ff96f
        f76f6b6ffd6ff96f
        f67ff6a6f6aff69f
        f67ff6b6f6dff69f
        f77f6aa6faaff99f
        f77f6ba6fdaff99f
        f77f6ab6fadff99f
        f77f6bb6fddff99f
        ff66ff66ff66ff66
        ff66ff88ff99ff66
        ff66ff78ffc9ff66
        ff66ff87ff9cff66
        ff66ff77ffccff66
        """,
    ),
    "even_depth_spam2": (
        1,
        """
        6fff
        f6ff
        ff6f
        fff6
        """,
    ),
    "full": (
        2,
        """
        6fff6fff
        7fff96ff
        f6fff6ff
        f7ff69ff
        ff6fff6f
        ff7fff96
        fff6fff6
        fff7ff69
        """,
    ),
    "vanilla": (
        9,
        """
        6fff6fff6fff6fff
        7fffa6ffafff9fff
        8fffe6ffefffcfff
        f6fff6fff6fff6ff
        f7ff6afffafff9ff
        f8ff6efffefffcff
        ff6fff6fff6fff6f
        ff7fffa6ffafff9f
        ff8fffe6ffefffcf
        fff6fff6fff6fff6
        fff7ff6afffafff9
        fff8ff6efffefffc
        66ff66ff66ff66ff
        77ffeeffaaff99ff
        87ffaeffeaffc9ff
        f66ff66ff66ff66f
        f76f6a6ffa6ff96f
        f86f6e6ffe6ffc6f
        f67ff6a6f6aff69f
        f77f6aa6faaff99f
        f87f6ea6feaffc9f
        f68ff6e6f6eff6cf
        f78f6ae6faeff9cf
        f88f6ee6feeffccf
        ff66ff66ff66ff66
        ff77ffeeffaaff99
        ff87ffaeffeaffc9
        """,
    ),
    "vanilla2": (
        3,
        """
        6fffffff6fffffff6fffffff6fffffff
        7fffffffa6ffffffafffffff9fffffff
        8fffffffe6ffffffefffffffcfffffff
        f6fffffff6fffffff6fffffff6ffffff
        f7ffffff6afffffffafffffff9ffffff
        f8ffffff6efffffffefffffffcffffff
        ff6fffffff6fffffff6fffffff6fffff
        ff7fffffffa6ffffffafffffff9fffff
        ff8fffffffe6ffffffefffffffcfffff
        fff6fffffff6fffffff6fffffff6ffff
        fff7ffffff6afffffffafffffff9ffff
        fff8ffffff6efffffffefffffffcffff
        ffff6fffffff6fffffff6fffffff6fff
        ffff7fffffffaf6fffffafffffff9fff
        ffff8fffffffef6fffffefffffffcfff
        ffffff6fffffff6fffffff6fffffff6f
        ffffff7fffff6fafffffffafffffff9f
        ffffff8fffff6fefffffffefffffffcf
        fffff6fffffff6fffffff6fffffff6ff
        fffff7fffffffaf6fffffafffffff9ff
        fffff8fffffffef6fffffefffffffcff
        fffffff6fffffff6fffffff6fffffff6
        fffffff7fffff6fafffffffafffffff9
        fffffff8fffff6fefffffffefffffffc
        """,
    ),
    "vanilla3": (
        9,
        """
        6fffffffffff6fffffffffff6fffffffffff6fffffffffff
        7fffffffffffa6ffffffffffafffffffffff9fffffffffff
        8fffffffffffe6ffffffffffefffffffffffcfffffffffff
        f6fffffffffff6fffffffffff6fffffffffff6ffffffffff
        f7ffffffffff6afffffffffffafffffffffff9ffffffffff
        f8ffffffffff6efffffffffffefffffffffffcffffffffff
        ff6fffffffffff6fffffffffff6fffffffffff6fffffffff
        ff7fffffffffffa6ffffffffffafffffffffff9fffffffff
        ff8fffffffffffe6ffffffffffefffffffffffcfffffffff
        fff6fffffffffff6fffffffffff6fffffffffff6ffffffff
        fff7ffffffffff6afffffffffffafffffffffff9ffffffff
        fff8ffffffffff6efffffffffffefffffffffffcffffffff
        ffff6fffffffffff6fffffffffff6fffffffffff6fffffff
        ffff7fffffffffffaf6fffffffffafffffffffff9fffffff
        ffff8fffffffffffef6fffffffffefffffffffffcfffffff
        ffffff6fffffffffff6fffffffffff6fffffffffff6fffff
        ffffff7fffffffff6fafffffffffffafffffffffff9fffff
        ffffff8fffffffff6fefffffffffffefffffffffffcfffff
        fffff6fffffffffff6fffffffffff6fffffffffff6ffffff
        fffff7fffffffffffaf6fffffffffafffffffffff9ffffff
        fffff8fffffffffffef6fffffffffefffffffffffcffffff
        fffffff6fffffffffff6fffffffffff6fffffffffff6ffff
        fffffff7fffffffff6fafffffffffffafffffffffff9ffff
        fffffff8fffffffff6fefffffffffffefffffffffffcffff
        ffffffff6fffffffffff6fffffffffff6fffffffffff6fff
        ffffffff7fffffffffffa6ffffffffffafffffffffff9fff
        ffffffff8fffffffffffe6ffffffffffefffffffffffcfff
        fffffffff6fffffffffff6fffffffffff6fffffffffff6ff
        fffffffff7ffffffffff6afffffffffffafffffffffff9ff
        fffffffff8ffffffffff6efffffffffffefffffffffffcff
        ffffffffff6fffffffffff6fffffffffff6fffffffffff6f
        ffffffffff7fffffffffffa6ffffffffffafffffffffff9f
        ffffffffff8fffffffffffe6ffffffffffefffffffffffcf
        fffffffffff6fffffffffff6fffffffffff6fffffffffff6
        fffffffffff7ffffffffff6afffffffffffafffffffffff9
        fffffffffff8ffffffffff6efffffffffffefffffffffffc
        66ffffffffff66ffffffffff66ffffffffff66ffffffffff
        77ffffffffffeeffffffffffaaffffffffff99ffffffffff
        87ffffffffffaeffffffffffeaffffffffffc9ffffffffff
        6ffff6ffffff6ffff6ffffff6ffff6ffffff6ffff6ffffff
        7ffff6ffffffa6fff6ffffffaffff6ffffff9ffff6ffffff
        8ffff6ffffffe6fff6ffffffeffff6ffffffcffff6ffffff
        6ffff7ffffff6ffffaf6ffff6ffffaffffff6ffff9ffffff
        7ffff7ffffffa6fffaf6ffffaffffaffffff9ffff9ffffff
        8ffff7ffffffe6fffaf6ffffeffffaffffffcffff9ffffff
        6ffff8ffffff6ffffef6ffff6ffffeffffff6ffffcffffff
        7ffff8ffffffa6fffef6ffffaffffeffffff9ffffcffffff
        8ffff8ffffffe6fffef6ffffeffffeffffffcffffcffffff
        f66ffffffffff66ffffffffff66ffffffffff66fffffffff
        f76fffffffff6a6ffffffffffa6ffffffffff96fffffffff
        f86fffffffff6e6ffffffffffe6ffffffffffc6fffffffff
        f67ffffffffff6a6fffffffff6affffffffff69fffffffff
        f77fffffffff6aa6fffffffffaaffffffffff99fffffffff
        f87fffffffff6ea6fffffffffeaffffffffffc9fffffffff
        f68ffffffffff6e6fffffffff6effffffffff6cfffffffff
        f78fffffffff6ae6fffffffffaeffffffffff9cfffffffff
        f88fffffffff6ee6fffffffffeeffffffffffccfffffffff
        ff66ffffffffff66ffffffffff66ffffffffff66ffffffff
        ff77ffffffffffeeffffffffffaaffffffffff99ffffffff
        ff87ffffffffffaeffffffffffeaffffffffffc9ffffffff
        fff66ffffffffff66ffffffffff66ffffffffff66fffffff
        fff76fffffffff6a6ffffffffffa6ffffffffff96fffffff
        fff86fffffffff6e6ffffffffffe6ffffffffffc6fffffff
        fff67ffffffffff6af6ffffffff6affffffffff69fffffff
        fff77fffffffff6aaf6ffffffffaaffffffffff99fffffff
        fff87fffffffff6eaf6ffffffffeaffffffffffc9fffffff
        fff68ffffffffff6ef6ffffffff6effffffffff6cfffffff
        fff78fffffffff6aef6ffffffffaeffffffffff9cfffffff
        fff88fffffffff6eef6ffffffffeeffffffffffccfffffff
        ffff6f6fffffffff6f6fffffffff6f6fffffffff6f6fffff
        ffff7f7fffffffffefefffffffffafafffffffff9f9fffff
        ffff8f7fffffffffafefffffffffefafffffffffcf9fffff
        ffffff6ffff6ffffff6ffff6ffffff6ffff6ffffff6ffff6
        ffffff7ffff6ffff6faffff6ffffffaffff6ffffff9ffff6
        ffffff8ffff6ffff6feffff6ffffffeffff6ffffffcffff6
        ffffff6ffff7ffffff6fff6affffff6ffffaffffff6ffff9
        ffffff7ffff7ffff6fafff6affffffaffffaffffff9ffff9
        ffffff8ffff7ffff6fefff6affffffeffffaffffffcffff9
        ffffff6ffff8ffffff6fff6effffff6ffffeffffff6ffffc
        ffffff7ffff8ffff6fafff6effffffaffffeffffff9ffffc
        ffffff8ffff8ffff6fefff6effffffeffffeffffffcffffc
        fffff6f6fffffffff6f6fffffffff6f6fffffffff6f6ffff
        fffff7f7fffffffffefefffffffffafafffffffff9f9ffff
        fffff8f7fffffffffafefffffffffefafffffffffcf9ffff
        fffffff66ffffffffff66ffffffffff66ffffffffff66fff
        fffffff76ffffffff6fa6ffffffffffa6ffffffffff96fff
        fffffff86ffffffff6fe6ffffffffffe6ffffffffffc6fff
        fffffff67ffffffffff6a6fffffffff6affffffffff69fff
        fffffff77ffffffff6faa6fffffffffaaffffffffff99fff
        fffffff87ffffffff6fea6fffffffffeaffffffffffc9fff
        fffffff68ffffffffff6e6fffffffff6effffffffff6cfff
        fffffff78ffffffff6fae6fffffffffaeffffffffff9cfff
        fffffff88ffffffff6fee6fffffffffeeffffffffffccfff
        ffffffff66ffffffffff66ffffffffff66ffffffffff66ff
        ffffffff77ffffffffffeeffffffffffaaffffffffff99ff
        ffffffff87ffffffffffaeffffffffffeaffffffffffc9ff
        fffffffff66ffffffffff66ffffffffff66ffffffffff66f
        fffffffff76fffffffff6a6ffffffffffa6ffffffffff96f
        fffffffff86fffffffff6e6ffffffffffe6ffffffffffc6f
        fffffffff67ffffffffff6a6fffffffff6affffffffff69f
        fffffffff77fffffffff6aa6fffffffffaaffffffffff99f
        fffffffff87fffffffff6ea6fffffffffeaffffffffffc9f
        fffffffff68ffffffffff6e6fffffffff6effffffffff6cf
        fffffffff78fffffffff6ae6fffffffffaeffffffffff9cf
        fffffffff88fffffffff6ee6fffffffffeeffffffffffccf
        ffffffffff66ffffffffff66ffffffffff66ffffffffff66
        ffffffffff77ffffffffffeeffffffffffaaffffffffff99
        ffffffffff87ffffffffffaeffffffffffeaffffffffffc9
        """,
    ),
}
"""Measured grouping instances, as ``name: (fewest known groups, one line per sequence)``."""


def _instance(name):
    """Return the fewest groups known for an instance, and its instruction sequences.

    Args:
        name: The name of the instance to rebuild.

    Returns:
        The fewest number of groups known to be achievable, and the instruction sequences to group,
        each holding the recorded partial permutation on every qubit followed by a gate.
    """
    fewest_known, grid = _INSTANCES[name]
    sequences = [
        InstructionSequence(
            start_fragment=[
                PartialPauliPermutation.from_sets(
                    [_PERMUTATION_SETS[int(index, 16)] for index in line]
                )
            ],
            repeatable_fragment=[ApplyGate("L")],
            end_fragment=[],
            fragment_depth=1,
        )
        for line in grid.split()
    ]

    return fewest_known, sequences


@pytest.mark.parametrize("name", sorted(_INSTANCES))
def test_default_strategies_attain_the_fewest_known_groups(name):
    """Test that the default strategies group each measured instance as tightly as recorded.

    This test requires equality - if it fails due to achieving a better number, update _INSTANCES.
    """
    fewest_known, sequences = _instance(name)

    groups = group_mergeable_instruction_sequences(sequences)

    assert sorted(idx for group in groups for idx in group) == list(range(len(sequences)))
    assert len(groups) == fewest_known, (
        f"grouping {name} with the default strategies gives {len(groups)} groups, worse than the "
        f"{fewest_known} recorded for it."
    )


def test_no_single_strategy_attains_the_fewest_known_groups_everywhere():
    """Test that combining several grouping strategies is what attains the recorded counts.

    Every default strategy is beaten by another on some instance, which is why
    ``DEFAULT_GROUPING_STRATEGIES`` holds more than one of them.
    """
    instances = [_instance(name) for name in sorted(_INSTANCES)]

    for grouping_strategy in _ALL_GROUPING_STRATEGIES:
        attained = [
            len(group_mergeable_instruction_sequences(sequences, [grouping_strategy])) <= fewest
            for fewest, sequences in instances
        ]
        assert not all(attained), (
            f"{grouping_strategy} alone attains the fewest known groups on every instance, so the "
            f"default strategies no longer need to combine several of them."
        )
