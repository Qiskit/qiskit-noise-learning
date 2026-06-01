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

"""Experiment builder module."""

from .experiment import Experiment
from .experiment_builder import ExperimentBuilder
from .experiment_generators import (
    depth0_path_generator,
    depth1_path_generator,
    even_depth_path_generator,
    even_depth_vanilla_path_generator,
    standard_vanilla_path_generator,
)
from .stage import ExperimentBuilder as ExperimentBuilderPipeline
from .stage import ExperimentBuilderStage
from .stages import (
    AddInstructionSequences,
    AddPaths,
    BindDepths,
    Complete,
    Depth0Paths,
    Depth1Paths,
    EvenDepthPaths,
    EvenDepthVanillaPaths,
    GenerateInstructionSequences,
    IdentifyRelations,
    MergeInstructionSequences,
    RankReduce,
    VanillaInstructionSequences,
)
