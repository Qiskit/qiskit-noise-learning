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

"""Noise learning options."""

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class LearningOptions(BaseModel):
    """Options for the noise learner."""

    num_randomizations: int = Field(32, ge=1)
    """The number of randomizations to use per learning circuit."""

    shots_per_randomizations: int = Field(128, ge=1)
    """The number of shots to use per randomization."""

    depths: list[int] = Field([0, 1, 2, 4, 16, 32])
    """The circuit depths to use."""

    k_locality: int = Field(2, ge=0)
    """The locality of the terms to include in the noise model."""

    path_generator: Literal["even_depth"] = "even_depth"
    """The path generator to use.

    By default, the generator produces even-depth paths for each gate for which to learn the
    noise.
    """

    analyzer: Literal["standard"] = "standard"
    """The analyzer to use.

    By default, the analyzer pipeline first computes observables, then curve fits exponentials, and
    finally uses non-negative least squares to solve for model parameters.
    """

    @field_validator("depths", mode="after")
    @classmethod
    def _nonnegative_list(cls, value: list[int], info: ValidationInfo) -> list[int]:
        if any(i < 0 for i in value):
            raise ValueError(f"`{cls.__name__}.{info.field_name}` option value must all be >= 0")
        return value
