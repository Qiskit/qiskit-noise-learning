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

import pytest
from qiskit.quantum_info import QubitSparsePauliList

from qiskit_noise_learning.math import ComposedLinearMap, IndexedMatrix, IndexedSpace, LinearMap
from qiskit_noise_learning.models import (
    IdentityFidelityModel,
    PauliLindbladModel,
    PauliLindbladSplit,
    contains_pauli_lindblad_model,
    get_noise_site,
    is_fidelity_model,
    split_pauli_lindblad_model,
)
from qiskit_noise_learning.sequences import LogPathMap


@pytest.fixture()
def generators_cz():
    return {
        "CZ": QubitSparsePauliList(["ZI", "XX"]),
        "P": QubitSparsePauliList(["XI", "IX"]),
        "M": QubitSparsePauliList(["XI", "IX"]),
    }


@pytest.fixture()
def plm(gate_set_cz, generators_cz):
    return PauliLindbladModel(gate_set_cz, generators_cz)


class _Empty(IndexedSpace):
    """A trivial empty space, for building toy maps in chains."""

    @property
    def dim(self) -> int:
        return 0

    def __contains__(self, index: object) -> bool:
        return False


class _Toy(LinearMap):
    """A non-PLM LinearMap usable as a filler in a composed chain."""

    def __init__(self):
        space = _Empty()
        super().__init__(input_space=space, output_space=space)

    def rows(self, output_indices):
        return IndexedMatrix()


# --------------------------------------------------------------------------------------------------
# is_fidelity_model
# --------------------------------------------------------------------------------------------------


def test_is_fidelity_model(plm, gate_set_cz):
    assert is_fidelity_model(plm)
    assert is_fidelity_model(IdentityFidelityModel(gate_set_cz))


def test_is_fidelity_model_false_for_path_model(plm):
    # a path model's output space is a LogPathSpace, not a LogFidelitySpace
    path_model = LogPathMap(plm.output_space) @ plm
    assert not is_fidelity_model(path_model)
    assert not is_fidelity_model(LogPathMap(plm.output_space))


# --------------------------------------------------------------------------------------------------
# contains_pauli_lindblad_model
# --------------------------------------------------------------------------------------------------


def test_contains_bare_plm(plm):
    assert contains_pauli_lindblad_model(plm)


def test_contains_composed_with_plm(plm):
    assert contains_pauli_lindblad_model(LogPathMap(plm.output_space) @ plm)


def test_contains_false(gate_set_cz):
    assert not contains_pauli_lindblad_model(IdentityFidelityModel(gate_set_cz))


# --------------------------------------------------------------------------------------------------
# get_noise_site
# --------------------------------------------------------------------------------------------------


def test_get_noise_site_bare_plm(plm):
    assert get_noise_site(plm) == plm.noise_site


def test_get_noise_site_composed_with_single_plm(plm):
    # A chain carrying exactly one PLM returns that PLM's noise site.
    assert get_noise_site(LogPathMap(plm.output_space) @ plm) == plm.noise_site


def test_get_noise_site_none_without_plm(gate_set_cz):
    assert get_noise_site(IdentityFidelityModel(gate_set_cz)) is None


def test_get_noise_site_none_with_multiple_plms(plm):
    assert get_noise_site(ComposedLinearMap([plm, _Toy(), plm])) is None


# --------------------------------------------------------------------------------------------------
# split_pauli_lindblad_model
# --------------------------------------------------------------------------------------------------


def test_split_bare_plm(plm):
    split = split_pauli_lindblad_model(plm)
    assert isinstance(split, PauliLindbladSplit)
    assert split.before is None
    assert split.model is plm
    assert split.after is None


def test_split_with_after_only(plm):
    # LogPathMap @ plm == ComposedLinearMap([plm, LogPathMap]): nothing before, LogPathMap after
    model = LogPathMap(plm.output_space) @ plm

    split = split_pauli_lindblad_model(model)

    assert split.model is plm
    assert split.before is None
    assert isinstance(split.after, ComposedLinearMap)
    # after @ plm reconstructs the chain (same map objects, in order)
    assert (split.after @ plm).maps == model.maps


def test_split_with_before_only(plm):
    before = _Toy()
    model = ComposedLinearMap([before, plm])

    split = split_pauli_lindblad_model(model)

    assert split.model is plm
    assert split.after is None
    assert isinstance(split.before, ComposedLinearMap)
    # plm @ before reconstructs the chain (same map objects, in order)
    assert (plm @ split.before).maps == model.maps


def test_split_with_both_sides(plm):
    before = _Toy()
    after = _Toy()
    model = ComposedLinearMap([before, plm, after])

    split = split_pauli_lindblad_model(model)

    assert split.model is plm
    assert isinstance(split.before, ComposedLinearMap)
    assert isinstance(split.after, ComposedLinearMap)
    assert (split.after @ plm @ split.before).maps == model.maps


def test_split_raises_without_plm(gate_set_cz):
    with pytest.raises(ValueError, match="exactly one PauliLindbladModel"):
        split_pauli_lindblad_model(IdentityFidelityModel(gate_set_cz))


def test_split_raises_with_multiple_plms(plm):
    model = ComposedLinearMap([plm, _Toy(), plm])
    with pytest.raises(ValueError, match="exactly one PauliLindbladModel"):
        split_pauli_lindblad_model(model)
