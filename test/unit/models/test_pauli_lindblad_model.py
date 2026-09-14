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

import warnings
from itertools import combinations, product

import numpy as np
import pytest
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, QubitSparsePauli, QubitSparsePauliList
from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.math import IndexedVector
from qiskit_noise_learning.models import GeneratorIndex, PauliLindbladModel, RateSpace
from qiskit_noise_learning.sequences import FidelityIndex


@pytest.fixture()
def pauli_str():
    return ["Z", "X", "Y"]


@pytest.fixture()
def two_q_pauli_str():
    return [
        "IZ",
        "IX",
        "IY",
        "ZI",
        "ZZ",
        "ZX",
        "ZY",
        "XI",
        "XZ",
        "XX",
        "XY",
        "YI",
        "YZ",
        "YX",
        "YY",
    ]


@pytest.fixture()
def generators_cz():
    return {
        "CZ": QubitSparsePauliList(
            ["ZI", "ZZ", "ZX", "ZY", "XI", "XZ", "XX", "XY", "YI", "YZ", "YX", "YY"]
        ),
        "P": QubitSparsePauliList(["XI", "IX", "XX"]),
        "M": QubitSparsePauliList(["XI", "IX", "XX"]),
    }


def test_construction(gate_set_cz, generators_cz):
    pauli_lindblad_model = PauliLindbladModel(gate_set_cz, generators_cz)
    assert pauli_lindblad_model.gate_set == gate_set_cz
    assert pauli_lindblad_model.generators == generators_cz
    assert pauli_lindblad_model.noise_site == {"CZ": "before", "P": "after", "M": "before"}

    pauli_lindblad_model = PauliLindbladModel(
        gate_set_cz, generators_cz, noise_site={"CZ": "before"}
    )
    assert pauli_lindblad_model.gate_set == gate_set_cz
    assert pauli_lindblad_model.generators == generators_cz
    assert pauli_lindblad_model.noise_site == {"CZ": "before", "P": "after", "M": "before"}


def test_construction_gate_set_errors():
    with pytest.raises(ValueError, match="both measured and prepared"):
        model_gate_set = ModelGateSet(2)
        model_gate_set.add_gate(
            ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2), meas_idxs=[0])
        )
        model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
        PauliLindbladModel(
            model_gate_set, {"P": QubitSparsePauliList(["IX"]), "M": QubitSparsePauliList(["XI"])}
        )

    with pytest.raises(ValueError, match="measurement contains non-trivial unitary part"):
        model_gate_set = ModelGateSet(2)
        model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
        model_gate_set.add_gate(
            ModelGate("M", cliffords=[((0, 1), CZGate())], qubit_idxs=range(2), meas_idxs=range(2))
        )
        PauliLindbladModel(
            model_gate_set, {"P": QubitSparsePauliList(["IX"]), "M": QubitSparsePauliList(["XI"])}
        )

    with pytest.raises(ValueError, match="does not measure all qubits"):
        model_gate_set = ModelGateSet(2)
        model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
        model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=[0]))
        PauliLindbladModel(
            model_gate_set, {"P": QubitSparsePauliList(["IX"]), "M": QubitSparsePauliList(["XI"])}
        )

    with pytest.raises(ValueError, match="preparation contains non-trivial unitary part"):
        model_gate_set = ModelGateSet(2)
        model_gate_set.add_gate(
            ModelGate("P", cliffords=[((0, 1), CZGate())], qubit_idxs=range(2), prep_idxs=range(2))
        )
        model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
        PauliLindbladModel(
            model_gate_set, {"P": QubitSparsePauliList(["IX"]), "M": QubitSparsePauliList(["XI"])}
        )

    with pytest.raises(ValueError, match="does not prepare all qubits"):
        model_gate_set = ModelGateSet(2)
        model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=[0]))
        model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
        PauliLindbladModel(
            model_gate_set, {"P": QubitSparsePauliList(["IX"]), "M": QubitSparsePauliList(["XI"])}
        )

    with pytest.raises(ValueError, match="does not contain a pure measurement"):
        model_gate_set = ModelGateSet(2)
        model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
        PauliLindbladModel(
            model_gate_set, {"P": QubitSparsePauliList(["IX"]), "M": QubitSparsePauliList(["XI"])}
        )

    with pytest.raises(ValueError, match="does not contain a pure preparation"):
        model_gate_set = ModelGateSet(2)
        model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
        PauliLindbladModel(
            model_gate_set, {"P": QubitSparsePauliList(["IX"]), "M": QubitSparsePauliList(["XI"])}
        )


def test_construction_generator_errors(gate_set_cz):
    with pytest.raises(ValueError, match="gate names of gate_set must match the keys"):
        PauliLindbladModel(gate_set_cz, {"P": QubitSparsePauliList(["IX"])})

    with pytest.raises(ValueError, match="gate names of gate_set must match the keys"):
        PauliLindbladModel(
            gate_set_cz,
            {
                "R": QubitSparsePauliList(["IX"]),
                "P": QubitSparsePauliList(["IX"]),
                "M": QubitSparsePauliList(["IX"]),
            },
        )

    with pytest.raises(ValueError, match="Generators for a given gate must be unique."):
        PauliLindbladModel(
            gate_set_cz,
            {
                "CZ": QubitSparsePauliList(["XX", "XX"]),
                "P": QubitSparsePauliList(["IX"]),
                "M": QubitSparsePauliList(["IX"]),
            },
        )

    with pytest.raises(
        ValueError, match="Generators must act only on the qubits the gate acts on."
    ):
        PauliLindbladModel(
            gate_set_cz,
            {
                "CZ": QubitSparsePauliList(["XXI"]),
                "P": QubitSparsePauliList(["IX"]),
                "M": QubitSparsePauliList(["IX"]),
            },
        )


def test_construction_spam_generators_must_be_x_only(gate_set_cz, generators_cz):
    with pytest.raises(ValueError, match="contains a Pauli other than X"):
        PauliLindbladModel(gate_set_cz, {**generators_cz, "P": QubitSparsePauliList(["IZ"])})

    with pytest.raises(ValueError, match="contains a Pauli other than X"):
        PauliLindbladModel(gate_set_cz, {**generators_cz, "M": QubitSparsePauliList(["XI", "IY"])})

    # a mixed-Pauli generator on a Clifford gate is still fine
    PauliLindbladModel(gate_set_cz, generators_cz)


def test_noise_site_errors(gate_set_cz, generators_cz):
    with pytest.raises(ValueError, match="Preparation noise models must occur after the gate."):
        PauliLindbladModel(
            gate_set=gate_set_cz, generators=generators_cz, noise_site={"P": "before"}
        )

    with pytest.raises(ValueError, match="Measurement noise models must occur before the gate."):
        PauliLindbladModel(
            gate_set=gate_set_cz, generators=generators_cz, noise_site={"M": "after"}
        )

    with pytest.raises(ValueError, match="can only take values"):
        PauliLindbladModel(
            gate_set=gate_set_cz, generators=generators_cz, noise_site={"CZ": "other"}
        )


def test_rows(gate_set_cz, generators_cz):
    pauli_lindblad_model = PauliLindbladModel(gate_set=gate_set_cz, generators=generators_cz)
    fidelity = FidelityIndex.from_gate(
        gate=gate_set_cz["CZ"],
        pauli=QubitSparsePauli("IX"),
        in_z_idxs=frozenset(),
        out_z_idxs=frozenset(),
    )
    # should be the generators that anticommute with ZX
    indices = [
        GeneratorIndex("CZ", pauli)
        for pauli in QubitSparsePauliList(["ZZ", "ZY", "XI", "XX", "YI", "YX"])
    ]
    expected = IndexedVector({index: 2.0 for index in indices})
    assert pauli_lindblad_model.rows([fidelity])[fidelity] == expected

    # same setup but with noise model after gate
    pauli_lindblad_model = PauliLindbladModel(
        gate_set=gate_set_cz, generators=generators_cz, noise_site={"CZ": "after"}
    )
    fidelity = FidelityIndex.from_gate(
        gate=gate_set_cz["CZ"],
        pauli=QubitSparsePauli("IX"),
        in_z_idxs=frozenset(),
        out_z_idxs=frozenset(),
    )
    indices = [
        GeneratorIndex("CZ", pauli)
        for pauli in QubitSparsePauliList(["ZZ", "ZY", "XZ", "XY", "YZ", "YY"])
    ]
    expected = IndexedVector({k: 2.0 for k in indices})
    assert pauli_lindblad_model.rows([fidelity])[fidelity] == expected

    # test measurement - pauli is ZI
    fidelity = FidelityIndex.from_gate(
        gate=gate_set_cz["M"],
        pauli=QubitSparsePauli("II"),
        in_z_idxs=frozenset([1]),
        out_z_idxs=frozenset(),
    )
    indices = [GeneratorIndex("M", pauli) for pauli in QubitSparsePauliList(["XI", "XX"])]
    expected = IndexedVector({k: 2.0 for k in indices})
    assert pauli_lindblad_model.rows([fidelity])[fidelity] == expected


def test_rows_unknown_gate_raises(gate_set_cz, generators_cz):
    # setup model with no CZ
    model_gate_set = ModelGateSet(2)
    model_gate_set.add_gate(gate_set_cz["P"])
    model_gate_set.add_gate(gate_set_cz["M"])
    pauli_lindblad_model = PauliLindbladModel(
        gate_set=model_gate_set, generators={"P": generators_cz["P"], "M": generators_cz["M"]}
    )

    fidelity = FidelityIndex.from_gate(
        gate=gate_set_cz["CZ"],
        pauli=QubitSparsePauli("IX"),
        in_z_idxs=frozenset(),
        out_z_idxs=frozenset(),
    )
    with pytest.raises(ValueError, match="Gate with name CZ not in gate set."):
        pauli_lindblad_model.rows([fidelity])


@pytest.mark.parametrize(
    ("cz_generators", "expected_labels"),
    [
        # a single generator that anticommutes with the transition Pauli ZX
        (["ZZ"], ["ZZ"]),
        # a single generator that commutes with ZX, giving an empty row
        (["ZI"], []),
        # a gate carrying no generators at all
        ([], []),
        # two generators, only one of which anticommutes
        (["ZI", "ZZ"], ["ZZ"]),
    ],
    ids=["one-anticommuting", "one-commuting", "no-generators", "two-mixed"],
)
def test_rows_few_generators(gate_set_cz, generators_cz, cz_generators, expected_labels):
    """Rows are correct when a gate has zero, one, or two generators."""
    generators = dict(generators_cz)
    generators["CZ"] = (
        QubitSparsePauliList(cz_generators)
        if cz_generators
        else QubitSparsePauliList.empty(gate_set_cz.num_qubits)
    )
    pauli_lindblad_model = PauliLindbladModel(gate_set=gate_set_cz, generators=generators)

    fidelity = FidelityIndex.from_gate(
        gate=gate_set_cz["CZ"],
        pauli=QubitSparsePauli("IX"),
        in_z_idxs=frozenset(),
        out_z_idxs=frozenset(),
    )
    rows = pauli_lindblad_model.rows([fidelity])

    if not expected_labels:
        # rows with no non-zero entries are dropped from the matrix entirely
        assert fidelity not in rows.row_index_map
    else:
        expected = IndexedVector(
            {GeneratorIndex("CZ", pauli): 2.0 for pauli in QubitSparsePauliList(expected_labels)}
        )
        assert rows[fidelity] == expected


def test_k_partition_local(two_q_pauli_str, gate_set_cz):
    pauli_lindblad_model = PauliLindbladModel.k_partition_local(gate_set=gate_set_cz, k=2)

    expected_generators = {
        "CZ": QubitSparsePauliList(two_q_pauli_str),
        "P": QubitSparsePauliList(["IX", "XI", "XX"]),
        "M": QubitSparsePauliList(["IX", "XI", "XX"]),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]

    # double CZ layer 1 local
    model_gate_set = ModelGateSet(4)
    model_gate_set.add_gate(
        ModelGate("CZ", [((0, 1), Clifford(CZGate())), ((2, 3), Clifford(CZGate()))])
    )
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(4), prep_idxs=range(4)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(4), meas_idxs=range(4)))

    pauli_lindblad_model = PauliLindbladModel.k_partition_local(gate_set=model_gate_set, k=1)
    expected_generators = {
        "CZ": QubitSparsePauliList(
            [s + "II" for s in two_q_pauli_str] + ["II" + s for s in two_q_pauli_str]
        ),
        "P": QubitSparsePauliList(["XIII", "IXII", "IIXI", "IIIX"]),
        "M": QubitSparsePauliList(["XIII", "IXII", "IIXI", "IIIX"]),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]

    # double CZ layer 2 local
    model_gate_set = ModelGateSet(4)
    model_gate_set.add_gate(
        ModelGate("CZ", [((0, 1), Clifford(CZGate())), ((2, 3), Clifford(CZGate()))])
    )
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(4), prep_idxs=range(4)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(4), meas_idxs=range(4)))

    pauli_lindblad_model = PauliLindbladModel.k_partition_local(gate_set=model_gate_set, k=2)
    expected_generators = {
        "CZ": QubitSparsePauliList(
            # 1-local
            [s + "II" for s in two_q_pauli_str]
            + ["II" + s for s in two_q_pauli_str]
            # 2-local
            + [s1 + s2 for s1, s2 in product(two_q_pauli_str, repeat=2)]
        ),
        "P": QubitSparsePauliList(
            ["XIII", "IXII", "IIXI", "IIIX", "XXII", "XIXI", "XIIX", "IXXI", "IXIX", "IIXX"]
        ),
        "M": QubitSparsePauliList(
            ["XIII", "IXII", "IIXI", "IIIX", "XXII", "XIXI", "XIIX", "IXXI", "IXIX", "IIXX"]
        ),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]

    # singles and doubles
    model_gate_set = ModelGateSet(3)
    model_gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))], qubit_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(3), prep_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(3), meas_idxs=range(3)))

    pauli_lindblad_model = PauliLindbladModel.k_partition_local(gate_set=model_gate_set, k=2)
    expected_generators = {
        "CZ": QubitSparsePauliList(
            # 1-local
            ["I" + s for s in two_q_pauli_str]
            + [s + "II" for s in ["Z", "X", "Y"]]
            # 2-local
            + [s1 + s2 for s1, s2 in product(["Z", "X", "Y"], two_q_pauli_str)]
        ),
        "P": QubitSparsePauliList(["XII", "IXI", "IIX", "XXI", "XIX", "IXX"]),
        "M": QubitSparsePauliList(["XII", "IXI", "IIX", "XXI", "XIX", "IXX"]),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]

    # 3-local
    model_gate_set = ModelGateSet(3)
    model_gate_set.add_gate(ModelGate("CZ", [], qubit_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(3), prep_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(3), meas_idxs=range(3)))

    pauli_lindblad_model = PauliLindbladModel.k_partition_local(gate_set=model_gate_set, k=3)
    expected_generators = {
        "CZ": QubitSparsePauliList(
            [s0 + s1 + s2 for s0, s1, s2 in product(["I", "Z", "X", "Y"], repeat=3)][1:]
        ),
        "P": QubitSparsePauliList(["XII", "IXI", "IIX", "XXI", "XIX", "IXX", "XXX"]),
        "M": QubitSparsePauliList(["XII", "IXI", "IIX", "XXI", "XIX", "IXX", "XXX"]),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]


def test_k_partition_local_default_coupling_map(two_q_pauli_str):
    """Test drawing coupling map from various places."""

    # if unspecified in gate set, use complete coupling map
    # double CZ layer 2 local
    model_gate_set = ModelGateSet(4)
    model_gate_set.add_gate(
        ModelGate("CZ", [((0, 1), Clifford(CZGate())), ((2, 3), Clifford(CZGate()))])
    )
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(4), prep_idxs=range(4)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(4), meas_idxs=range(4)))

    pauli_lindblad_model = PauliLindbladModel.k_partition_local(gate_set=model_gate_set, k=2)
    expected_generators = {
        "CZ": QubitSparsePauliList(
            # 1-local
            [s + "II" for s in two_q_pauli_str]
            + ["II" + s for s in two_q_pauli_str]
            # 2-local
            + [s1 + s2 for s1, s2 in product(two_q_pauli_str, repeat=2)]
        ),
        "P": QubitSparsePauliList(
            ["XIII", "IXII", "IIXI", "IIIX", "XXII", "XIXI", "XIIX", "IXXI", "IXIX", "IIXX"]
        ),
        "M": QubitSparsePauliList(
            ["XIII", "IXII", "IIXI", "IIIX", "XXII", "XIXI", "XIIX", "IXXI", "IXIX", "IIXX"]
        ),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]

    # use gate set coupling map if present
    model_gate_set = ModelGateSet(4, coupling_map=CouplingMap([[0, 1], [1, 2], [2, 3]]))
    model_gate_set.add_gate(
        ModelGate("CZ", [((0, 1), Clifford(CZGate())), ((2, 3), Clifford(CZGate()))])
    )
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(4), prep_idxs=range(4)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(4), meas_idxs=range(4)))

    pauli_lindblad_model = PauliLindbladModel.k_partition_local(gate_set=model_gate_set, k=2)
    expected_generators = {
        "CZ": QubitSparsePauliList(
            # 1-local
            [s + "II" for s in two_q_pauli_str]
            + ["II" + s for s in two_q_pauli_str]
            # 2-local
            + [s1 + s2 for s1, s2 in product(two_q_pauli_str, repeat=2)]
        ),
        "P": QubitSparsePauliList(["XIII", "IXII", "IIXI", "IIIX", "XXII", "IXXI", "IIXX"]),
        "M": QubitSparsePauliList(["XIII", "IXII", "IIXI", "IIIX", "XXII", "IXXI", "IIXX"]),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]


def test_k_partition_local_errors(gate_set_cz):
    with pytest.raises(ValueError, match="k:`3`"):
        PauliLindbladModel.k_partition_local(gate_set=gate_set_cz, k=3)

    # qubit_partitions has invalid name
    with pytest.raises(ValueError, match="Gates {'not_in_gate_set'}"):
        PauliLindbladModel.k_partition_local(
            gate_set=gate_set_cz, k=1, qubit_partitions={"not_in_gate_set": [{0}, {1}]}
        )

    # partition not disjoint
    with pytest.raises(ValueError, match="CZ contains duplicates"):
        PauliLindbladModel.k_partition_local(
            gate_set=gate_set_cz, k=1, qubit_partitions={"CZ": [{0, 1}, {1}]}
        )

    # partition missing elements
    with pytest.raises(ValueError, match="Union of qubit partition"):
        PauliLindbladModel.k_partition_local(
            gate_set=gate_set_cz, k=1, qubit_partitions={"CZ": [{0}]}
        )

    # partition contains qubits it shouldn't
    with pytest.raises(ValueError, match="Union of qubit partition"):
        PauliLindbladModel.k_partition_local(
            gate_set=gate_set_cz, k=1, qubit_partitions={"CZ": [{0, 1, 2}]}
        )

    # local_paulis has invalid name
    with pytest.raises(ValueError, match="Gates {'not_in_gate_set'}"):
        PauliLindbladModel.k_partition_local(
            gate_set=gate_set_cz,
            k=1,
            local_paulis={"not_in_gate_set": [QubitSparsePauliList(["X"])]},
        )

    # len(local_paulis["CZ"]) too short
    with pytest.raises(ValueError, match="less than largest partition"):
        PauliLindbladModel.k_partition_local(
            gate_set=gate_set_cz, k=1, local_paulis={"CZ": [QubitSparsePauliList(["X"])]}
        )

    # local_paulis["CZ"][1] incorrect number of qubits
    with pytest.raises(ValueError, match="num_qubits != 2"):
        PauliLindbladModel.k_partition_local(
            gate_set=gate_set_cz, k=1, local_paulis={"CZ": [QubitSparsePauliList(["X"])] * 2}
        )


def test_k_partition_local_per_gate_k(two_q_pauli_str):
    model_gate_set = ModelGateSet(4)
    model_gate_set.add_gate(
        ModelGate("CZ", [((0, 1), Clifford(CZGate())), ((2, 3), Clifford(CZGate()))])
    )
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(4), prep_idxs=range(4)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(4), meas_idxs=range(4)))

    # CZ gets k=2 (includes cross-partition terms), P and M get k=1 via default
    pauli_lindblad_model = PauliLindbladModel.k_partition_local(
        gate_set=model_gate_set, k=1, gate_k={"CZ": 2}
    )

    expected_cz_generators = QubitSparsePauliList(
        # 1-local
        [s + "II" for s in two_q_pauli_str]
        + ["II" + s for s in two_q_pauli_str]
        # 2-local
        + [s1 + s2 for s1, s2 in product(two_q_pauli_str, repeat=2)]
    )
    expected_p_generators = QubitSparsePauliList(["XIII", "IXII", "IIXI", "IIIX"])
    expected_m_generators = QubitSparsePauliList(["XIII", "IXII", "IIXI", "IIIX"])

    assert len(pauli_lindblad_model.generators["CZ"]) == len(expected_cz_generators)
    for generator in pauli_lindblad_model.generators["CZ"]:
        assert generator in expected_cz_generators

    assert len(pauli_lindblad_model.generators["P"]) == len(expected_p_generators)
    for generator in pauli_lindblad_model.generators["P"]:
        assert generator in expected_p_generators

    assert len(pauli_lindblad_model.generators["M"]) == len(expected_m_generators)
    for generator in pauli_lindblad_model.generators["M"]:
        assert generator in expected_m_generators


def test_k_partition_local_per_gate_k_errors(gate_set_cz):
    # gate_k has extra gate not in gate_set
    with pytest.raises(ValueError, match="not in gate_set"):
        PauliLindbladModel.k_partition_local(gate_set=gate_set_cz, gate_k={"X": 1})

    # gate_k value too large
    with pytest.raises(ValueError, match="k:`3`"):
        PauliLindbladModel.k_partition_local(gate_set=gate_set_cz, gate_k={"CZ": 3})


def test_k_local(two_q_pauli_str, pauli_str, gate_set_cz):
    pauli_lindblad_model = PauliLindbladModel.k_local(gate_set=gate_set_cz, k=2)

    expected_generators = {
        "CZ": QubitSparsePauliList(two_q_pauli_str),
        "P": QubitSparsePauliList(["IX", "XI", "XX"]),
        "M": QubitSparsePauliList(["IX", "XI", "XX"]),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]

    # double CZ layer 1 local
    model_gate_set = ModelGateSet(4)
    model_gate_set.add_gate(
        ModelGate("CZ", [((0, 1), Clifford(CZGate())), ((2, 3), Clifford(CZGate()))])
    )
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(4), prep_idxs=range(4)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(4), meas_idxs=range(4)))

    pauli_lindblad_model = PauliLindbladModel.k_local(gate_set=model_gate_set, k=1)
    expected_generators = {
        "CZ": QubitSparsePauliList(
            [s + "III" for s in pauli_str]
            + ["I" + s + "II" for s in pauli_str]
            + ["II" + s + "I" for s in pauli_str]
            + ["III" + s for s in pauli_str]
        ),
        "P": QubitSparsePauliList(["XIII", "IXII", "IIXI", "IIIX"]),
        "M": QubitSparsePauliList(["XIII", "IXII", "IIXI", "IIIX"]),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]

    # double CZ layer 2 local
    model_gate_set = ModelGateSet(4)
    model_gate_set.add_gate(
        ModelGate("CZ", [((0, 1), Clifford(CZGate())), ((2, 3), Clifford(CZGate()))])
    )
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(4), prep_idxs=range(4)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(4), meas_idxs=range(4)))

    pauli_lindblad_model = PauliLindbladModel.k_local(gate_set=model_gate_set, k=2)
    expected_generators = {
        "CZ": QubitSparsePauliList(
            # 1 local
            [s + "III" for s in pauli_str]
            + ["I" + s + "II" for s in pauli_str]
            + ["II" + s + "I" for s in pauli_str]
            + ["III" + s for s in pauli_str]
            # 2 local
            + [s1 + s2 + "II" for s1, s2 in product(pauli_str, repeat=2)]
            + [s1 + "I" + s2 + "I" for s1, s2 in product(pauli_str, repeat=2)]
            + [s1 + "II" + s2 for s1, s2 in product(pauli_str, repeat=2)]
            + ["I" + s1 + s2 + "I" for s1, s2 in product(pauli_str, repeat=2)]
            + ["I" + s1 + "I" + s2 for s1, s2 in product(pauli_str, repeat=2)]
            + ["II" + s1 + s2 for s1, s2 in product(pauli_str, repeat=2)]
        ),
        "P": QubitSparsePauliList(
            ["XIII", "IXII", "IIXI", "IIIX", "XXII", "XIXI", "XIIX", "IXXI", "IXIX", "IIXX"]
        ),
        "M": QubitSparsePauliList(
            ["XIII", "IXII", "IIXI", "IIIX", "XXII", "XIXI", "XIIX", "IXXI", "IXIX", "IIXX"]
        ),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]

    # singles and doubles
    model_gate_set = ModelGateSet(3)
    model_gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))], qubit_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(3), prep_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(3), meas_idxs=range(3)))

    pauli_lindblad_model = PauliLindbladModel.k_local(gate_set=model_gate_set, k=2)
    expected_generators = {
        "CZ": QubitSparsePauliList(
            # 1 local
            [s + "II" for s in pauli_str]
            + ["I" + s + "I" for s in pauli_str]
            + ["II" + s for s in pauli_str]
            # 2 local
            + [s1 + s2 + "I" for s1, s2 in product(pauli_str, repeat=2)]
            + [s1 + "I" + s2 for s1, s2 in product(pauli_str, repeat=2)]
            + ["I" + s1 + s2 for s1, s2 in product(pauli_str, repeat=2)]
        ),
        "P": QubitSparsePauliList(["XII", "IXI", "IIX", "XXI", "XIX", "IXX"]),
        "M": QubitSparsePauliList(["XII", "IXI", "IIX", "XXI", "XIX", "IXX"]),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]

    # 3-local
    model_gate_set = ModelGateSet(3)
    model_gate_set.add_gate(ModelGate("CZ", [], qubit_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(3), prep_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(3), meas_idxs=range(3)))

    pauli_lindblad_model = PauliLindbladModel.k_local(gate_set=model_gate_set, k=3)
    expected_generators = {
        "CZ": QubitSparsePauliList(
            [s0 + s1 + s2 for s0, s1, s2 in product(["I", "Z", "X", "Y"], repeat=3)][1:]
        ),
        "P": QubitSparsePauliList(["XII", "IXI", "IIX", "XXI", "XIX", "IXX", "XXX"]),
        "M": QubitSparsePauliList(["XII", "IXI", "IIX", "XXI", "XIX", "IXX", "XXX"]),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]


def test_k_local_empty(gate_set_cz):
    pauli_lindblad_model = PauliLindbladModel.k_local(
        gate_set=gate_set_cz, k=2, paulis={"CZ": QubitSparsePauliList.empty(1)}
    )

    expected_generators = {
        "CZ": QubitSparsePauliList.empty(2),
        "P": QubitSparsePauliList(["IX", "XI", "XX"]),
        "M": QubitSparsePauliList(["IX", "XI", "XX"]),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]


def test_k_local_errors(gate_set_cz):
    # paulis has invalid name
    with pytest.raises(ValueError, match="Gates {'not_in_gate_set'}"):
        PauliLindbladModel.k_local(
            gate_set=gate_set_cz,
            k=1,
            paulis={"not_in_gate_set": QubitSparsePauliList(["X"])},
        )

    # invalid num qubits
    with pytest.raises(ValueError, match="!= 1"):
        PauliLindbladModel.k_local(
            gate_set=gate_set_cz,
            k=1,
            paulis={"CZ": QubitSparsePauliList(["XY"])},
        )


def test_k_local_per_gate_k(two_q_pauli_str, pauli_str, gate_set_cz):
    pauli_lindblad_model = PauliLindbladModel.k_local(gate_set=gate_set_cz, k=1, gate_k={"CZ": 2})

    expected_generators = {
        "CZ": QubitSparsePauliList(two_q_pauli_str),
        "P": QubitSparsePauliList(["IX", "XI"]),
        "M": QubitSparsePauliList(["IX", "XI"]),
    }

    for name, generator_list in pauli_lindblad_model.generators.items():
        assert len(generator_list) == len(expected_generators[name])
        for generator in generator_list:
            assert generator in expected_generators[name]


def test_to_pauli_lindblad_maps(gate_set_cz, generators_cz):
    """Test to_pauli_lindblad_maps returns one PauliLindbladMap per non-SPAM gate."""
    model = PauliLindbladModel(gate_set_cz, generators_cz)
    parameter_indices = []
    parameter_values = []
    for gate_name, gen_list in generators_cz.items():
        for gen in gen_list:
            parameter_indices.append(GeneratorIndex(gate_name, gen))
            parameter_values.append(0.01)

    covariance = np.eye(len(parameter_values), dtype=float)
    model_fit = ModelData.from_arrays(
        parameter_indices=parameter_indices,
        parameter_values=np.array(parameter_values),
        covariance=covariance,
        time_lbs=np.empty(len(parameter_indices), dtype="datetime64[us]"),
        time_ubs=np.empty(len(parameter_indices), dtype="datetime64[us]"),
    )

    maps_no_spam = model.to_pauli_lindblad_maps(model_fit, include_spam=False)
    assert "CZ" in maps_no_spam
    assert "P" not in maps_no_spam
    assert "M" not in maps_no_spam

    maps_with_spam = model.to_pauli_lindblad_maps(model_fit, include_spam=True)
    assert "CZ" in maps_with_spam
    assert "P" in maps_with_spam
    assert "M" in maps_with_spam


def test_to_pauli_lindblad_maps_raises(gate_set_cz, generators_cz):
    """Test to_pauli_lindblad_maps raises ValueError for unknown gate names."""
    model = PauliLindbladModel(gate_set_cz, generators_cz)

    bad_gen_index = GeneratorIndex("UNKNOWN", QubitSparsePauli("IX"))
    model_fit = ModelData.from_arrays(
        parameter_indices=[bad_gen_index],
        parameter_values=np.array([0.01]),
        covariance=np.array([[1.0]]),
        time_lbs=np.empty(1, dtype="datetime64[us]"),
        time_ubs=np.empty(1, dtype="datetime64[us]"),
    )

    with pytest.raises(ValueError, match="not present in gate set"):
        model.to_pauli_lindblad_maps(model_fit)


# --------------------------------------------------------------------------------------------------
# to_pauli_lindblad_maps, symmetrize_spam argument
# --------------------------------------------------------------------------------------------------


@pytest.fixture()
def gate_set_3q():
    """A 3-qubit gate set with only pure preparation ``P`` and measurement ``M``."""
    model_gate_set = ModelGateSet(3)
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(3), prep_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(3), meas_idxs=range(3)))
    return model_gate_set


def _pauli(letter, support, num_qubits):
    return QubitSparsePauli.from_sparse_label((letter * len(support), list(support)), num_qubits)


def _spam_model_and_data(gate_set, rates, num_qubits):
    """Build an X-only SPAM model, and fit data for it, from a mapping from support to rate."""
    generators = {
        name: QubitSparsePauliList.from_qubit_sparse_paulis(
            [_pauli("X", support, num_qubits) for support in rates]
        )
        for name in gate_set
    }
    parameter_indices = [
        GeneratorIndex(name, _pauli("X", support, num_qubits))
        for name in gate_set
        for support in rates
    ]
    size = len(parameter_indices)

    model_data = ModelData.from_arrays(
        parameter_indices=parameter_indices,
        parameter_values=np.array([rates[support] for _ in gate_set for support in rates]),
        covariance=np.eye(size),
        time_lbs=np.empty(size, dtype="datetime64[us]"),
        time_ubs=np.empty(size, dtype="datetime64[us]"),
    )
    return PauliLindbladModel(gate_set, generators), model_data


def _rates_by_support(noise_map):
    """A mapping from qubit subset to the set of rates carried by the Paulis on that subset."""
    rates = {}
    for generator, rate in zip(noise_map.generators(), noise_map.rates):
        rates.setdefault(tuple(sorted(int(idx) for idx in generator.indices)), set()).add(
            float(rate)
        )
    return rates


def test_symmetrize_spam_one_local(gate_set_cz):
    """Each single-qubit rate is halved and spread over X, Y and Z."""
    rates = {(0,): 1e-2, (1,): 8e-3}
    model, model_data = _spam_model_and_data(gate_set_cz, rates, 2)

    symmetrized = model.to_pauli_lindblad_maps(model_data, True, symmetrize_spam=True)["M"]

    assert _rates_by_support(symmetrized) == {(0,): {5e-3}, (1,): {4e-3}}
    assert symmetrized.num_terms == 6


def test_symmetrize_spam_two_local(gate_set_cz, generators_cz):
    """A pair rate quarters, and corrects the single-qubit rates it overlaps."""
    rates = {(0,): 1e-2, (1,): 8e-3, (0, 1): 1e-3}
    model, model_data = _spam_model_and_data(gate_set_cz, rates, 2)

    symmetrized = model.to_pauli_lindblad_maps(model_data, True, symmetrize_spam=True)["M"]

    # r_j = x_j / 2 - x_01 / 4, and r_01 = x_01 / 4
    assert _rates_by_support(symmetrized) == {
        (0,): {4.75e-3},
        (1,): {3.75e-3},
        (0, 1): {2.5e-4},
    }
    assert symmetrized.num_terms == 15

    # left alone, the map keeps the X-only generators it was fit with
    plain = model.to_pauli_lindblad_maps(model_data, True, symmetrize_spam=False)["M"]
    assert plain.to_sparse_list() == [("X", [0], 1e-2), ("X", [1], 8e-3), ("XX", [0, 1], 1e-3)]


def test_symmetrize_spam_three_local(gate_set_3q):
    """A weight-three rate pulls in the subsets of its support, including absent ones."""
    # the connected 3-local supports on the path 0 - 1 - 2, which omit (0, 2)
    rates = {
        (0,): 1e-2,
        (1,): 1e-2,
        (2,): 1e-2,
        (0, 1): 1e-3,
        (1, 2): 1e-3,
        (0, 1, 2): 1e-4,
    }
    model, model_data = _spam_model_and_data(gate_set_3q, rates, 3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        symmetrized = model.to_pauli_lindblad_maps(model_data, True, symmetrize_spam=True)["M"]
        plain = model.to_pauli_lindblad_maps(model_data, True, symmetrize_spam=False)["M"]

    # (0, 2) is absent from the model but is a subset of (0, 1, 2), so it is introduced, one qubit
    # smaller and therefore with a single negated term
    assert _rates_by_support(symmetrized) == {
        (0,): {4.7625e-3},
        (1,): {4.5125e-3},
        (2,): {4.7625e-3},
        (0, 1): {2.375e-4},
        (0, 2): {-1.25e-5},
        (1, 2): {2.375e-4},
        (0, 1, 2): {1.25e-5},
    }
    assert symmetrized.num_terms == 63

    # the point of the exercise: the two maps act identically on every Z-type Pauli
    for size in [1, 2, 3]:
        for support in combinations(range(3), size):
            pauli = _pauli("Z", support, 3)
            assert symmetrized.pauli_fidelity(pauli) == pytest.approx(plain.pauli_fidelity(pauli))


def test_symmetrize_spam_warns_only_on_new_negative_rates(gate_set_cz):
    """A negative rate is worth flagging only if the rates being symmetrized had none."""
    # a pair rate above twice the single-qubit rates drives r_j negative
    model, model_data = _spam_model_and_data(gate_set_cz, {(0,): 1e-3, (1,): 1e-3, (0, 1): 5e-3}, 2)
    with pytest.warns(UserWarning, match="negative rate"):
        model.to_pauli_lindblad_maps(model_data, True, symmetrize_spam=True)

    # a caller whose fit already has negative rates has accepted them
    model, model_data = _spam_model_and_data(
        gate_set_cz, {(0,): -1e-3, (1,): 5e-3, (0, 1): 5e-3}, 2
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.to_pauli_lindblad_maps(model_data, True, symmetrize_spam=True)


def test_rate_space_dim(generators_cz):
    # 12 (CZ) + 3 (P) + 3 (M)
    assert RateSpace(generators_cz).dim == 18


def test_rate_space_contains(generators_cz):
    rate_space = RateSpace(generators_cz)
    cz_generator = next(iter(generators_cz["CZ"]))

    assert GeneratorIndex("CZ", cz_generator) in rate_space
    # a valid Pauli that is not among the CZ generators (IZ, IX, IY are absent)
    assert GeneratorIndex("CZ", QubitSparsePauli("IZ")) not in rate_space
    # correct generator but a gate name not in the space
    assert GeneratorIndex("nope", cz_generator) not in rate_space
    # a non-GeneratorIndex object
    assert "not a generator index" not in rate_space
