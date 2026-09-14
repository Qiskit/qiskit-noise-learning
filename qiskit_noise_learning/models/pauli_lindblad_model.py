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

"""PauliLindbladModel"""

import warnings
from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass
from itertools import chain, combinations, product
from typing import Literal, Self

import numpy as np
from qiskit.quantum_info import PauliLindbladMap, QubitSparsePauli, QubitSparsePauliList
from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.gate_sets import GateSet, ModelGateSet
from qiskit_noise_learning.math import IndexedMatrix, IndexedSpace, IndexedVector, LinearMap
from qiskit_noise_learning.sequences import FidelityIndex

from .log_fidelity_space import LogFidelitySpace


@dataclass
class GeneratorIndex:
    gate_name: str
    generator: QubitSparsePauli

    def __hash__(self) -> int:
        if not hasattr(self, "_hash"):
            self._hash = hash(
                (self.gate_name, tuple(self.generator.paulis), tuple(self.generator.indices))
            )
        return self._hash


class RateSpace(IndexedSpace[GeneratorIndex]):
    """Space of Pauli-Lindblad generator rates, indexed by :class:`GeneratorIndex`.

    Args:
        generators: A dictionary mapping gate name to the gate's Pauli-Lindblad generators.
    """

    def __init__(self, generators: dict[str, QubitSparsePauliList]):
        self._generators = generators

    @property
    def generators(self) -> dict[str, QubitSparsePauliList]:
        """The Pauli-Lindblad generators for each gate."""
        return self._generators

    @property
    def dim(self) -> int | float:
        return sum(len(gate_generators) for gate_generators in self._generators.values())

    def __contains__(self, index: object) -> bool:
        if not isinstance(index, GeneratorIndex) or index.gate_name not in self._generators:
            return False
        return any(index.generator == generator for generator in self._generators[index.gate_name])


class PauliLindbladModel(LinearMap[GeneratorIndex, FidelityIndex]):
    r"""A linear mapping from Pauli-Lindblad generator rates to log fidelities.

    This model class assumes that every gate in the gate set is either a unitary, a pure preparation
    (all qubits are prepared and the unitary part is trivial), or a pure measurement (all qubits are
    measured and the unitary part is trivial). It is further assumed that at least one preparation
    and at least one measurement gate are present.

    The noise model for each gate in this case is a Pauli channel, parameterized in terms of the
    rates of a Pauli-Lindblad decomposition :math:`\exp(\sum_{P \in \mathcal{P}_n} r_P L(P))`, where
    the :math:`r_P` are the rates, and :math:`L(P) = P \cdot P - \cdot`. In the case of unitary
    gates, the noise can be modelled as either occuring before or after the ideal unitary.

    Args:
        gate_set: The gate set whose fidelities are being modelled. To be converted to a
            :class:`ModelGateSet`.
        generators: A dictionary mapping gate name to the set of Pauli-Lindblad generators for the
            noise model of that gate. The generators for each gate must be unique. Generators for
            preparation and measurement layers must be built from I and X only.
        noise_site: A dictionary specifying, for each gate name, whether the noise model occurs
            before or after the gate, indicated with strings ``"before"`` and ``"after"``. Any
            unspecified values for the gate set will be populated with default values: ``"before"``
            for unitary gates and pure measurement gates, and ``"after"`` for pure preparation. An
            error will be raised if a value for pure measurement or preparation is specified that
            differs from the default.

    Raises:
        ValueError: If the gate set is not of the required form, or if ``noise_model_before_gate``
            has any invalid values.
    """

    def __init__(
        self,
        gate_set: GateSet,
        generators: dict[str, QubitSparsePauliList],
        noise_site: dict[str, str] | None = None,
    ):
        gate_set = gate_set.model_gate_set
        prep_names, meas_names = _validate_gate_set_form(gate_set)

        self._meas_names = set(meas_names)
        self._prep_names = set(prep_names)

        _validate_generators(gate_set, generators, prep_names, meas_names)

        self._noise_site = _validate_and_complete_noise_site_dict(
            gate_set, noise_site, prep_names, meas_names
        )

        super().__init__(
            input_space=RateSpace(generators),
            output_space=LogFidelitySpace(gate_set),
        )

    @property
    def gate_set(self) -> ModelGateSet:
        """The gate set whose fidelities are being modelled."""
        return self.output_space.gate_set

    @property
    def generators(self) -> dict[str, QubitSparsePauliList]:
        """The generators for the noise model."""
        return self.input_space.generators

    @property
    def meas_names(self) -> set[str]:
        """The names of the measurements in this model."""
        return self._meas_names

    @property
    def noise_site(self) -> dict[str, str]:
        """Whether each noise map occurs before or after each gate."""
        return self._noise_site

    @property
    def prep_names(self) -> set[str]:
        """The names of the preparation in this model."""
        return self._prep_names

    def rows(
        self, output_indices: Iterable[FidelityIndex]
    ) -> IndexedMatrix[FidelityIndex, GeneratorIndex]:
        """Construct the sub-matrix whose rows are the given fidelity indices.

        Args:
            output_indices: The fidelity indices labelling the desired rows.

        Returns:
            An :class:`~.IndexedMatrix` indexed by the (non-zero) requested fidelity indices and by
            the generator indices appearing in those rows.

        Raises:
            ValueError: If any fidelity index labels a gate that is not in the model.
        """
        output_indices = list(output_indices)

        generator_lookup: dict[str, tuple[list[GeneratorIndex], QubitSparsePauliList | None]] = {}
        for gate_name in dict.fromkeys(index.gate_name for index in output_indices):
            if gate_name not in self.generators:
                raise ValueError(f"Gate with name {gate_name} not in gate set.")
            generators = self.generators[gate_name]
            generator_indices = [
                GeneratorIndex(gate_name=gate_name, generator=generator) for generator in generators
            ]
            generator_lookup[gate_name] = (
                generator_indices,
                generators if len(generators) > 0 else None,
            )

        return IndexedMatrix.from_rows(
            output_indices,
            [self._row(index, *generator_lookup[index.gate_name]) for index in output_indices],
        )

    def _row(
        self,
        fidelity_index: FidelityIndex,
        generator_indices: list[GeneratorIndex],
        generators: QubitSparsePauliList | None,
    ) -> IndexedVector[GeneratorIndex]:
        """The row of the log-fidelity parameterization matrix for a fidelity index.

        Returns an :class:`~.IndexedVector` whose labels correspond to the generators of the gate
        noise model that anti-commute with the relevant Pauli operator
        (``fidelity_index.transition[0]`` if the noise model is before the gate, and
        ``fidelity_index.transition[1]`` if the noise model is after the gate). The data values of
        the indexed vector are all :math:`2`.

        Args:
            fidelity_index: The fidelity index labelling the requested row.
            generator_indices: The generator index objects for the fidelity index's gate, in order.
            generators: The gate's generators, or ``None`` if the gate has no generators.
        """
        if generators is None:
            return IndexedVector[GeneratorIndex]()

        # retrieve the relevant Pauli
        pauli: QubitSparsePauli = (
            fidelity_index.transition[0]
            if self.noise_site[fidelity_index.gate_name] == "before"
            else fidelity_index.transition[1]
        )
        commuting = generators.commutes(pauli.to_qubit_sparse_pauli_list()).ravel()

        return IndexedVector[GeneratorIndex](
            {generator_indices[i]: 2.0 for i in np.nonzero(~commuting)[0]}
        )

    @staticmethod
    def k_partition_local(
        gate_set: GateSet,
        k: int = 2,
        gate_k: dict[str, int] | None = None,
        qubit_partitions: dict[str, list[set[int]]] | None = None,
        local_paulis: dict[str, list[QubitSparsePauliList]] | None = None,
        noise_site: dict[str, Literal["before"] | Literal["after"]] | None = None,
    ) -> Self:
        r"""Construct a k-local model according to qubit partitions.

        Note that the default partition for this method (described below) results in differing
        default behaviour for this method as compared to :meth:`.PauliLindbladModel.k_local`.

        This method defines k-locality in terms of partitions of a set of :math:`n`-qubits,
        generalizing the usual notion beyond single qubits. A partition is a collection of disjoint
        sets of qubit indices that covers :math:`\{0, ..., n - 1\}`. Two qubit indices :math:`i, j`
        are neighbours if either edge :math:`(i, j)` or :math:`(j, i)` is in the coupling map, which
        is drawn from ``gate_set`` or defaults to the complete coupling map. Given a partition
        :math:`P`, two distinct qubit index sets :math:`B_0, B_1 \in P` are neighbours if there
        exists indices :math:`i \in B_0` and :math:`j \in B_1` that are neighbours. Similarly,
        distinct :math:`B_0, B_1 \in P` are next-nearest neighbours if there exists a distinct
        :math:`B_2 \in P` which is neighbours with both :math:`B_0` and :math:`B_1`, and so on.

        For a given gate, the :math:`k`-local model is built recursively starting with
        :math:`1`-local terms defined in the argument ``local_paulis``, with :math:`1`-local terms
        for subsets of size ``m`` given by ``local_paulis[m]``. :math:`2`-local terms are built via
        tensor product of :math:`1`-local terms between pairs of neighbouring subsets. Generally,
        :math:`k`-local terms are built from the tensor product of :math:`(k - 1)`-local terms and
        :math:`1`-local terms on sets of :math:`k` connected sets.

        If no ``qubit_partition`` is supplied, the partition of singletons is assumed. If no
        ``local_paulis`` is supplied, the set of all possible Paulis on the given number of qubits
        is assumed.

        Args:
            gate_set: The gate set being modelled. Must contain only Clifford, pure preparation, and
                pure measurement layers. To be converted to a :class:`ModelGateSet`. The coupling
                map is drawn from ``gate_set.model_gate_set.coupling_map``, or if it is ``None``,
                defaults to the complete coupling map.
            k: The default degree of locality of the model. Applies to all gates not specified
                in ``gate_k``. Defaults to ``2``.
            gate_k: A dictionary mapping gate names to per-gate locality values that override
                ``k`` for the specified gates.
            qubit_partitions: A dictionary indicating a qubit partition for each gate. Any
                unspecified partitions will be populated with a default in which qubits are
                grouped together if they are connected by unitary gate operations.
            local_paulis: A dictionary indicating the 1-local Paulis to use each qubit partition for
                each gate. I.e. ``len(local_paulis[gate_name])`` must equal the maximum partition
                size in ``qubit_partitions[gate_name]``, and
                ``local_paulis[gate_name][k].num_qubits`` must equal ``k``. For Clifford gates,
                ``local_paulis[gate_name][k]`` defaults to all possible non-identity Paulis on ``k``
                qubits, and for measurement and preparation, it defaults to all Paulis consisting of
                :math:`\{I, X\}` on ``k`` qubits.
            noise_site: Dictionary indicating whether to model gate noise as ``"before"``
                or ``"after"`` the gate.

        Returns:
            A new :class:`~.PauliLindbladModel` instance.

        Raises:
            ValueError: If any ``k`` value exceeds ``len(gate_set.qubit_subset)``.
            ValueError: If ``gate_k`` contains names not in the gate set.
            ValueError: Name not in ``gate_set`` is used in any other dictionary.
            ValueError: Any partition is ill-formed.
            ValueError: ``local_paulis`` does not satisfy the assumed form.
        """

        gate_set = gate_set.model_gate_set

        # validate k
        if k > len(gate_set.qubit_subset):
            raise ValueError(
                f"k:`{k}` must be less than or equal to the number of qubits: "
                f"`{len(gate_set.qubit_subset)}`."
            )

        # validate gate_k
        gate_k = gate_k or {}

        if not gate_k.keys() <= gate_set.keys():
            raise ValueError(
                f"gate_k contains gates not in gate_set: {set(gate_k) - set(gate_set)}"
            )

        for name, k_val in gate_k.items():
            if k_val > len(gate_set.qubit_subset):
                raise ValueError(
                    f"k:`{k_val}` for gate '{name}' must be less than or equal to the number of "
                    f"qubits: `{len(gate_set.qubit_subset)}`."
                )

        # default coupling map
        coupling_map = gate_set.coupling_map or CouplingMap.from_full(gate_set.num_qubits)

        # validate and construct partitions
        qubit_partitions = qubit_partitions or dict()

        if not qubit_partitions.keys() <= gate_set.keys():
            raise ValueError(
                f"Gates {set(qubit_partitions) - set(gate_set)} in qubit_partitions not in "
                "gate_set."
            )

        for name, gate in gate_set.items():
            if name in qubit_partitions:
                # validate partition
                union = set()
                count = 0
                for s in qubit_partitions[name]:
                    union = union.union(s)
                    count += len(s)

                if count != len(union):
                    raise ValueError(f"Partition for gate {name} contains duplicates.")

                if not union == set(gate.qubit_idxs):
                    raise ValueError(
                        f"Union of qubit partition for gate {name} is not set(gate.qubit_idxs)."
                    )
            else:
                current_partition = [{idx} for idx in gate.qubit_idxs]

                for qubit_idxs, _ in gate.cliffords:
                    if len(qubit_idxs) == 1:
                        continue

                    qubit_idxs = set(qubit_idxs)
                    next_partition = []
                    clifford_set = set()
                    for subset in current_partition:
                        if subset.intersection(qubit_idxs):
                            clifford_set = clifford_set.union(subset)
                        else:
                            next_partition.append(subset)
                    next_partition.append(clifford_set)
                    current_partition = next_partition
                qubit_partitions[name] = current_partition

        # construct default local paulis
        local_paulis = local_paulis or dict()

        if not local_paulis.keys() <= gate_set.keys():
            raise ValueError(
                f"Gates {set(local_paulis) - set(gate_set)} in local_paulis not in gate_set."
            )

        # need prep and meas names for local_paulis defaults
        prep_names, meas_names = _validate_gate_set_form(gate_set)

        for name, gate in gate_set.items():
            max_local_size = max(map(len, qubit_partitions[name]))
            if name in local_paulis:
                # validate there are enough local paulis specified
                if len(local_paulis[name]) < max_local_size:
                    raise ValueError(f"len(local_paulis[{name}]) less than largest partition size.")

                # validate the lists of Paulis are on the right number of qubits
                for idx, single_local_paulis in enumerate(local_paulis[name]):
                    if single_local_paulis.num_qubits != idx + 1:
                        raise ValueError(f"local_paulis[{name}][{idx}].num_qubits != {idx + 1}")
            else:
                gate_local_paulis = []
                if name in prep_names + meas_names:
                    pauli_strings = ["I", "X"]
                else:
                    pauli_strings = ["I", "Z", "X", "Y"]

                for local_size in range(1, max_local_size + 1):
                    labels = ["".join(p) for p in product(pauli_strings, repeat=local_size)][1:]
                    gate_local_paulis.append(QubitSparsePauliList.from_list(labels))
                local_paulis[name] = gate_local_paulis

        generators = dict()
        for name in gate_set:
            generators[name] = _k_local_paulis(
                k=gate_k.get(name, k),
                num_qubits=gate_set.num_qubits,
                coupling_map=coupling_map,
                qubit_partition=qubit_partitions[name],
                local_paulis=local_paulis[name],
            )

        return PauliLindbladModel(
            gate_set=gate_set,
            generators=generators,
            noise_site=noise_site,
        )

    @staticmethod
    def k_local(
        gate_set: GateSet,
        k: int = 2,
        gate_k: dict[str, int] | None = None,
        paulis: dict[str, QubitSparsePauliList] | None = None,
        noise_site: dict[str, Literal["before"] | Literal["after"]] | None = None,
    ) -> Self:
        r"""Construct a k-local model.

        This is equivalent to calling :meth:`.PauliLindbladModel.k_partition_local` with each
        partition specified as partition of singletons for each gate.

        Args:
            gate_set: The gate set being modelled. Must contain only Clifford, pure preparation, and
                pure measurement layers. To be converted to a :class:`ModelGateSet`.
            k: The default degree of locality of the model. Applies to all gates not specified
                in ``gate_k``. Defaults to ``2``.
            gate_k: A dictionary mapping gate names to per-gate locality values that override
                ``k`` for the specified gates.
            paulis: A dictionary indicating the single-qubit Paulis to use in the k-local
                model for each gate. For Clifford gates, defaults to all single qubit Paulis, and
                for measurement and preparation, defaults to :math:`\{I, X\}`.
            noise_site: Dictionary indicating whether to model gate noise as ``"before"``
                or ``"after"`` the gate.

        Returns:
            A new :class:`~.PauliFidelityModel` instance.
        """

        return PauliLindbladModel.k_partition_local(
            gate_set=gate_set,
            k=k,
            gate_k=gate_k,
            qubit_partitions={
                name: [{x} for x in gate.qubit_idxs] for name, gate in gate_set.items()
            },
            local_paulis=None if paulis is None else {name: [p] for name, p in paulis.items()},
            noise_site=noise_site,
        )

    def to_pauli_lindblad_maps(
        self,
        model_data: ModelData,
        include_spam: bool = False,
        symmetrize_spam: bool = False,
    ) -> dict[str, PauliLindbladMap]:
        r"""Return a dictionary of :class:`PauliLindbladMap` for each gate in the model.

        Args:
            model_data: The fitted model parameters.
            include_spam: Whether to include SPAM gates in the output.
            symmetrize_spam: Whether to symmetrize the preparation and measurement noise maps into
                fully-specified generalized depolarizing maps. Has no effect unless ``include_spam``
                is ``True``. Noise for SPAM layers is restricted to be specified with only I and X
                Paulis. If ``False``, this method returns the SPAM maps with the generators as they
                are specified: with only I and X Paulis. If ``True``, the SPAM noise maps are
                alternatively returned as generalized depolarizing maps; meaning all generators
                that have the same pattern necessarily have the same rate. The generalized
                depolarizing map is chosen such that: (1) It has the same fidelities on Z-only
                operators as the original X-only map, and (2) It does not add any new maximal
                patterns. Note that the number of generators may increase: the mapping requires that
                the qubit subsets specified by the generators is closed under taking subsets, which
                will not typically be true for k-local maps. Lastly, note that this may introduce
                negative rates that were not there before; in which case a warning is raised.

        Returns:
            A dictionary from gate names to corresponding noise maps.

        Raises:
            ValueError: If ``model_data`` does not contain the correct parameters.
        """

        noise_maps = {}

        # rates of the gates to symmetrize, gathered here and converted once the rates of a whole
        # gate are known. Stays empty unless symmetrize_spam and include_spam are both set.
        rates_to_symmetrize = {}

        for generator_index, rate in zip(
            model_data.dataset["parameter_index"].data, model_data.dataset["parameter_values"].data
        ):
            if (gate_name := generator_index.gate_name) not in self.gate_set:
                raise ValueError(f"Encountered generator for {gate_name} not present in gate set.")

            is_spam = gate_name in self._meas_names or gate_name in self._prep_names
            if is_spam and not include_spam:
                continue

            if is_spam and symmetrize_spam:
                # generators of SPAM gates are X-only, so each is determined by its support
                support = tuple(int(idx) for idx in generator_index.generator.indices)
                rates_to_symmetrize.setdefault(gate_name, {})[support] = float(rate)
                # claim the key now, so that the returned gates stay in order of first appearance
                noise_maps.setdefault(gate_name, [])
            else:
                noise_maps.setdefault(gate_name, []).append(
                    PauliLindbladMap.GeneratorTerm(rate, generator_index.generator)
                )

        for gate_name, rates in rates_to_symmetrize.items():
            symmetrized = _symmetrized_spam_rates(rates)
            # raise a warning if symmetrization introduces rate negativity
            if min(rates.values(), default=0.0) >= 0.0:
                _warn_on_negative_rates(gate_name, symmetrized)
            noise_maps[gate_name] = _generalized_depolarizing_terms(
                symmetrized, self.gate_set.num_qubits
            )

        return {
            gate_name: PauliLindbladMap.from_terms(generators)
            for gate_name, generators in noise_maps.items()
        }


def _validate_gate_set_form(gate_set: ModelGateSet) -> tuple[list[str], list[str]]:
    """Validate that the gate set contains only unitary, and pure measurement and preparation gates.

    Also, returns a list of preparation gate names and measurement gate names.

    Args:
        gate_set: The gate set to validate.

    Returns:
        A pair of lists indicating preparation and measurement gate names.

    Raises:
        ValueError: if any of the gate set assumptions of :class:`PauliLindbladModel` are violated.
    """
    preparation_names = []
    measurement_names = []

    for name, gate in gate_set.items():
        if len(gate.meas_idxs) > 0:
            if len(gate.prep_idxs) > 0:
                raise ValueError("Gate set contains gate with both measured and prepared qubits.")
            if len(gate.cliffords) > 0:
                raise ValueError("Gate with measurement contains non-trivial unitary part.")
            if len(gate.meas_idxs) < len(gate_set.qubit_subset):
                raise ValueError(
                    "Gate with measurement does not measure all qubits acted on by the gate set."
                )
            measurement_names.append(name)
        elif len(gate.prep_idxs) > 0:
            if len(gate.cliffords) > 0:
                raise ValueError("Gate with preparation contains non-trivial unitary part.")

            if len(gate.prep_idxs) < len(gate_set.qubit_subset):
                raise ValueError(
                    "Gate with preparation does not prepare all qubits acted on by the gate set."
                )
            preparation_names.append(name)

    if not measurement_names:
        raise ValueError("Gate set does not contain a pure measurement gate.")

    if not preparation_names:
        raise ValueError("Gate set does not contain a pure preparation gate.")

    return preparation_names, measurement_names


def _validate_generators(
    gate_set: ModelGateSet,
    generators: dict[str, QubitSparsePauliList],
    prep_names: list[str],
    meas_names: list[str],
):
    """Validate the generators for the gate set.

    Ensures the generators act only on the qubits the gate acts on, that each generator is unique,
    and that preparation and measurement generators are built from :math:`I` and :math:`X` only.

    Args:
        gate_set: The gate set.
        generators: The generators to validate.
        prep_names: The names of the pure preparation gates.
        meas_names: The names of the pure measurement gates.

    Raises:
        ValueError: If the names of the gates don't match the keys of the dictionary, any generator
            acts outside the bounds of the gate, any generator list does not have unqiue elements,
            or a preparation or measurement generator contains a Pauli other than :math:`X`.
    """

    if set(gate_set) != set(generators):
        raise ValueError("The gate names of gate_set must match the keys of generators.")

    for name, gate in gate_set.items():
        is_spam = name in prep_names or name in meas_names

        unique_pauli_list = []
        for pauli in generators[name]:
            if pauli in unique_pauli_list:
                raise ValueError("Generators for a given gate must be unique.")
            if not set(pauli.indices).issubset(gate.qubit_idxs):
                raise ValueError("Generators must act only on the qubits the gate acts on.")
            if is_spam and not (pauli.paulis == QubitSparsePauli.Pauli.X).all():
                raise ValueError(
                    f"Generator {pauli} of preparation or measurement gate {name!r} contains a "
                    "Pauli other than X; preparation and measurement are assumed to occur in the "
                    "Z basis, so their generators must be built from I and X only."
                )

            unique_pauli_list.append(pauli)


def _validate_and_complete_noise_site_dict(
    gate_set: ModelGateSet,
    noise_site: dict[str, str] | None,
    prep_names: list[str],
    meas_names: list[str],
) -> dict[str, bool]:
    """Complete the noise model before gate dict, raising any errors if assumptions are violated.

    Args:
        gate_set: The gate set.
        noise_site: Whether the noise models appear before or after the gates.
        prep_names: The names of the pure preparation gates.
        meas_names: The names of the pure measurement gates.

    Returns:
        A completely specified and valid ``noise_site`` dictionary.

    Raises:
        ValueError: If any of the assumptions of :class:`PauliLindbladModel` are violated.
    """

    noise_site = noise_site.copy() if noise_site else dict()

    for name in gate_set:
        if name in prep_names:
            if name in noise_site:
                if noise_site[name] != "after":
                    raise ValueError("Preparation noise models must occur after the gate.")
            else:
                noise_site[name] = "after"
        elif name in meas_names:
            if name in noise_site:
                if noise_site[name] != "before":
                    raise ValueError("Measurement noise models must occur before the gate.")
            else:
                noise_site[name] = "before"
        else:
            if name in noise_site:
                if noise_site[name] not in ["before", "after"]:
                    raise ValueError("Noise site can only take values 'before' or 'after'.")
            else:
                noise_site[name] = "before"

    return noise_site


def _index_sets_are_adjacent(set0: set[int], set1: set[int], coupling_map: CouplingMap) -> bool:
    """Given two sets of qubit indices, determine whether any in the first set are coupled to any in
    the second.
    """
    distance_matrix = coupling_map.distance_matrix
    dist_1 = np.abs(distance_matrix - 1.0)
    return any(
        dist_1[idx0, idx1] > -0.1 and dist_1[idx0, idx1] < 0.1 for idx0, idx1 in product(set0, set1)
    )


def _k_local_paulis(
    k: int,
    num_qubits: int,
    coupling_map: CouplingMap,
    qubit_partition: list[set[int]],
    local_paulis: list[QubitSparsePauliList],
) -> QubitSparsePauliList:
    """Build a set of k-local operators as called by :meth:`PauliLindbladMap.k_partition_local`.

    This function assumes the inputs are well-formed.
    """

    # initialize with 1-local operators
    paulis = []
    for subset in qubit_partition:
        index_list = sorted(subset)
        paulis.append(local_paulis[len(subset) - 1].apply_layout(index_list, num_qubits=num_qubits))

    # recurse through successive values of k
    previous_k_local_sets_and_paulis = zip(qubit_partition, copy(paulis))

    for _ in range(k - 1):
        current_k_local_sets = []
        current_k_paulis = []

        # iterate through the sets and paulis for the previous k
        for previous_k_local_set, previous_k_paulis in previous_k_local_sets_and_paulis:
            # iterate over subsets
            for qubit_subset in qubit_partition:
                # if disjoint and adjacent
                if qubit_subset.isdisjoint(previous_k_local_set) and _index_sets_are_adjacent(
                    qubit_subset, previous_k_local_set, coupling_map=coupling_map
                ):
                    new_qubit_subset = qubit_subset.union(previous_k_local_set)
                    if new_qubit_subset in current_k_local_sets:
                        continue
                    current_k_local_sets.append(new_qubit_subset)

                    index_list = sorted(qubit_subset)
                    new_paulis = []
                    for pauli0, pauli1 in product(
                        local_paulis[len(qubit_subset) - 1].apply_layout(
                            index_list, num_qubits=num_qubits
                        ),
                        previous_k_paulis,
                    ):
                        new_paulis.append(pauli0 @ pauli1)
                    current_k_paulis.append(
                        QubitSparsePauliList.from_qubit_sparse_paulis(new_paulis)
                        if new_paulis
                        else QubitSparsePauliList.empty(num_qubits)
                    )

        paulis.extend(current_k_paulis)
        previous_k_local_sets_and_paulis = zip(current_k_local_sets, current_k_paulis)

    paulis = list(chain.from_iterable(paulis))
    return (
        QubitSparsePauliList.from_qubit_sparse_paulis(paulis)
        if paulis
        else QubitSparsePauliList.empty(num_qubits)
    )


def _symmetrized_spam_rates(rates: dict[tuple[int, ...], float]) -> dict[tuple[int, ...], float]:
    """Convert X-only rates into generalized depolarizing rates with the same Z-Pauli fidelities.

    Both parameterizations are indexed by the qubit subset a generator is supported on. Requiring
    equal fidelities on every ``Z^S`` determines the result uniquely, as

        out[R] = sum over R' containing R of (-1) ** (|R'| - |R|) * rates[R'] / 2 ** |R'|,

    which is computed here by distributing each input rate over the subsets of its own support. The
    result is therefore supported on the subsets of the supports of ``rates``, so no new qubit is
    introduced and no support grows.

    Args:
        rates: A mapping from qubit subsets to X-only rates. Subsets must be sorted tuples.

    Returns:
        A mapping from qubit subsets to generalized depolarizing rates.
    """

    symmetrized: dict[tuple[int, ...], float] = {}

    for support, rate in rates.items():
        weight = len(support)
        for size in range(1, weight + 1):
            sign = -1.0 if (weight - size) % 2 else 1.0
            contribution = sign * rate / 2**weight
            for subset in combinations(support, size):
                symmetrized[subset] = symmetrized.get(subset, 0.0) + contribution

    return symmetrized


def _generalized_depolarizing_terms(
    rates: dict[tuple[int, ...], float], num_qubits: int
) -> list[PauliLindbladMap.GeneratorTerm]:
    """Return the generator terms of a generalized depolarizing map.

    Every Pauli supported on a given qubit subset carries that subset's rate.

    Args:
        rates: A mapping from qubit subsets to rates.
        num_qubits: The number of qubits the generators act on.

    Returns:
        A list of generator terms.
    """

    terms = []
    for support, rate in rates.items():
        for paulis in product("XYZ", repeat=len(support)):
            terms.append(
                PauliLindbladMap.GeneratorTerm(
                    rate,
                    QubitSparsePauli.from_sparse_label(
                        ("".join(paulis), list(support)), num_qubits=num_qubits
                    ),
                )
            )
    return terms


def _warn_on_negative_rates(gate_name: str, rates: dict[tuple[int, ...], float]):
    """Warn if any rate is negative, listing at most ten of the offending qubit subsets.

    Args:
        gate_name: The name of the gate the rates belong to.
        rates: A mapping from qubit subsets to rates.
    """

    negative = sorted(support for support, rate in rates.items() if rate < 0.0)
    if not negative:
        return

    shown = ", ".join(str(support) for support in negative[:10])
    if len(negative) > 10:
        shown += f", and {len(negative) - 10} more"

    warnings.warn(
        f"Symmetrizing the noise of gate {gate_name!r} produced {len(negative)} negative rate(s), "
        "so the resulting map is not completely positive and cannot be sampled from; its action on "
        f"Z-type Paulis is unaffected. Qubit subsets carrying a negative rate: {shown}.",
        stacklevel=2,
    )
