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

"""Tests for the unsupported post-selection metadata helpers of the vanilla protocol."""

import numpy as np
import pytest
import xarray as xr

from qiskit_noise_learning.data import RawData
from qiskit_noise_learning.protocols.vanilla import _fraction_kept, _success_rates
from qiskit_noise_learning.sequences import ApplyGate, InstructionSequence

QUBITS = np.array([5, 6])
NUM_RANDOMIZATIONS = 2
NUM_SHOTS = 100


def make_raw_data(
    creg_names,
    clbit_qubit_idxs,
    fragment_depths,
    *,
    repeatable_gates=None,
    empty_repeatable=(),
    data=None,
    measurement_flips=None,
    data_mask=None,
    include_mask=True,
):
    """Build single-leaf raw data with one instruction sequence per entry of ``fragment_depths``.

    Each sequence's repeatable fragment applies the corresponding gate of ``repeatable_gates``,
    which defaults to ``"CZ"`` throughout. Entries whose index appears in ``empty_repeatable`` get
    a sequence with no repeatable fragment instead, standing in for a SPAM sequence.

    Every sequence gets :data:`NUM_RANDOMIZATIONS` randomizations of :data:`NUM_SHOTS` shots, so
    ``data``, ``measurement_flips`` and ``data_mask`` are given as single arrays spanning all of
    them along the leading axis, and are split per sequence here. Each defaults to all-false.
    Pass ``include_mask=False`` to leave the mask variable off the dataset entirely.
    """
    num_bits = sum(len(idxs) for idxs in clbit_qubit_idxs.values())
    repeatable_gates = repeatable_gates or ["CZ"] * len(fragment_depths)
    sequences = [
        InstructionSequence(
            start_fragment=[ApplyGate("P")],
            repeatable_fragment=[] if idx in empty_repeatable else [ApplyGate(gate)],
            end_fragment=[ApplyGate("M")],
            fragment_depth=depth,
        )
        for idx, (depth, gate) in enumerate(zip(fragment_depths, repeatable_gates))
    ]
    num_randomizations = NUM_RANDOMIZATIONS * len(sequences)

    def per_sequence(values, shape):
        """Split an array over the leading randomization axis, one chunk per sequence."""
        if values is None:
            values = np.zeros((num_randomizations, *shape), dtype=bool)
        return [
            values[idx * NUM_RANDOMIZATIONS : (idx + 1) * NUM_RANDOMIZATIONS]
            for idx in range(len(sequences))
        ]

    times = np.array(["2026-01-01"] * NUM_RANDOMIZATIONS, dtype="datetime64[us]")
    raw_data = RawData.from_arrays(
        creg_names=creg_names,
        clbit_qubit_idxs=clbit_qubit_idxs,
        instruction_sequences=sequences,
        data=per_sequence(data, (NUM_SHOTS, num_bits)),
        measurement_flips=per_sequence(measurement_flips, (num_bits,)),
        time_lbs=[times for _ in sequences],
        time_ubs=[times for _ in sequences],
    )

    # from_arrays always writes an all-false mask, so a given one replaces it afterwards.
    if data_mask is None and include_mask:
        return raw_data

    dataset = raw_data.datatree["0"].dataset
    if not include_mask:
        dataset = dataset.drop_vars("data_mask")
    else:
        dataset = dataset.assign(data_mask=xr.DataArray(data_mask, dims=["randomization", "shot"]))
    return RawData(xr.DataTree.from_dict({"0": dataset}))


def test_fraction_kept_averages_the_shots_kept_at_each_depth():
    mask = np.zeros((2 * NUM_RANDOMIZATIONS, NUM_SHOTS), dtype=bool)
    mask[:NUM_RANDOMIZATIONS, 80:] = True
    mask[NUM_RANDOMIZATIONS:, 25:] = True
    raw_data = make_raw_data(["meas"], {"meas": QUBITS}, [2, 4], data_mask=mask)

    assert _fraction_kept(raw_data, ["CZ"]) == [pytest.approx({2: 0.8, 4: 0.25})]


def test_fraction_kept_reports_one_entry_per_instruction():
    """Only the sequences applying a gate report against the instruction that became it."""
    mask = np.zeros((2 * NUM_RANDOMIZATIONS, NUM_SHOTS), dtype=bool)
    mask[:NUM_RANDOMIZATIONS, 80:] = True
    mask[NUM_RANDOMIZATIONS:, 25:] = True
    raw_data = make_raw_data(
        ["meas"], {"meas": QUBITS}, [2, 4], repeatable_gates=["CZ", "CX"], data_mask=mask
    )

    assert _fraction_kept(raw_data, ["CZ", "CX"]) == [
        pytest.approx({2: 0.8}),
        pytest.approx({4: 0.25}),
    ]


def test_fraction_kept_shares_spam_across_every_instruction():
    """SPAM sequences probe no gate in particular, so they land in every instruction's entry."""
    mask = np.zeros((2 * NUM_RANDOMIZATIONS, NUM_SHOTS), dtype=bool)
    mask[:NUM_RANDOMIZATIONS, 70:] = True
    mask[NUM_RANDOMIZATIONS:, 90:] = True
    raw_data = make_raw_data(
        ["meas"], {"meas": QUBITS}, [2, 0], empty_repeatable=[1], data_mask=mask
    )

    assert _fraction_kept(raw_data, ["CZ", "M"]) == [
        pytest.approx({0: 0.9, 2: 0.7}),
        pytest.approx({0: 0.9}),
    ]


def test_fraction_kept_merges_spam_into_the_depth_zero_entry():
    """SPAM sequences are bound at depth zero, so they average in with the depth-zero sequences."""
    mask = np.zeros((2 * NUM_RANDOMIZATIONS, NUM_SHOTS), dtype=bool)
    mask[:NUM_RANDOMIZATIONS, 70:] = True
    mask[NUM_RANDOMIZATIONS:, 90:] = True
    raw_data = make_raw_data(
        ["meas"], {"meas": QUBITS}, [0, 0], empty_repeatable=[1], data_mask=mask
    )

    assert _fraction_kept(raw_data, ["CZ"]) == [pytest.approx({0: 0.8})]


def test_fraction_kept_is_empty_for_an_instruction_with_no_sequences():
    """An instruction that no sequence probes, and that no SPAM sequence is there to cover."""
    raw_data = make_raw_data(["meas"], {"meas": QUBITS}, [0, 2])

    assert _fraction_kept(raw_data, ["CZ", "CX"]) == [{0: 1.0, 2: 1.0}, {}]


def test_fraction_kept_is_one_without_a_mask():
    raw_data = make_raw_data(["meas"], {"meas": QUBITS}, [0, 2], include_mask=False)

    assert _fraction_kept(raw_data, ["CZ"]) == [{0: 1.0, 2: 1.0}]


def test_success_rates_counts_the_bits_the_pulse_flipped():
    """Bits 0, 1 are meas(q5, q6) and bits 2, 3 are its post-selection copy.

    Runs with the default ``apply_measurement_flips=False``, so the bits below are the physical
    measurement outcomes and the recorded flips do not enter the rule.
    """
    # The post-selection bit differs from the base bit on exactly the successful shots.
    bits = np.zeros((NUM_RANDOMIZATIONS, NUM_SHOTS, 4), dtype=bool)
    bits[:, :90, 2] = True
    bits[:, :40, 3] = True
    raw_data = make_raw_data(
        ["meas", "meas_ps"], {"meas": QUBITS, "meas_ps": QUBITS}, [2], data=bits
    )

    (rates,) = _success_rates(raw_data, ["CZ"])
    assert rates.keys() == {2}
    assert rates[2] == pytest.approx({5: 0.9, 6: 0.4})


def test_success_rates_report_one_entry_per_instruction():
    """Only the sequences applying a gate report against the instruction that became it."""
    bits = np.zeros((2 * NUM_RANDOMIZATIONS, NUM_SHOTS, 4), dtype=bool)
    bits[:NUM_RANDOMIZATIONS, :90, 2:] = True
    bits[NUM_RANDOMIZATIONS:, :40, 2:] = True
    raw_data = make_raw_data(
        ["meas", "meas_ps"],
        {"meas": QUBITS, "meas_ps": QUBITS},
        [2, 4],
        repeatable_gates=["CZ", "CX"],
        data=bits,
    )

    cz_rates, cx_rates = _success_rates(raw_data, ["CZ", "CX"])
    assert cz_rates.keys() == {2}
    assert cz_rates[2] == pytest.approx({5: 0.9, 6: 0.9})
    assert cx_rates.keys() == {4}
    assert cx_rates[4] == pytest.approx({5: 0.4, 6: 0.4})


def test_success_rates_share_spam_across_every_instruction():
    """SPAM sequences probe no gate in particular, so they land in every instruction's entry."""
    bits = np.zeros((2 * NUM_RANDOMIZATIONS, NUM_SHOTS, 4), dtype=bool)
    bits[:NUM_RANDOMIZATIONS, :80, 2:] = True
    bits[NUM_RANDOMIZATIONS:, :60, 2:] = True
    raw_data = make_raw_data(
        ["meas", "meas_ps"],
        {"meas": QUBITS, "meas_ps": QUBITS},
        [2, 0],
        empty_repeatable=[1],
        data=bits,
    )

    cz_rates, meas_rates = _success_rates(raw_data, ["CZ", "M"])
    assert cz_rates.keys() == {0, 2}
    assert cz_rates[0] == pytest.approx({5: 0.6, 6: 0.6})
    assert cz_rates[2] == pytest.approx({5: 0.8, 6: 0.8})
    assert meas_rates.keys() == {0}
    assert meas_rates[0] == pytest.approx({5: 0.6, 6: 0.6})


def test_success_rates_merges_spam_into_the_depth_zero_entry():
    """As for the fraction kept, SPAM sequences land in the depth-zero entry."""
    bits = np.zeros((2 * NUM_RANDOMIZATIONS, NUM_SHOTS, 4), dtype=bool)
    # The depth-zero learning sequence succeeds on 80% of shots, the SPAM sequence on 60%.
    bits[:NUM_RANDOMIZATIONS, :80, 2:] = True
    bits[NUM_RANDOMIZATIONS:, :60, 2:] = True
    raw_data = make_raw_data(
        ["meas", "meas_ps"],
        {"meas": QUBITS, "meas_ps": QUBITS},
        [0, 0],
        empty_repeatable=[1],
        data=bits,
    )

    (rates,) = _success_rates(raw_data, ["CZ"])
    assert rates.keys() == {0}
    assert rates[0] == pytest.approx({5: 0.7, 6: 0.7})


def test_success_rates_are_empty_for_an_instruction_with_no_sequences():
    """An instruction that no sequence probes, and that no SPAM sequence is there to cover."""
    raw_data = make_raw_data(["meas"], {"meas": QUBITS}, [0, 2])

    assert _success_rates(raw_data, ["CZ", "CX"]) == [
        {0: {5: 1.0, 6: 1.0}, 2: {5: 1.0, 6: 1.0}},
        {},
    ]


def test_success_rates_are_one_without_post_selection_registers():
    """Every measured qubit still reports a rate, at every depth."""
    raw_data = make_raw_data(["meas"], {"meas": QUBITS}, [0, 2])

    assert _success_rates(raw_data, ["CZ"]) == [{0: {5: 1.0, 6: 1.0}, 2: {5: 1.0, 6: 1.0}}]


def test_success_rates_are_one_for_a_qubit_with_no_post_selection_partner():
    """A register without a partner leaves its qubits unchecked, and so fully successful."""
    # Bits 0, 1 are meas(q5, q6); bit 2 is extra(q7) and bit 3 its copy.
    bits = np.zeros((NUM_RANDOMIZATIONS, NUM_SHOTS, 4), dtype=bool)
    bits[:, :40, 3] = True
    raw_data = make_raw_data(
        ["meas", "extra", "extra_ps"],
        {"meas": QUBITS, "extra": np.array([7]), "extra_ps": np.array([7])},
        [2],
        data=bits,
    )

    (rates,) = _success_rates(raw_data, ["CZ"])
    assert rates.keys() == {2}
    assert rates[2] == pytest.approx({5: 1.0, 6: 1.0, 7: 0.4})


def test_success_rates_with_flips_applied_undoes_differing_flips_in_a_pair():
    """With ``apply_measurement_flips=True``, flips that differ across a pair change the rate.

    The physical outcome of a bit is its stored value with the recorded flip undone, so a flip on
    one register of the pair and not the other inverts what the comparison sees.
    """
    # Stored bits agree everywhere, so with no correction no bit flipped at all.
    bits = np.zeros((NUM_RANDOMIZATIONS, NUM_SHOTS, 4), dtype=bool)
    # A flip on the post-selection bits only, making every physical outcome differ instead.
    flips = np.zeros((NUM_RANDOMIZATIONS, 4), dtype=bool)
    flips[:, 2:] = True
    raw_data = make_raw_data(
        ["meas", "meas_ps"],
        {"meas": QUBITS, "meas_ps": QUBITS},
        [2],
        data=bits,
        measurement_flips=flips,
    )

    assert _success_rates(raw_data, ["CZ"])[0][2] == pytest.approx({5: 0.0, 6: 0.0})
    assert _success_rates(raw_data, ["CZ"], apply_measurement_flips=True)[0][2] == pytest.approx(
        {5: 1.0, 6: 1.0}
    )


def test_success_rates_agree_either_way_on_a_shared_frame():
    """Where a pair's two registers carry equal flips, both settings give the same rates.

    Equal flips are what a twirl applied once, ahead of both measurements, produces. The flip then
    shifts both stored values alike and cancels in the comparison between them.
    """
    bits = np.zeros((NUM_RANDOMIZATIONS, NUM_SHOTS, 4), dtype=bool)
    bits[:, :90, 2] = True
    bits[:, :40, 3] = True
    per_qubit_flips = np.array([[True, False], [False, True]])
    flips = np.tile(per_qubit_flips, 2)

    cregs = (["meas", "meas_ps"], {"meas": QUBITS, "meas_ps": QUBITS}, [2])
    untwirled = make_raw_data(*cregs, data=bits)
    twirled = make_raw_data(*cregs, data=bits ^ flips[:, np.newaxis, :], measurement_flips=flips)

    assert _success_rates(twirled, ["CZ"], apply_measurement_flips=True)[0][2] == pytest.approx(
        _success_rates(untwirled, ["CZ"])[0][2]
    )
