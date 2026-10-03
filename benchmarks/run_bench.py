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

"""Run the experiment-building benchmark suite.

Examples::

    python benchmarks/run_bench.py --self-check
    python benchmarks/run_bench.py --list
    python benchmarks/run_bench.py hex32 grid32
    python benchmarks/run_bench.py --all --json results.json

See ``benchmarks/README.md`` for what the cases are and how to read the output.
"""

import argparse
import cProfile
import json
import pstats
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qnl_bench import SUITE, case_by_name  # noqa: E402
from qnl_bench.build import BuildResult, timed_build  # noqa: E402
from qnl_bench.cases import BASIS_GATES, build_gate_set  # noqa: E402
from qnl_bench.lattices import heavy_hex_lattice  # noqa: E402

#: How many individual instrumented calls to list per report.
LARGEST_CALLS = 8

#: Default number of rows in a ``--profile`` table.
PROFILE_ROWS = 30


def profiled(build: Callable[[], BuildResult]) -> tuple[BuildResult, pstats.Stats]:
    """Run a build under :mod:`cProfile`.

    Args:
        build: A callable that runs and returns the build.

    Returns:
        The build result and its profile statistics.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        result = build()
    finally:
        profiler.disable()
    return result, pstats.Stats(profiler)


def self_check() -> int:
    """Verify the synthetic topologies against real device data and internal invariants.

    Returns:
        A process exit status: ``0`` if every check passed.
    """
    failures = []

    try:
        from qiskit_ibm_runtime.fake_provider import FakeFez

        expected = {tuple(sorted(edge)) for edge in FakeFez().coupling_map.get_edges()}
        _, edges = heavy_hex_lattice(8, 16)
        generated = {tuple(sorted(edge)) for edge in edges}
        if generated == expected:
            print(f"heavy_hex_lattice(8, 16) == FakeFez coupling map ({len(expected)} edges)")
        else:
            failures.append(
                f"heavy_hex(8, 16) differs from FakeFez: "
                f"{len(generated - expected)} extra, {len(expected - generated)} missing"
            )
    except ImportError:
        print("skipping FakeFez comparison: qiskit_ibm_runtime is not installed")

    for case in SUITE:
        gate_set, topology = build_gate_set(case)
        num_qubits = len(topology.coupling_map.physical_qubits)
        if num_qubits != case.num_qubits:
            failures.append(f"{case.name}: got {num_qubits} qubits, wanted {case.num_qubits}")
        seen: set[int] = set()
        for layer in topology.layers:
            qubits = [qubit for pair in layer for qubit in pair]
            if len(set(qubits)) != len(qubits):
                failures.append(f"{case.name}: a layer repeats a qubit")
            seen.update(qubits)
        if len(seen) != case.num_qubits:
            failures.append(f"{case.name}: {case.num_qubits - len(seen)} qubits are in no layer")
        if set(gate_set.model_gate_set) != {*topology.gate_names, "P", "M"}:
            failures.append(f"{case.name}: unexpected model gate names")
        print(
            f"{case.name:>8}: {case.num_qubits:>4}q  {topology.num_edges:>4} edges  "
            f"{len(topology.layers)} layers  "
            f"{topology.num_edges / case.num_qubits:.2f} edges/qubit"
        )

    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


def _fmt(seconds: float) -> str:
    """Format a duration for the report."""
    return f"{seconds:7.3f}s"


def report(result: BuildResult) -> None:
    """Print the stage and component breakdown of one build.

    Args:
        result: The build to report on.
    """
    case = result.case
    rows, cols = result.design_matrix_shape
    layer_generators = {
        name: count for name, count in result.generators.items() if name not in ("P", "M")
    }
    per_layer = next(iter(layer_generators.values()), 0)

    print()
    print("=" * 78)
    print(f"{case}  --  {_fmt(result.seconds)} total, {result.peak_rss_mb:.0f} MiB peak RSS")
    print("=" * 78)
    print(
        f"  topology     : {result.num_edges} edges "
        f"({result.num_edges / case.num_qubits:.2f}/qubit), {result.num_layers} layers"
    )
    print(
        f"  generators   : {result.total_generators} total, "
        f"{per_layer} per layer gate"
        + ("" if len(set(layer_generators.values())) <= 1 else " (varies)")
    )
    print(f"  design matrix: {rows} x {cols}, density {result.design_matrix_density:.2e}")
    print(
        f"  output       : {result.num_paths} paths, {result.num_sequences} sequences, "
        f"{result.num_bound_sequences} bound"
    )
    if result.rank is not None:
        print(f"  rank         : {result.rank}")

    print()
    print("  stages" + " " * 24 + "calls      seconds   share")
    groups = result.timeline.grouped()
    total = result.timeline.total
    for group in sorted(groups.values(), key=lambda g: -g.seconds):
        share = group.seconds / total if total else 0.0
        print(f"  {group.label:<28} {group.calls:>5}  {_fmt(group.seconds)}  {share:6.1%}")
    print(f"  {'TOTAL (stages)':<28} {'':>5}  {_fmt(total)}")

    print()
    print("  components" + " " * 20 + "calls    self (s)   gross (s)")
    for record in sorted(result.components.values(), key=lambda r: -r.seconds):
        print(
            f"  {record.label:<28} {record.calls:>5}  {_fmt(record.seconds)}  "
            f"{_fmt(record.gross_seconds)}"
        )
    instrumented = sum(record.seconds for record in result.components.values())
    print(f"  {'TOTAL (instrumented self)':<28} {'':>5}  {_fmt(instrumented)}")
    print(f"  {'unattributed':<28} {'':>5}  {_fmt(result.seconds - instrumented)}")

    print()
    print("  most expensive individual calls")
    biggest = sorted(
        (
            (seconds, record.label, description)
            for record in result.components.values()
            for seconds, description in record.call_log
        ),
        reverse=True,
    )
    for seconds, label, description in biggest[:LARGEST_CALLS]:
        print(f"    {_fmt(seconds)}  {label:<42} {description}")

    if result.fingerprint:
        print()
        print("  fingerprint")
        for name, digest in sorted(result.fingerprint.items()):
            print(f"    {name:<16} {digest}")


def _serializable(result: BuildResult) -> dict:
    """Convert a build result to plain JSON-compatible data."""
    data = asdict(result)
    data["case"] = asdict(result.case)
    data["timeline"] = {
        "entries": result.timeline.entries,
        "grouped": {
            label: {"seconds": group.seconds, "calls": group.calls}
            for label, group in result.timeline.grouped().items()
        },
        "total": result.timeline.total,
    }
    data["components"] = {
        label: {
            "calls": record.calls,
            "seconds": record.seconds,
            "gross_seconds": record.gross_seconds,
            "call_log": record.call_log,
        }
        for label, record in result.components.items()
    }
    data["design_matrix_density"] = result.design_matrix_density
    data["total_generators"] = result.total_generators
    return data


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested cases.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", help="Case names to run; see --list.")
    parser.add_argument("--all", action="store_true", help="Run every case in the suite.")
    parser.add_argument("--list", action="store_true", help="List the suite and exit.")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Validate the synthetic topologies and exit without building anything.",
    )
    parser.add_argument(
        "--rank",
        action="store_true",
        help="Also compute the design matrix rank. Roughly doubles the run time.",
    )
    parser.add_argument("--no-fingerprint", action="store_true", help="Skip output digests.")
    parser.add_argument("--json", type=Path, help="Write full results to this JSON file.")
    parser.add_argument("--quiet", action="store_true", help="Do not print phase progress.")
    parser.add_argument(
        "--profile",
        nargs="?",
        type=int,
        const=PROFILE_ROWS,
        metavar="N",
        help="Also run under cProfile and print the top N functions by self time. Function-call "
        "overhead inflates every number in such a run, so use it to find hot Python and not to "
        "quote timings.",
    )
    parser.add_argument("--pstats", type=Path, help="With --profile, dump raw pstats here.")
    args = parser.parse_args(argv)

    if args.list:
        print(f"basis gates: {', '.join(BASIS_GATES)}")
        for case in SUITE:
            print(f"  {case}")
        return 0

    if args.self_check:
        return self_check()

    names = list(args.cases)
    if args.all:
        names = [case.name for case in SUITE]
    if not names:
        parser.error("give at least one case name, or --all, or --list.")

    progress = None if args.quiet else lambda message: print(f"  ... {message}", flush=True)
    results = []
    for name in names:

        def build(name=name) -> BuildResult:
            return timed_build(
                case_by_name(name),
                compute_rank=args.rank,
                fingerprint=not args.no_fingerprint,
                progress=progress,
            )

        if args.profile:
            result, stats = profiled(build)
            report(result)
            print()
            print(f"  cProfile: top {args.profile} by self time (inflated by profiling overhead)")
            stats.sort_stats("tottime").print_stats(args.profile)
            if args.pstats:
                stats.dump_stats(str(args.pstats))
                print(f"wrote {args.pstats}")
        else:
            result = build()
            report(result)
        results.append(result)

    if len(results) > 1:
        print()
        print("=" * 78)
        print(f"  {'case':<10} {'qubits':>7} {'gens':>8} {'paths':>7} {'seqs':>6} {'seconds':>9}")
        for result in results:
            print(
                f"  {result.case.name:<10} {result.case.num_qubits:>7} "
                f"{result.total_generators:>8} {result.num_paths:>7} "
                f"{result.num_sequences:>6} {result.seconds:>9.3f}"
            )

    if args.json:
        args.json.write_text(json.dumps([_serializable(r) for r in results], indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
