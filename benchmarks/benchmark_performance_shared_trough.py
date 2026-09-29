"""
Third benchmark variant, isolating one variable from the other two:

- benchmark_performance.py: uniform shape, ONE shared trough source        -> fast-ish but
  still super-linear (shared source creates one big interconnected subgraph).
- benchmark_performance_mixed.py: messy mixed shapes, but each 28-op unit gets its OWN
  dedicated trough/plates -> no shared nodes across units, decomposes into many small
  independent subgraphs, and turned out to scale *better* despite "harder" per-op shapes.

This script keeps the messy mixed-shape unit from benchmark_performance_mixed.py exactly as
is, but swaps every unit's dedicated trough for ONE trough shared across all units (only the
trough -- smp/dil plates stay per-unit and independent). This isolates whether it's the
"shared source node" structure specifically (not per-op shape complexity) that drives the
super-linear cost, by comparing directly against benchmark_performance_mixed.py at the same
op counts.

    python benchmarks/benchmark_performance_shared_trough.py
    python benchmarks/benchmark_performance_shared_trough.py --sizes 56 168 336 --csv results.csv
"""
import argparse
import csv
import sys
import time
from math import ceil

from evo_pipetting_optimiser import AdvancedLabware, AdvancedTrough, AutoWorklist

UNIT_OPS = 28


def add_unit(wl, i, trough, wash_scheme):
    smp = AdvancedLabware(
        f"smp{i}", 8, 12, min_volume=0, max_volume=10_000, initial_volumes=5_000,
        location=(40 + i, 1),
    )
    dil = AdvancedLabware(
        f"dil{i}", 8, 12, min_volume=0, max_volume=10_000, initial_volumes=0,
        location=(40 + i, 2),
    )

    # a) column -> row
    wl.auto_transfer(smp, smp.wells[:, 0], dil, dil.wells[0, :8], 50,
                      liquid_class="Water free dispense", wash_scheme=wash_scheme)
    # b) row -> column
    wl.auto_transfer(smp, smp.wells[0, :8], dil, dil.wells[:, 1], 50,
                      liquid_class="Water free dispense", wash_scheme=wash_scheme)
    # c) offset-shifted column
    wl.auto_transfer(smp, smp.wells[[0, 2, 4, 6], 2], dil, dil.wells[[1, 3, 5, 7], 2], 50,
                      liquid_class="Water free dispense", wash_scheme=wash_scheme)
    # d) trough broadcast -- SHARED trough across every unit (the one variable being tested)
    wl.auto_transfer(trough, trough.wells, dil, dil.wells[:, 3], 50,
                      liquid_class="Water free dispense", wash_scheme=wash_scheme)


def build_and_commit(n_ops, wash_scheme, max_concurrent_ops, tmpdir):
    n_units = ceil(n_ops / UNIT_OPS)
    actual_n_ops = n_units * UNIT_OPS

    with AutoWorklist(
        waste_location=(1, 2),
        cleaner_location=(1, 3),
        max_concurrent_ops=max_concurrent_ops,
    ) as wl:
        trough = AdvancedTrough(
            "trough", 8, 1, min_volume=0, max_volume=1_000_000_000, initial_volumes=1_000_000_000,
            location=(30, 3),
        )
        for i in range(n_units):
            add_unit(wl, i, trough, wash_scheme)

        assert len(wl.pending_ops) == actual_n_ops, (
            f"expected {actual_n_ops} pending ops, got {len(wl.pending_ops)} "
            "-- unit construction may have changed shape unexpectedly"
        )

        start = time.perf_counter()
        wl.commit(report=False)
        elapsed = time.perf_counter() - start

    return elapsed, actual_n_ops, wl.asp_count, wl.disp_count, wl.wash_count


def parse_wash_scheme(value):
    if value == "None":
        return None
    if value == "1":
        return 1
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[56, 90, 180, 364, 728],
        help="Target operation counts, rounded up to the nearest 28-op unit "
             "(same default range as benchmark_performance_mixed.py, for direct comparison)",
    )
    parser.add_argument(
        "--wash-scheme",
        type=str,
        default="smart",
        help="wash_scheme to pass to auto_transfer: D, smart, 1, or None (default: smart)",
    )
    parser.add_argument(
        "--max-concurrent-ops",
        type=int,
        default=None,
        help="Batch size for AutoWorklist's max_concurrent_ops (default: no batching)",
    )
    parser.add_argument(
        "--tmpdir",
        type=str,
        default="/tmp",
        help="Directory to write scratch .gwl files to (default: /tmp)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write results as CSV",
    )
    args = parser.parse_args()

    wash_scheme = parse_wash_scheme(args.wash_scheme)

    print(f"{'req':>6}{'actual':>8}{'time(s)':>12}{'ops/sec':>12}{'asp':>8}{'disp':>8}{'wash':>8}")

    rows = []
    for n_ops in args.sizes:
        elapsed, actual_n_ops, asp, disp, wash = build_and_commit(
            n_ops, wash_scheme, args.max_concurrent_ops, args.tmpdir
        )
        ops_per_sec = actual_n_ops / elapsed if elapsed > 0 else float("inf")
        print(f"{n_ops:>6}{actual_n_ops:>8}{elapsed:>12.3f}{ops_per_sec:>12.2f}"
              f"{asp:>8}{disp:>8}{wash:>8}")
        rows.append(
            {
                "requested_n_ops": n_ops,
                "actual_n_ops": actual_n_ops,
                "time_s": elapsed,
                "ops_per_sec": ops_per_sec,
                "asp_count": asp,
                "disp_count": disp,
                "wash_count": wash,
                "wash_scheme": args.wash_scheme,
                "max_concurrent_ops": args.max_concurrent_ops,
            }
        )
        sys.stdout.flush()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
    main()
