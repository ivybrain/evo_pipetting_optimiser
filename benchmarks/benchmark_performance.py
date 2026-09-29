"""
Claude-generated benchmarking script
Benchmark the wall-clock time of AutoWorklist.commit() (i.e. the full auto_transfer ->
optimise -> write pipeline) across a range of operation counts.

The README's "Calculation Time" section quotes fixed numbers that go stale as the optimiser changes.
Run this script to regenerate current
numbers:

    python benchmarks/benchmark_performance.py
    python benchmarks/benchmark_performance.py --sizes 50 100 200 400 800 --csv results.csv
    python benchmarks/benchmark_performance.py --wash-scheme None --max-concurrent-ops 200

Each size is run in a fresh AutoWorklist/labware so results aren't affected by leftover
volume/composition/wash state from a previous size. Scenario: a trough (unlimited,
dependency-free source -- repeated aspirates from it don't serialise the plan) dispensing
into a single large destination plate sized to hold exactly `n_ops` independent wells, so
op count is controlled precisely without any operations depending on each other.
"""
import argparse
import csv
import sys
import time
from math import ceil

from evo_pipetting_optimiser import AdvancedLabware, AdvancedTrough, AutoWorklist


def build_and_commit(n_ops, wash_scheme, max_concurrent_ops, tmpdir):
    rows = 8
    columns = ceil(n_ops / rows)

    dest_plate = AdvancedLabware(
        "dest",
        rows,
        columns,
        min_volume=0,
        max_volume=10_000,
        initial_volumes=0,
        location=(32, 2),
    )
    source_trough = AdvancedTrough(
        "src",
        8,
        1,
        min_volume=0,
        max_volume=1_000_000,
        initial_volumes=1_000_000,
        location=(2, 1),
    )

    worklist_path = f"{tmpdir}/bench_{n_ops}.gwl"
    with AutoWorklist(
        worklist_path,
        waste_location=(1, 2),
        cleaner_location=(1, 3),
        max_concurrent_ops=max_concurrent_ops,
    ) as wl:
        placed = 0
        for col in range(columns):
            for row in range(rows):
                if placed >= n_ops:
                    break
                wl.auto_transfer(
                    source_trough,
                    source_trough.wells[0, 0],
                    dest_plate,
                    dest_plate.wells[row, col],
                    50,
                    liquid_class="Water free dispense",
                    wash_scheme=wash_scheme,
                )
                placed += 1

        start = time.perf_counter()
        wl.commit(report=False)
        elapsed = time.perf_counter() - start

    return elapsed, wl.asp_count, wl.disp_count, wl.wash_count


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
        default=[50, 90, 180, 360, 720],
        help="Operation counts to benchmark (default mirrors the range quoted in README.md)",
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

    print(f"{'n_ops':>8}{'time(s)':>12}{'ops/sec':>12}{'asp':>8}{'disp':>8}{'wash':>8}")

    rows = []
    for n_ops in args.sizes:
        elapsed, asp, disp, wash = build_and_commit(
            n_ops, wash_scheme, args.max_concurrent_ops, args.tmpdir
        )
        ops_per_sec = n_ops / elapsed if elapsed > 0 else float("inf")
        print(f"{n_ops:>8}{elapsed:>12.3f}{ops_per_sec:>12.2f}{asp:>8}{disp:>8}{wash:>8}")
        rows.append(
            {
                "n_ops": n_ops,
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
