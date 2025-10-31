import itertools
from collections import deque
from robotools import *
from ..utils import *
import numpy as np
from ..robotools_extensions import TransferOperation
from ortools.linear_solver import pywraplp


def create_conflict_graph(pending_ops):

    max_op_id = max([op.id for op in pending_ops])

    # Show which ops can have their sources grouped together
    conflicts = np.full((max_op_id + 1, max_op_id + 1), 0, dtype=np.uint8)

    for i_op in pending_ops:
        for j_op in pending_ops:
            i = i_op.id
            j = j_op.id
            # Can always include op with itself
            if i == j:
                conflicts[i, j] = 1
                continue

            # # Check liquid class is the same
            # if i_op.liquid_class != j_op.liquid_class:
            #     continue

            conflicts[i, j] = 1

            # # If can't do in 1 asp or 1 disp, then can't group
            # source_conflict = False
            # dest_conflict = False
            # # Check source is the same
            # if i_op.source != j_op.source:
            #     source_conflict = True

            # if i_op.source_pos[1] != j_op.source_pos[1]:
            #     source_conflict = True

            # # well overlap
            # if (
            #     i_op.source_pos[0] == j_op.source_pos[0]
            #     and not (i_op.source.virtual_rows or 0) >= i_op.source_pos[0]
            # ):
            #     source_conflict = True

            # if i_op.destination != j_op.destination:
            #     dest_conflict = True
            # if i_op.dest_pos[1] != j_op.dest_pos[1]:
            #     dest_conflict = True
            # if (
            #     i_op.dest_pos[0] == j_op.dest_pos[0]
            #     and not (i_op.destination.virtual_rows or 0) >= i_op.dest_pos[0]
            # ):
            #     dest_conflict = True

            # if not source_conflict:
            #     source_groups[i, j] = i_op.source_loc

            # if not dest_conflict:
            #     dest_groups[i, j] = i_op.dest_loc

    return conflicts


def ortools_grouping(
    ops,
    conflicts,
):
    ops = list(ops)
    max_bins = len(ops)
    n_tips = 8

    for i, op in enumerate(ops):
        if i != op.id:
            raise Exception("Please provide ops with id increasing from 0")

    solver = pywraplp.Solver("", pywraplp.Solver.BOP_INTEGER_PROGRAMMING)
    # solver.EnableOutput()

    solver.SetTimeLimit(30000)
    solver.EnableOutput()

    if not solver:
        raise Exception("Error creating solver")

    # Track which op is in which bin
    x = {}
    for i in range(len(ops)):
        for j in range(max_bins):
            x[(i, j)] = solver.BoolVar(f"x_{i}_{j}")

    # Track bins used
    y = {}
    for j in range(max_bins):
        y[j] = solver.BoolVar(f"y_{j}")

    possible_locations = range(ops[0].loc_counter)

    liquid_classes = range(ops[0].lc_counter)

    # source_present[l, b] is 1 if location l is present in the sources of ops in bin b
    source_present = {}
    dest_present = {}
    for l in possible_locations:
        for b in range(max_bins):
            source_present[(l, b)] = solver.BoolVar(f"source_{l}_{b}")
            dest_present[(l, b)] = solver.BoolVar(f"dest_{l}_{b}")

    # constraints to set values of source and dest present
    for l in possible_locations:
        ops_with_source = [i for i, op in enumerate(ops) if op.source_loc == l]
        ops_with_dest = [i for i, op in enumerate(ops) if op.dest_loc == l]

        for b in range(max_bins):

            solver.Add(
                sum(x[i, b] for i in ops_with_source)
                <= len(ops_with_source) * source_present[(l, b)]
            )

            solver.Add(
                sum(x[i, b] for i in ops_with_dest)
                <= len(ops_with_dest) * dest_present[(l, b)]
            )

    # bin_lc[b, lc] if ops in bin 1 are liquid class lc
    bin_lc = {}
    for lc in liquid_classes:
        for b in range(max_bins):
            bin_lc[(b, lc)] = solver.BoolVar(f"bin_lc_{b}_{lc}")

    # Each bin can only have one LC
    for b in range(max_bins):
        solver.Add(sum(bin_lc[(b, lc)] for lc in liquid_classes) == 1)

    # All ops in bin must have matching lc
    for i in range(len(ops)):
        lc = ops[i].lc_id
        for b in range(max_bins):
            # Want to add that x[i, b] => bin_lc[b, lc]
            solver.Add(x[(i, b)] <= bin_lc[(b, lc)])

    # Track tips for each op
    op_tip = {}
    for i in range(len(ops)):
        for t in range(n_tips):
            op_tip[i, t] = solver.BoolVar(f"op_tip_{i}_{t}")
        # Each op must have a tip assigned
        solver.Add(sum(op_tip[i, t] for t in range(n_tips)) == 1)

    # Each bin can only have one of each tip
    for b in range(max_bins):
        for t in range(n_tips):
            tip_bin_link_vars = []
            for i in range(len(ops)):
                link = solver.BoolVar(f"link_{i}_{b}_{t}")
                tip_bin_link_vars.append(link)

                # Link var can only be true if op i is in bin b
                solver.Add(link <= x[i, b])

                # Link var can only be true if op i has tip t
                solver.Add(link <= op_tip[i, t])

                # Link var is true if both are true
                solver.Add(link >= x[i, b] + op_tip[i, t] - 1)

            solver.Add(sum(tip_bin_link_vars) <= 1)

    # Each item must be in exactly one bin.
    for i in range(len(ops)):
        solver.Add(sum(x[i, b] for b in range(max_bins)) == 1)

    # Bin capacity 8 items
    for b in range(max_bins):
        solver.Add(sum(x[(i, b)] for i in range(len(ops))) <= y[b] * 8)

    # Enforce op dependency

    def add_op_dep(child, dep):
        for b in range(max_bins):
            # Enforce that dep is in an earlier bin than child
            # If child is in bin b, then dep must be in one of the bins for dep_bin < b
            solver.Add(
                x[child.id, b] <= sum(x[dep.id, dep_bin] for dep_bin in range(b))
            )

    for op in ops:
        if op.source_dep:
            add_op_dep(op, op.source_dep)
        if op.dest_dep:
            add_op_dep(op, op.dest_dep)

    # Item conflict constraints from graph
    for i, i_op in enumerate(ops):
        for j, j_op in enumerate(ops):
            if conflicts[i_op.id, j_op.id] == 0:
                for b in range(max_bins):

                    solver.Add(x[i, b] + x[j, b] <= 1)

    solver.Minimize(
        solver.Sum(
            source_present[(l, b)] + dest_present[(l, b)]
            for l in possible_locations
            for b in range(max_bins)
        )
    )

    status = solver.Solve()

    result_groups = []
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        num_bins = 0
        for j in range(max_bins):
            if y[j].solution_value() == 1:
                bin_items = []
                for i in range(len(ops)):
                    if x[i, j].solution_value() > 0:
                        tip = 0
                        for t in range(n_tips):
                            if op_tip[i, t].solution_value() > 0:
                                tip = t
                                break
                        bin_items.append((tip, ops[i]))

                if bin_items:
                    result_groups.append(bin_items)
                    num_bins += 1
                    print(f"Group {num_bins} {bin_items}")
        print()
        print("Time = ", solver.WallTime(), " milliseconds")
        print(f"Objective {solver.Objective().Value()}")
    else:
        print("The problem does not have an optimal solution.")

    return result_groups


def group_ops(pending_ops, conflicts, available_tips=list(range(8))):
    """
    A group is a 5-tuple of
    (
        pipetting steps required
        'source' or 'destination'
        the ops included across groups
        the group details, which is a 3-tuple of (ops, tip-op assignment for primary, tip-op assignment for secondary ),
        the overall tips used
    )
    """

    # Get open ops - the pending operations that don't have an unfulfilled dependency
    # Ie all ops we can select from at this time point
    open_ops = [
        op
        for op in pending_ops
        if op.source_dep not in pending_ops and op.dest_dep not in pending_ops
    ]

    open_ops.sort()

    selected_groups = ortools_grouping(
        pending_ops,
        conflicts,
    )

    total_asps = 0
    total_disp = 0

    for i, (group, tips) in enumerate(selected_groups):

        source_grouped = group_movements_needed(group, "source")
        dest_grouped = group_movements_needed(group, "dest")

        # print(f"Group {i}, {len(source_grouped)} sources, {len(dest_grouped)} dests")
        total_asps += len(source_grouped)
        total_disp += len(dest_grouped)
        pass

    print(f"Total {total_asps} {total_disp}")

    # np.savetxt("conflicts.csv", conflicts, delimiter=",", fmt="%d")

    # Get the best groups of ops when grouping by source,
    # and grouping by destination
    # See group_by method for more details
    best_groupings = group_by(open_ops, "source", available_tips)
    best_groupings += group_by(open_ops, "destination", available_tips)

    # Sort the groups by best cost, then earliest op, then the most ops achieved
    def group_sort_key(group):
        (steps, _, ops, _, _) = group
        # Cost is the number of steps (aspirates + dispenses) needed for the group,
        # divided by the number of ops (well transfers) achieved
        cost = steps / len(ops)
        return (cost, min([op.id for op in ops]), -1 * len(ops))

    best_groupings.sort(key=group_sort_key)
    return best_groupings
