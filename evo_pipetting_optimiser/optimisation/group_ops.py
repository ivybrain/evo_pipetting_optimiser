import itertools
from collections import deque
from robotools import *
from ..utils import *
import numpy as np
from ..robotools_extensions import TransferOperation
from ortools.linear_solver import pywraplp
import time
from copy import copy


def ortools_grouping(
    ops,
):
    start_time = time.time()
    ops = list(ops)
    n_tips = 8
    max_bins = len(ops)
    max_groups = int(len(ops) // 8) + 1

    min_offset = -7
    max_offset = 15
    offsets = range(min_offset, max_offset)

    for i, op in enumerate(ops):
        if i != op.id:
            raise Exception("Please provide ops with id increasing from 0")

    solver = pywraplp.Solver("", pywraplp.Solver.BOP_INTEGER_PROGRAMMING)
    # solver.EnableOutput()

    solver.SetTimeLimit(20000)
    solver.EnableOutput()

    if not solver:
        raise Exception("Error creating solver")

    possible_locations = range(ops[0].loc_counter)

    # Track which op is in which bin
    x = {}
    for i in range(len(ops)):
        for b in range(max_bins):
            x[(i, b)] = solver.BoolVar(f"x_{i}_{b}")

    # Track bins used
    y = {}
    for b in range(max_bins):
        y[b] = solver.BoolVar(f"y_{b}")

    # track source and dest location of each bin
    bin_source = {}
    bin_dest = {}

    for l in possible_locations:
        for b in range(max_bins):
            bin_source[(b, l)] = solver.BoolVar(f"bin_source_{b}_{l}")
            bin_dest[(b, l)] = solver.BoolVar(f"bin_dest_{b}_{l}")

    # Each bin can only have one loc
    for b in range(max_bins):
        solver.Add(solver.Sum(bin_source[(b, l)] for l in possible_locations) == 1)
        solver.Add(solver.Sum(bin_dest[(b, l)] for l in possible_locations) == 1)

    # All ops in bin must have matching source and dest loc
    for i in range(len(ops)):
        source = ops[i].source_loc
        dest = ops[i].dest_loc
        for b in range(max_bins):
            # Want to add that x[i, b] => bin_source[b, l]
            solver.Add(x[(i, b)] <= bin_source[(b, source)])
            solver.Add(x[(i, b)] <= bin_dest[(b, dest)])

    # # track offset of each bin
    bin_source_offset = {}
    bin_dest_offset = {}
    for o in offsets:
        for b in range(max_bins):
            bin_source_offset[(b, o)] = solver.BoolVar(f"bin_source_offset_{b}_{o}")
            bin_dest_offset[(b, o)] = solver.BoolVar(f"bin_dest_offset_{b}_{o}")

    # Each bin can only have one offset
    for b in range(max_bins):
        solver.Add(solver.Sum(bin_source_offset[(b, o)] for o in offsets) == 1)
        solver.Add(solver.Sum(bin_dest_offset[(b, o)] for o in offsets) == 1)

    # Track groups, which may contain more than one bin, as long as the bins have non-overlapping tips (can be pipetted without wash)
    groups = {}
    # At worst, 1 group per bin
    for g in range(max_groups):
        groups[g] = solver.BoolVar(f"group_{g}")

    # Track which bin is in which group
    bin_in_group = {}
    for b in range(max_bins):
        for g in range(max_groups):
            bin_in_group[(b, g)] = solver.BoolVar(f"bin_in_group_{b}_{g}")

    # If bin is in group, group must be 1
    for g in range(max_groups):
        solver.Add(
            groups[g] * 8 >= solver.Sum(bin_in_group[(b, g)] for b in range(max_bins))
        )

    # Each bin must be in exactly one group
    for b in range(max_bins):
        solver.Add(solver.Sum(bin_in_group[b, g] for g in range(max_groups)) == 1)

    # Track tips for each op
    op_tip = {}
    for i in range(len(ops)):
        for t in range(n_tips):
            op_tip[i, t] = solver.BoolVar(f"op_tip_{i}_{t}")
        # Each op must have a tip assigned
        solver.Add(solver.Sum(op_tip[i, t] for t in range(n_tips)) == 1)

    # Track tips for each bin
    bin_tip = {}
    for b in range(max_bins):
        for t in range(n_tips):
            bin_tip[b, t] = solver.BoolVar(f"bin_tip_{b}_{t}")

    # If op is in bin, op_tip set equal to op.source_pos[0] - bin_source_offset
    for i in range(len(ops)):
        for b in range(max_bins):

            possible_offsets = copy(offsets)

            # Require that the dest offset matches the tip, if dest is not a trough
            if not (
                getattr(ops[i].destination, "virtual_rows", 0) or 0 > ops[i].dest_pos[0]
            ):
                for t in range(n_tips):
                    # The offset that matches the given dest row and this tip
                    dest_offset = ops[i].dest_pos[0] - t

                    # If op i is in bin b AND op i has tip t, the dest offset must be set according to the dest row and that tip
                    solver.Add(
                        bin_dest_offset[b, dest_offset] >= x[i, b] + op_tip[i, t] - 1
                    )
            else:
                # Otherwise, just set the dest offset to 0 as it's irrelevant
                possible_offsets = [0]

            # if source has virtual rows, the only constraint on tips is labware limits
            if getattr(ops[i].source, "virtual_rows", 0) or 0 > ops[i].source_pos[0]:
                # Set offset 0, as it's irrelevant
                possible_offsets = [0]

            # Now link tip assignment to offset
            # For each offset o, if bin b has offset o and op i is in bin b,
            # then op_tip[i, t] must correspond to the correct offset
            for o in possible_offsets:

                tip_index = ops[i].source_pos[0] - o

                # If bin b has offset o and op i is in bin b, then enforce tip mapping
                # We use implication: x[i,b] AND bin_source_offset[b,o] => op_tip[i, t] when t = op_row - o
                if 0 <= tip_index < n_tips:
                    solver.Add(
                        op_tip[i, tip_index]
                        >= x[(i, b)] + bin_source_offset[(b, o)] - 1
                    )

                    # Also, if this op is in this bin with this offset, the bin must have the matching tip enabled
                    solver.Add(
                        bin_tip[(b, tip_index)]
                        >= x[(i, b)] + bin_source_offset[(b, o)] - 1
                    )
                else:
                    # Otherwise, this op in this bin with this offset is not feasible
                    # Cannot have both x[i,b] and bin_source_offset[b,o]
                    solver.Add(x[i, b] + bin_source_offset[b, o] <= 1)

    # Each group must have only one of each tip
    for g in range(max_groups):
        for t in range(n_tips):
            tip_group_link_vars = []
            for b in range(max_bins):
                link = solver.BoolVar(f"link_{i}_{t}_{b}")

                tip_group_link_vars.append(link)

                # Link var can only be true if bin b is in group g
                solver.Add(link <= bin_in_group[(b, g)])

                # Link var can only be true if bin b has tip t
                solver.Add(link <= bin_tip[(b, t)])

                # Link var is true if both are true
                solver.Add(link >= bin_in_group[b, g] + bin_tip[b, t] - 1)

            solver.Add(solver.Sum(tip_group_link_vars) <= 1)

    # Track bin liquid classes
    liquid_classes = range(ops[0].lc_counter)

    # bin_lc[b, lc] if ops in bin 1 are liquid class lc
    bin_lc = {}
    for lc in liquid_classes:
        for b in range(max_bins):
            bin_lc[(b, lc)] = solver.BoolVar(f"bin_lc_{b}_{lc}")

    # Each bin can only have one LC
    for b in range(max_bins):
        solver.Add(solver.Sum(bin_lc[(b, lc)] for lc in liquid_classes) == 1)

    # All ops in bin must have matching lc
    for i in range(len(ops)):
        lc = ops[i].lc_id
        for b in range(max_bins):
            # Want to add that x[i, b] => bin_lc[b, lc]
            solver.Add(x[(i, b)] <= bin_lc[(b, lc)])

    # Each item must be in exactly one bin.
    for i in range(len(ops)):
        solver.Add(solver.Sum(x[i, b] for b in range(max_bins)) == 1)

    # Bin capacity 8 items
    for b in range(max_bins):
        solver.Add(solver.Sum(x[(i, b)] for i in range(len(ops))) <= y[b] * 8)

    # Enforce op dependency

    def add_op_dep(child, dep):
        for b in range(max_bins):
            # Enforce that dep is in an earlier bin than child
            # If child is in bin b, then dep must be in one of the bins for dep_bin < b
            solver.Add(
                x[child.id, b] <= solver.Sum(x[dep.id, dep_bin] for dep_bin in range(b))
            )

    for op in ops:
        if op.source_dep:
            add_op_dep(op, op.source_dep)
            # print(f"Source dep {op.id} -> {op.source_dep}")
        if op.dest_dep:
            add_op_dep(op, op.dest_dep)
            # print(f"Dest dep {op.id} -> {op.dest_dep}")

    # Don't let ops with row conflicts be in same bin
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            if (
                ops[i].source_pos[0] == ops[j].source_pos[0]
                or ops[i].dest_pos[0] == ops[j].dest_pos[0]
            ):
                for b in range(max_bins):
                    solver.Add(x[i, b] + x[j, b] <= 1)

    solver.Minimize(
        (2 * solver.Sum(groups[g] for g in range(max_groups)))
        + solver.Sum(y[b] for b in range(max_bins))
    )

    print(f"Model set-up: {time.time()-start_time:.4f} seconds")

    # with open("model.txt", "w") as f:

    #     f.write(solver.ExportModelAsLpFormat(False))

    start_time = time.time()
    status = solver.Solve()

    print(f"Solve time: {time.time()-start_time:.4f} seconds")

    result_groups = []
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        num_groups = 0
        num_bins = 0
        for g in range(max_groups):

            if groups[g].solution_value() != 1:
                continue

            group_items = []

            for b in range(max_bins):
                if bin_in_group[(b, g)].solution_value() == 1:
                    bin_items = []
                    bin_tips = []
                    bin_tips_2 = []
                    source = 0
                    dest = 0
                    source_offset = 0
                    dest_offset = 0

                    for l in possible_locations:
                        if bin_source[b, l].solution_value():
                            source = l
                        if bin_dest[b, l].solution_value():
                            dest = l

                    for o in offsets:
                        if bin_source_offset[b, o].solution_value():
                            source_offset = o
                        if bin_dest_offset[b, o].solution_value():
                            dest_offset = o
                    for t in range(n_tips):
                        if bin_tip[(b, t)].solution_value():
                            bin_tips_2.append(t)
                    for i in range(len(ops)):
                        if x[i, b].solution_value() > 0:
                            tip = 0
                            for t in range(n_tips):
                                if op_tip[i, t].solution_value():
                                    tip = t
                            bin_tips.append(tip)
                            bin_items.append(ops[i])

                    if bin_items:
                        group_items.append((bin_items, bin_tips))
                        num_bins += 1
                        print(
                            f"Bin {num_bins} {bin_items} {bin_tips} {bin_tips_2} {source}-{source_offset} {dest}-{dest_offset}"
                        )

            result_groups.append(group_items)
            num_groups += 1
            print(f"Group {num_groups} {group_items}")

        print()
        print("Time = ", solver.WallTime(), " milliseconds")
        print(f"Objective {solver.Objective().Value()}")
    else:
        print("The problem does not have an optimal solution.")
        raise Exception("Solver error")

    return result_groups


def group_ops(pending_ops):
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

    selected_groups = ortools_grouping(
        pending_ops,
    )

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
