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
    max_groups = int(len(ops) // 4) + 1

    min_offset = -7
    max_offset = 15
    offsets = range(min_offset, max_offset)

    for i, op in enumerate(ops):
        if i != op.id:
            raise Exception("Please provide ops with id increasing from 0")

    solver = pywraplp.Solver("", pywraplp.Solver.BOP_INTEGER_PROGRAMMING)
    # solver.EnableOutput()

    possible_locations = range(ops[0].loc_counter)
    complexity = len(ops) * len(possible_locations)

    # solver.SetTimeLimit(int((complexity / 20) * 1000))
    solver.SetTimeLimit(100000)
    solver.EnableOutput()

    if not solver:
        raise Exception("Error creating solver")

    # Track groups, which may contain more than one bin, as long as the bins have non-overlapping tips (can be pipetted without wash)
    groups = {}
    # At worst, 1 group per bin
    for g in range(max_groups):
        groups[g] = solver.BoolVar(f"group_{g}")
        # Baseline solution has all groups
        # solver.SetHint([groups[g]], [1])

    # Track which op is in which group
    op_in_group = {}
    for i in range(len(ops)):
        for g in range(max_groups):
            op_in_group[(i, g)] = solver.BoolVar(f"op_in_group_{i}_{g}")

    # If op is in group, group must be 1. Max capacity 8 ops
    for g in range(max_groups):
        solver.Add(
            groups[g] * 8 >= solver.Sum(op_in_group[(i, g)] for i in range(len(ops)))
        )

    # Each op must be in exactly one group
    for i in range(len(ops)):
        solver.Add(solver.Sum(op_in_group[i, g] for g in range(max_groups)) == 1)

    # Track tips for each op
    op_tip = {}
    for i in range(len(ops)):
        for t in range(n_tips):
            op_tip[i, t] = solver.BoolVar(f"op_tip_{i}_{t}")
        # Each op must have a tip assigned
        solver.Add(solver.Sum(op_tip[i, t] for t in range(n_tips)) == 1)

    # Track offset for each op
    op_source_offset = {}
    op_dest_offset = {}
    for i in range(len(ops)):
        for o in range(min_offset, max_offset):
            op_source_offset[i, o] = solver.BoolVar(f"op_source_offset_{i}_{o}")
            op_dest_offset[i, o] = solver.BoolVar(f"op_dest_offset_{i}_{o}")
            # solver.SetHint([op_source_offset[i, o], op_dest_offset[i, o]], [0, 0])

        solver.Add(solver.Sum(op_source_offset[i, o] for o in offsets) == 1)
        solver.Add(solver.Sum(op_dest_offset[i, o] for o in offsets) == 1)

    # track sources and dests present in each group
    group_source_offset = {}
    group_dest_offset = {}

    for g in range(max_groups):

        for l in possible_locations:
            for o in offsets:
                group_source_offset[(g, l, o)] = solver.BoolVar(
                    f"group_source_{g}_{l}_{o}"
                )
                group_dest_offset[(g, l, o)] = solver.BoolVar(f"group_dest_{g}_{l}_{o}")

        for i in range(len(ops)):
            for o in range(min_offset, max_offset):

                source = ops[i].source_loc
                dest = ops[i].dest_loc

                # If a op has a source or dest with offset, and op is in a group, that group must have that source dest and offset

                solver.Add(
                    group_source_offset[(g, source, o)]
                    >= op_in_group[(i, g)] + op_source_offset[i, o] - 1
                )
                solver.Add(
                    group_dest_offset[(g, dest, o)]
                    >= op_in_group[(i, g)] + op_dest_offset[i, o] - 1
                )

        solver.Add(
            solver.Sum(
                group_source_offset[(g, l, o)]
                for l in possible_locations
                for o in offsets
            )
            <= solver.Sum(op_in_group[(i, g)] for i in range(len(ops)))
        )

        # Each group can have at most 8 dest-offsets
        solver.Add(
            solver.Sum(
                group_dest_offset[(g, l, o)]
                for l in possible_locations
                for o in offsets
            )
            <= solver.Sum(op_in_group[(i, g)] for i in range(len(ops)))
        )

    # If op_tip set equal to op.source_pos[0] - op_source_offset
    for i in range(len(ops)):

        for t in range(n_tips):

            if getattr(ops[i].source, "virtual_rows", 0) or 0 > ops[i].source_pos[0]:
                solver.Add(op_source_offset[i, 0] == 1)
            else:
                source_offset = ops[i].source_pos[0] - t

                # If op i is in bin b AND op i has tip t, the dest offset must be set according to the dest row and that tip
                solver.Add(op_source_offset[i, source_offset] == op_tip[i, t])

            if getattr(ops[i].destination, "virtual_rows", 0) or 0 > ops[i].dest_pos[0]:
                solver.Add(op_dest_offset[i, 0] == 1)
            else:
                dest_offset = ops[i].dest_pos[0] - t

                # If op i is in bin b AND op i has tip t, the dest offset must be set according to the dest row and that tip
                solver.Add(op_dest_offset[i, dest_offset] == op_tip[i, t])

    # Each group must have only one of each tip
    for g in range(max_groups):
        for t in range(n_tips):
            tip_group_link_vars = []
            for i in range(len(ops)):
                link = solver.BoolVar(f"link_{i}_{t}")

                tip_group_link_vars.append(link)

                # Link var can only be true if op i is in group g
                solver.Add(link <= op_in_group[(i, g)])

                # Link var can only be true if op i has tip t
                solver.Add(link <= op_tip[(i, t)])

                # Link var is true if both are true
                solver.Add(link >= op_in_group[i, g] + op_tip[i, t] - 1)

            solver.Add(solver.Sum(tip_group_link_vars) <= 1)

    # Track group liquid classes
    liquid_classes = range(ops[0].lc_counter)

    # group_lc[g, lc] if ops in group g are liquid class lc
    group_lc = {}
    for lc in liquid_classes:
        for g in range(max_groups):
            group_lc[(g, lc)] = solver.BoolVar(f"group_lc_{g}_{lc}")

    # Each bin can only have one LC
    for g in range(max_groups):
        solver.Add(solver.Sum(group_lc[(g, lc)] for lc in liquid_classes) == 1)

    # All ops in bin must have matching lc
    for i in range(len(ops)):
        lc = ops[i].lc_id
        for g in range(max_groups):
            # Want to add that x[i, b] => bin_lc[b, lc]
            solver.Add(op_in_group[(i, g)] <= group_lc[(g, lc)])

    # Each item must be in exactly one group.
    for i in range(len(ops)):
        solver.Add(solver.Sum(op_in_group[i, b] for b in range(max_groups)) == 1)

    # Enforce op dependency

    def add_op_dep(child, dep):
        for g in range(max_groups):
            # Enforce that dep is in an earlier bin than child
            # If child is in bin b, then dep must be in one of the bins for dep_bin < b
            solver.Add(
                op_in_group[child.id, g]
                <= solver.Sum(op_in_group[dep.id, dep_group] for dep_group in range(g))
            )

    for op in ops:
        if op.source_dep:
            add_op_dep(op, op.source_dep)
            # print(f"Source dep {op.id} -> {op.source_dep}")
        if op.dest_dep:
            add_op_dep(op, op.dest_dep)
            # print(f"Dest dep {op.id} -> {op.dest_dep}")

    solver.Minimize(
        solver.Sum(groups[g] for g in range(max_groups))
        + solver.Sum(
            group_source_offset[g, l, o] + group_dest_offset[g, l, o]
            for g in range(max_groups)
            for l in possible_locations
            for o in offsets
        )
    )

    solver.SetNumThreads(4)

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

            sum_sources = 0
            sum_dests = 0

            for l in possible_locations:
                for o in range(min_offset, max_offset):

                    if (g, l, o) in group_source_offset:
                        sum_sources += group_source_offset[(g, l, o)].solution_value()
                    if (g, l, o) in group_dest_offset:
                        sum_dests += group_dest_offset[(g, l, o)].solution_value()

            for i in range(len(ops)):
                if op_in_group[(i, g)].solution_value() != 1:
                    continue

                tip = 0
                for t in range(n_tips):
                    if op_tip[i, t].solution_value():
                        tip = t

                source_offset = 0
                for o in range(min_offset, max_offset):
                    if op_source_offset[i, o].solution_value():
                        source_offset = o
                    if op_dest_offset[i, o].solution_value():
                        dest_offset = o
                group_items.append((ops[i], tip))  # , source_offset, dest_offset))

            if group_items:
                group_items.sort(key=lambda x: x[1])
                result_groups.append(group_items)
                num_groups += 1

                print(
                    f"Group {num_groups} {group_items} . {sum_sources} sources, {sum_dests} dests"
                )

        # print("Source offsets")
        # for key in op_source_offset.keys():
        #     print(f"{key} = {op_source_offset[key].solution_value()}")

        # print("Dest offsets")
        # for key in op_dest_offset.keys():
        #     print(f"{key} = {op_dest_offset[key].solution_value()}")

        # print("Sources")
        # for key in group_source_offset.keys():
        #     print(f"{key} = {group_source_offset[key].solution_value()}")

        # print("\nDests")
        # for key in group_dest_offset.keys():
        #     print(f"{key} = {group_dest_offset[key].solution_value()}")

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
