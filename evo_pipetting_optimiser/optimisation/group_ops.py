import itertools
from collections import deque
from robotools import *
from ..utils import *
import numpy as np
import pulp
from ..robotools_extensions import TransferOperation


def create_conflict_graph(pending_ops):

    max_op_id = max([op.id for op in pending_ops])

    # Show which ops can have their sources grouped together
    # If 0, can't be grouped
    # If positive, can be grouped at the source with a certain labware,column index
    source_groups = np.full((max_op_id + 1, max_op_id + 1), 0, dtype=np.uint16)

    # Same for dest
    dest_groups = np.full((max_op_id + 1, max_op_id + 1), 0, dtype=np.uint16)

    for i_op in pending_ops:
        for j_op in pending_ops:
            i = i_op.id
            j = j_op.id
            # Can always include op with itself
            if i == j:
                source_groups[i, j] = i_op.source_loc
                dest_groups[i, j] = i_op.dest_loc
                continue

            # Check liquid class is the same
            if i_op.liquid_class != j_op.liquid_class:
                continue

            # If can't do in 1 asp or 1 disp, then can't group
            source_conflict = False
            dest_conflict = False
            # Check source is the same
            if i_op.source != j_op.source:
                source_conflict = True

            if i_op.source_pos[1] != j_op.source_pos[1]:
                source_conflict = True

            # well overlap
            if (
                i_op.source_pos[0] == j_op.source_pos[0]
                and not (i_op.source.virtual_rows or 0) >= i_op.source_pos[0]
            ):
                source_conflict = True

            if i_op.destination != j_op.destination:
                dest_conflict = True
            if i_op.dest_pos[1] != j_op.dest_pos[1]:
                dest_conflict = True
            if (
                i_op.dest_pos[0] == j_op.dest_pos[0]
                and not (i_op.destination.virtual_rows or 0) >= i_op.dest_pos[0]
            ):
                dest_conflict = True

            if not source_conflict:
                source_groups[i, j] = i_op.source_loc

            if not dest_conflict:
                dest_groups[i, j] = i_op.dest_loc

    return source_groups, dest_groups


def pulp_grouping(ops, source_groups, dest_groups, tips_available=8):

    prob = pulp.LpProblem("Pipetting ops", pulp.LpMinimize)
    ops = list(ops)
    max_bins = (len(ops) // 8) + 1

    # Decision variable of include op or not
    # x = [pulp.LpVariable(f"x_{i}", 0, 1, pulp.LpBinary) for i in range(len(ops))]
    op_in_bin = pulp.LpVariable.dicts(
        "OpInBin", (range(len(ops)), range(max_bins)), 0, 1, pulp.LpBinary
    )

    bin_used = pulp.LpVariable.dicts("BinUsed", range(max_bins), 0, 1, pulp.LpBinary)

    for b in bin_used:
        prob += (
            pulp.lpSum(op_in_bin[i][b] for i in range(len(ops))) <= 8 * bin_used[b],
            f"Capacity_Bin_{b}",
        )

    prob += (
        pulp.lpSum(bin_used),
        "Objective minimise bins used",
    )

    # Assignment constraint
    for i in range(len(ops)):
        prob += (
            pulp.lpSum([op_in_bin[i][b] for b in range(max_bins)]) == 1,
            f"Assign_Item_{i}",
        )

    for i in range(len(ops)):
        for b in bin_used:
            prob += op_in_bin[i][b] <= bin_used[b], f"Link_{i}_{b}"

    # Constraints from conflict graph
    for i, i_op in enumerate(ops):
        for j, j_op in enumerate(ops):
            if (
                source_groups[i_op.id, j_op.id] == 0
                and dest_groups[i_op.id, j_op.id] == 0
            ):
                for b in range(max_bins):
                    prob += (
                        op_in_bin[i][b] + op_in_bin[j][b] <= 1,
                        f"Conflict between {i_op.id} and {j_op.id} in bin {b}",
                    )

    prob.solve()

    result = []

    for b in bin_used:
        if pulp.value(bin_used[b]) == 1:
            this_bin = []
            for i in range(len(ops)):
                if pulp.value(op_in_bin[i][b]) == 1:
                    this_bin.append(ops[i])

            result.append(this_bin)

    return result


def group_ops(pending_ops, source_groups, dest_groups, available_tips=list(range(8))):
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

    selected_groups = pulp_grouping(
        pending_ops, source_groups, dest_groups, tips_available=8
    )

    for i, group in enumerate(selected_groups):

        source_grouped = group_movements_needed(group, "source")
        dest_grouped = group_movements_needed(group, "dest")
        print(f"Group {i}, {len(source_grouped)} sources, {len(dest_grouped)} dests")
        pass

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
