import itertools
from collections import deque
from robotools import *
from ..utils import *
import numpy as np
import pulp


def create_conflict_graph(pending_ops):

    max_op_id = max([op.id for op in pending_ops])

    conflicts = np.full((max_op_id + 1, max_op_id + 1), 0, dtype="i2")
    for i_op in pending_ops:
        for j_op in pending_ops:
            i = i_op.id
            j = j_op.id
            # Can always include op with itself
            if i == j:
                conflicts[i, j] = 1
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

            if source_conflict:
                if dest_conflict:
                    continue
                else:
                    conflicts[i, j] = 2
            else:
                conflicts[i, j] = 1

    return conflicts


def pulp_grouping(ops, conflicts):

    prob = pulp.LpProblem("Pipetting ops", pulp.LpMaximize)

    # Decision variable of include op or not
    x = [pulp.LpVariable(f"x_{i}", 0, 1, pulp.LpBinary) for i in range(len(ops))]

    prob += pulp.lpSum(x[i] for i in range(len(ops))), "Ops Included"
    prob += pulp.lpSum(x[i] for i in range(len(ops))) <= 8, "tip constraint"

    # Constraints from conflict graph
    for i, i_op in enumerate(ops):
        for j, j_op in enumerate(ops):
            if conflicts[i_op.id, j_op.id] == 0:
                prob += x[i] + x[j] <= 1, f"Conflict between {i_op.id} and {j_op.id}"

    prob.solve()

    selected_ops = [ops[i] for i in range(len(ops)) if x[i].varValue == 1]

    return selected_ops


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

    group = pulp_grouping(open_ops, conflicts)

    # Conflict graph for which ops cannot be included in pipetting operation together
    # 0 is conflict exists, 1 is can be grouped by source, 2 is can be grouped by dest

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
