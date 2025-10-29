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


def pulp_grouping(ops, source_groups, dest_groups):

    prob = pulp.LpProblem("Pipetting ops", pulp.LpMinimize)

    # Decision variable of include op or not
    x = [pulp.LpVariable(f"x_{i}", 0, 1, pulp.LpBinary) for i in range(len(ops))]

    # Variables for each possible group of labware, column
    possible_locations = range(ops[0].loc_counter)

    # Variables to track if locations are present in the selected set

    # sp[i] is 1 if location i is present in the selected set of x sources
    source_present = pulp.LpVariable.dicts("sp", possible_locations, cat="Binary")

    # dp[i] is 1 if location i is present in the selected set of x dests
    dest_present = pulp.LpVariable.dicts("dp", possible_locations, cat="Binary")

    # constraints for source_present and dest_present

    for l in possible_locations:
        ops_with_source = [i for i, op in enumerate(ops) if op.source_loc == l]
        ops_with_dest = [i for i, op in enumerate(ops) if op.dest_loc == l]

        prob += (
            pulp.lpSum(x[i] for i in ops_with_source)
            <= len(ops_with_source) * source_present[l],
            f"Source_Used_{l}",
        )
        prob += (
            pulp.lpSum(x[i] for i in ops_with_dest)
            <= len(ops_with_dest) * dest_present[l],
            f"Dest_Used_{l}",
        )

    prob += (
        pulp.lpSum(source_present) + pulp.lpSum(dest_present),
        "Objective",
    )
    prob += pulp.lpSum(x) == 8, "tip constraint"

    # Constraints from conflict graph
    for i, i_op in enumerate(ops):
        for j, j_op in enumerate(ops):
            if (
                source_groups[i_op.id, j_op.id] == 0
                and dest_groups[i_op.id, j_op.id] == 0
            ):
                prob += x[i] + x[j] <= 1, f"Conflict between {i_op.id} and {j_op.id}"

    prob.solve()

    selected_ops = [ops[i] for i in range(len(ops)) if x[i].varValue == 1]

    print(selected_ops)

    for source in range(ops[0].loc_counter):
        if pulp.value(source_present[source]) == 1:
            print(f"  - {source}")

    source_grouped = group_movements_needed(selected_ops, "source")
    dest_grouped = group_movements_needed(selected_ops, "dest")

    for v in prob.variables():
        print(v.name, "=", v.varValue)

    return selected_ops


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

    group = pulp_grouping(open_ops, source_groups, dest_groups)

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
