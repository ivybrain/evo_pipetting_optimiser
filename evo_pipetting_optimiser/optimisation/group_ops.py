import itertools
from collections import deque
from robotools import *
from ..utils import *
import numpy as np


def group_ops(pending_ops):

    # Get open ops - the pending operations that don't have an unfulfilled dependency
    # Ie all ops we can select from at this time point
    open_ops = [
        op
        for op in pending_ops
        if op.source_dep not in pending_ops and op.dest_dep not in pending_ops
    ]

    open_ops.sort()

    # Get the best groups of ops when grouping by source,
    # and grouping by destination
    # See group_by method for more details
    best_groupings = group_by(open_ops, "source")
    best_groupings += group_by(open_ops, "destination")

    # Sort the groups by best cost, then earliest op, then the most ops achieved
    def group_sort_key(group):
        (steps, _, ops, _, _) = group
        # Cost is the number of steps (aspirates + dispenses) needed for the group,
        # divided by the number of ops (well transfers) achieved
        cost = steps / len(ops)
        return (cost, min([op.id for op in ops]), -1 * len(ops))

    best_groupings.sort(key=group_sort_key)
    return best_groupings


def group_by(open_ops, primary="source"):
    """
    Group open operations into efficient possibilities with an associated cost
    If we group by source (primary = source), then for each possible aspiration source (labware, column),
    We find the most ops we can complete with one aspiration from that source
    If we group by primary=destination, we find the most ops we can complete with one dispense to that destination
    The 'seccondary' is the opposite of the primary. I.e. if primary is source, seccondary is destination. We may need
    multiple seccondary dispenses for our one primary aspiration. However we find the combination which needs the least
    seccondary ops, for the single primary op
    """

    # Set the variables we need to track primary=source or primary=destination
    if primary == "source":
        # function that gives the source pos (row, column) when we call for primary
        primary_pos = lambda op: op.source_pos
        secondary_pos = lambda op: op.dest_pos
        primary_labware = lambda op: op.source
        secondary_labware = lambda op: op.destination
        secondary = "destination"
    elif primary == "destination":
        primary_pos = lambda op: op.dest_pos
        secondary_pos = lambda op: op.source_pos
        primary_labware = lambda op: op.destination
        secondary_labware = lambda op: op.source
        secondary = "source"

    # Group available operations by primary labware and primary column
    # i.e. Group by what can be achieved in a single aspiration (for primary=source)
    primary_dict = group_movements_needed(open_ops, primary)

    # Track the groupings we've found so far
    best_groupings = []

    primary_queue = deque(primary_dict.values())

    # Loop through all the primary groups we've found
    while len(primary_queue) > 0:
        selected_ops = primary_queue.popleft()

        primary_nrows = primary_labware(selected_ops[0]).n_rows

        # Check if this set of ops takes more than 8 rows (in say a 12 row tube carrier)
        # In which case, split into subgroups by row
        primary_rows = [(primary_pos(op)[0], op) for op in selected_ops]
        primary_rows.sort()

        if max(primary_rows)[0] - min(primary_rows)[0] >= 8:
            window_size = len(primary_rows) - 1
            for window_slide in range(2):
                window_selected = primary_rows[
                    window_slide : window_slide + window_size
                ]

                primary_queue.appendleft([row[1] for row in window_selected])
                pass
            continue

        # Group by secondary
        # i.e. when primary=source, find all the destination (labware, column) pairs we need to dispense to
        secondary_labware_col = group_movements_needed(selected_ops, secondary)
        secondary_labware_col_queue = deque(
            [set(x) for x in secondary_labware_col.values()]
        )

        # Track operation sets that are confirmed to be reachable in one dispense
        secondary_labware_col_reachable = []

        # Process each destination group of labware, column
        while len(secondary_labware_col_queue) > 0:
            secondary_op_group = secondary_labware_col_queue.popleft()

            # Calculate the number of pipetting steps needed to satisfy this group
            # It will be one step if the tips can line up from the source and the dest
            # Otherwise more

            # Get the rows needed among the source and the destination
            primary_rows_group = {}
            secondary_rows_group = {}
            row_conflict = False
            for op in secondary_op_group:
                primary_row = primary_pos(op)[0]
                secondary_row = secondary_pos(op)[0]
                # Check for row conflicts - if there is a repeated row in a labware (not trough) we can't use this group
                if (
                    primary_row in primary_rows_group
                    and not isinstance(primary_labware(op), Trough)
                ) or (
                    secondary_row in secondary_rows_group
                    and not isinstance(secondary_labware(op), Trough)
                ):
                    row_conflict = True
                primary_rows_group[primary_row] = op
                secondary_rows_group[secondary_row] = op

            # Represent the rows as a list with op.id if the row is used by an op, or none if it isn't
            primary_rows_mask = [
                primary_rows_group[i] if i in primary_rows_group else None
                for i in range(primary_nrows)
            ]
            secondary_rows_mask = [
                secondary_rows_group[i] if i in secondary_rows_group else None
                for i in range(min(secondary_rows_group), max(secondary_rows_group) + 1)
            ]

            # If the secondary rows, represented in a list (like [0, None, 1] for ops [0,1] on rows [5,7])
            # Are a substring of the primary rows (like [None, None, 0, None, 1, None, None, None])
            # We can aspirate this group in one shot
            # If the source is a trough, the rows are flexible, so we don't need this check
            # For sources with more than 12 rows, we can only take 8 rows
            if not row_conflict and (
                isinstance(selected_ops[0].source, Trough)
                or check_sublist(primary_rows_mask, secondary_rows_mask)
            ):

                # This means that the seccondary rows line up with the primary rows
                # Append the group to the confirmed reachable list, along with the rows needed
                secondary_labware_col_reachable.append(
                    (secondary_op_group, primary_rows_mask, secondary_rows_mask)
                )
            else:
                # Otherwise, we can't pipette these seccondary rows in one step. Split up to the smaller available subsets
                # And add back to the queue
                for op_group in itertools.combinations(
                    secondary_op_group, len(secondary_op_group) - 1
                ):
                    if set(op_group) in secondary_labware_col_queue:
                        continue
                    secondary_labware_col_queue.append(set(op_group))

        # Sort by biggest secondary groups
        def group_sort_key(group):
            # Want to sort by biggest, so use negative
            size = -1 * len(group[0])
            # If size is tied, sort by lowest primary row
            first_row = 0
            while group[1][first_row] == None:
                first_row += 1
            return (size, first_row)

        secondary_labware_col_reachable.sort(key=group_sort_key)

        # Now, we have grouped the secondary ops by what can be accomplished in one dispense (in case of secondary=destination)
        # Next, we want to select the most efficient non-conflicting combination of these groups
        # i.e. fill all 8 tips with the least number of groups
        # And, make sure that no two ops rely on the same primary well (as that would break the assumption of the group needing a single primary aspiration)

        # Store all non-conflicting combinations of groups
        valid_combinations = []
        # The number of tips we've filled so far
        most_tips_achieved = 0
        # Tips that are used in our final selected group
        tips_for_combo = []

        # NOTE: Efficiency could be drastically improved with dynamic programming
        # First, see how many tips we fill with combos of only 1 group
        # increase this to combos of up to 8 groups
        for combo_size in range(1, 8):
            # Iterate all possible combinations of groups
            # Of the specified size
            for combination in itertools.combinations(
                secondary_labware_col_reachable, combo_size
            ):
                # Add the tips needed across all groups in this combination
                tips_needed = sum([len(group[2]) for group in combination])
                # Check that this combo isn't over the tip limit
                if tips_needed > 8:
                    continue

                # Check that we don't have a conflict in primary mask
                # Get all indices in the primary mask for each group

                if primary == "source" and isinstance(selected_ops[0].source, Trough):
                    # If the primary is a trough, the primary rows are irrelevant, so get tips from the secondary rows instead

                    tips_used = 0
                    tip_indices = []
                    for group in combination:
                        group_indices = (
                            (np.array(list(group[2])) != None).nonzero()[0].tolist()
                        )

                        # Adjust so we don't overlap with the tips from the previous group
                        tip_indices.append([tip + tips_used for tip in group_indices])

                        tips_used = max(group_indices) + 1

                else:
                    # Otherwise, determine tip indices based on the primary mask
                    tip_indices = [
                        (np.array(list(group[1])) != None).nonzero()[0].tolist()
                        for group in combination
                    ]

                # We need to make sure none of the rows in the primary are repeated
                # As this would require more than one aspirate/dispense

                all_tips = [tip for group_tips in tip_indices for tip in group_tips]
                # If there are repeats, skip this group
                if len(set(all_tips)) != len(all_tips):
                    continue

                # List the tips needed for this group, if we offset tips to use tip 1 first
                # This will be used to assign tips for pipetting later
                tips_for_combo = [tip - min(all_tips) for tip in all_tips]

                skip_group = False
                tip_index = 0
                for group, _, _ in combination:
                    op = next(iter(group))

                    tip = tips_for_combo[tip_index]

                    # Check if an up offset limit or a down offset limit is voilated
                    offset_up_check, offset_down_check = check_offset_limits(op, tip)

                    if offset_up_check < 0:
                        # This means tips are travelling too far up, so we need to switch to earlier tips
                        # However, we already started at tip 1 for this group. This means the group can't work
                        # So skip it
                        skip_group = True

                    if offset_down_check > 0:

                        for i in range(len(tips_for_combo)):
                            # If we need more than 8 tips to work with this offset, the group can't work
                            if tips_for_combo[i] + offset_down_check >= 8:
                                skip_group = True
                            tips_for_combo[i] += offset_down_check

                    tip_index += len(group)

                if skip_group:
                    continue

                # Track the maximum tips we've seen for a combo
                most_tips_achieved = max(most_tips_achieved, tips_needed)
                # Append this combo to the valid list
                valid_combinations.append((tips_needed, combination, tips_for_combo))

            # If we've already found a combo that uses all 8 tips, stop searching
            if most_tips_achieved == 8:
                break

        # Sort by the number of tips used by this combo
        valid_combinations.sort(reverse=True)

        # Extract the list of groups from this combo
        selected_groups = [
            (
                sorted(
                    list(ops),
                    key=lambda op: (primary_pos(op)[0], secondary_pos(op)[0]),
                ),
                primary_mask,
                secondary_mask,
            )
            for (ops, primary_mask, secondary_mask) in valid_combinations[0][1]
        ]
        # Extract the list of ops among all groups in this combo
        selected_ops = [op for group in selected_groups for op in list(group[0])]
        tips_used = valid_combinations[0][2]

        # Track the total number of steps required for this group
        # It will always be one op for the primary primary plus the number of seccondary op groups we have
        total_steps = 1 + len(selected_groups)

        # Add this group to the list
        best_groupings.append(
            (
                total_steps,
                primary,
                selected_ops,
                selected_groups,
                tips_used,
            )
        )

    return best_groupings
