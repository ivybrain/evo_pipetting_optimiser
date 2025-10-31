from .group_ops import group_ops, create_conflict_graph
from copy import copy, deepcopy
import cProfile, pstats
import time


class OffsetLimitException(Exception):
    pass


def make_plan(pending_ops):
    """
    Create a list of pipetting actions from a set of pending ops
    Actions are a tuple of
    (Type, [ops], [tips])
    Where type is one of "Aspirate", "Dispense", "Wash"
    """

    # Count the aspirates, dispenses, and washes we use
    # for performance tracking
    asp_count = 0
    disp_count = 0
    wash_count = 0

    actions = []

    # Create a copy of pending ops set that we can mutate
    pending_ops = copy(pending_ops)

    completed_ops = set()

    labware_copies = {}

    def labware_copy(labware):
        if labware.name not in labware_copies:
            labware_copies[labware.name] = deepcopy(labware)

        return labware_copies[labware.name]

    start_time = time.time()

    prev_tip_op = {}
    # profiler = cProfile.Profile()
    # profiler.enable()

    conflicts = create_conflict_graph(pending_ops)

    # Repeat until we have no more operations pending
    while len(pending_ops) > 0:
        # Group the operations. See group_ops method for details

        best_groupings = group_ops(pending_ops, conflicts)

        # TO implement: All groups that do not contain newly completed ops are still valid
        # keep in memory and skip calculation for next

        # In the simple case, we can just take the single best group. However, if the best group doesn't use all 8 tips,
        # We can select other groups to use the remaining tips
        # Before washing them all together

        # Track how many tips we've used across the currently selected groups
        tips_used = 0
        # Track the selected groups, and the ops among those selected groups
        selected_groups = []
        selected_ops = []

        # Store the cost of executing the second best group available
        # This is a benchmark for whether to include subsequent groups
        # e.g. if it's more efficient to select the second best group and do an additional wash,
        # compared to adding an unefficient group to the remaining tips,
        # Then don't add any more groups, and leave the remaining tips empty
        # second best cost is the number of ((aspirates + dispenses) + 2 (as a wash takes two movements into the waste then cleaner)),
        # divided by the number of ops achieved by that group
        second_best_cost = (best_groupings[1][0] + 2) / len(best_groupings[1][2])

        (steps, group_type, ops, target_groups, tips_selected) = best_groupings[0]

        selected_groups.append((group_type, ops, target_groups, tips_selected))
        selected_ops += ops

        if len(selected_ops) == 0:
            raise OffsetLimitException(
                "No valid groups to select. This probably means you're trying to transfer between two plates with offset restrictions that don't allow this transfer"
            )

        # Process the groups into a list of sources to aspirate and a list of destinations to dispense
        source_list = []
        dest_list = []
        op_tip = {op: tip + 1 for op, tip in zip(ops, tips_selected)}

        # Check if we need to wash any tips
        wash_needed = False
        for tip, op in zip(tips_selected, ops):
            # Standard or decontamination mean we require a wash
            if op.wash_scheme in [1, "D"] and tip in prev_tip_op:
                wash_needed = True

            if op.wash_scheme == "smart" and tip in prev_tip_op:
                past_op = prev_tip_op[tip]

                past_source = labware_copy(past_op.source)
                past_dest = labware_copy(past_op.destination)
                source = labware_copy(op.source)
                dest = labware_copy(op.destination)

                past_source_comp = past_source.get_well_composition(
                    past_source.wells[past_op.source_pos]
                )

                past_dest_comp = past_dest.get_well_composition(
                    past_dest.wells[past_op.dest_pos]
                )

                source_comp = source.get_well_composition(source.wells[op.source_pos])

                past_touched = list(past_source_comp.items()) + list(
                    past_dest_comp.items()
                )

                for prev_key, prev_conc in past_touched:
                    # If the previous op touched a liquid that is not present in our source, we need to wash to not contaminate it
                    if prev_key not in source_comp:
                        wash_needed = True
                        break

                    # If last op touched a well with more than twice the conc of the source, wash to not contaminate with the higher conc
                    if prev_conc > 2 * source_comp[prev_key]:
                        wash_needed = True
                        break

                # If we've already determined we need a wash, no point checking the other tips
                if wash_needed:
                    break

            prev_tip_op[tip] = op

        if wash_needed:
            actions.append(("Wash", ops, list(op_tip.values())))
            wash_count += 1

        for group_type, ops, target_groups, tips_selected in selected_groups:

            if group_type == "source":
                # If we've grouped by source, we can aspirate all ops in the group at once
                source_list += [ops]
                # Destinations will depend on the subgroups selected, one for each subgroup
                dest_list += [group[0] for group in target_groups]

            else:
                # If we've grouped by destination, we can dispense all ops in the group at once
                dest_list += [ops]
                # Sources will depend on the subgroups selected, one for each subgroup
                source_list += [group[0] for group in target_groups]

        # Sort the source list and dest list by the tips used
        # This just makes sure the pipetting occurs in an order that's less confusing visually,
        # starting from tip 1 to tip 8
        group_sort_key = lambda ops: min([op_tip[op] for op in ops])
        source_list.sort(key=group_sort_key)
        dest_list.sort(key=group_sort_key)

        for source_group in source_list:

            # Sort the group by assigned tip
            source_group.sort(key=lambda op: op_tip[op])

            actions.append(
                ("Aspirate", source_group, [op_tip[op] for op in source_group])
            )

            # Remove liquid from our virtual copy of the plates so we can track composition

            virtual_source = labware_copy(source_group[0].source)
            virtual_source.remove(
                [virtual_source.wells[op.source_pos] for op in source_group],
                [op.volume for op in source_group],
            )

            asp_count += 1

        for dest_group in dest_list:

            # Sort the group by assigned tip
            dest_group.sort(key=lambda op: op_tip[op])

            actions.append(("Dispense", dest_group, [op_tip[op] for op in dest_group]))

            # Add liquid to our virtual copy of the plates so we can track composition
            virtual_dest = labware_copy(dest_group[0].destination)
            compositions = [
                labware_copy(op.source).get_well_composition(
                    op.source.wells[op.source_pos]
                )
                for op in dest_group
            ]
            virtual_dest.add(
                [virtual_dest.wells[op.dest_pos] for op in dest_group],
                [op.volume for op in dest_group],
                compositions=compositions,
            )

            disp_count += 1

        pending_ops.difference_update(selected_ops)
        completed_ops.update(selected_ops)

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats("program.cprof")

    cost = asp_count + disp_count + (wash_count * 2)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")

    return actions, cost, (asp_count, disp_count, wash_count)
