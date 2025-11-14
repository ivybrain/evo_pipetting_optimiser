from .group_ops import group_ops, ortools_grouping
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

    labware_copies = {}

    def labware_copy(labware):
        if labware.name not in labware_copies:
            labware_copies[labware.name] = deepcopy(labware)

        return labware_copies[labware.name]

    start_time = time.time()

    prev_tip_op = {}
    # profiler = cProfile.Profile()
    # profiler.enable()

    # Group the operations. See group_ops method for details

    best_groupings = ortools_grouping(pending_ops)

    # In the simple case, we can just take the single best group. However, if the best group doesn't use all 8 tips,
    # We can select other groups to use the remaining tips
    # Before washing them all together

    for group in best_groupings:

        # Process the groups into a list of sources to aspirate and a list of destinations to dispense
        ops = []
        tips = []
        source_list = {}
        dest_list = {}
        op_tip = {}

        for op, tip, source, dest in group:
            ops.append(op)
            tips.append(tip + 1)
            op_tip[op] = tip + 1

            if source not in source_list:
                source_list[source] = []
            source_list[source].append(op)

            if dest not in dest_list:
                dest_list[dest] = []
            dest_list[dest].append(op)

        # Check if we need to wash any tips
        wash_needed = False
        for op, tip, _, _ in group:
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

        # for group_type, ops, target_groups, tips_selected in selected_groups:

        #     if group_type == "source":
        #         # If we've grouped by source, we can aspirate all ops in the group at once
        #         source_list += [ops]
        #         # Destinations will depend on the subgroups selected, one for each subgroup
        #         dest_list += [group[0] for group in target_groups]

        #     else:
        #         # If we've grouped by destination, we can dispense all ops in the group at once
        #         dest_list += [ops]
        #         # Sources will depend on the subgroups selected, one for each subgroup
        #         source_list += [group[0] for group in target_groups]

        # Sort the source list and dest list by the tips used
        # This just makes sure the pipetting occurs in an order that's less confusing visually,
        # starting from tip 1 to tip 8

        for source_group in source_list.values():

            actions.append(
                ("Aspirate", source_group, [op_tip[op] for op in source_group])
            )

            # # Remove liquid from our virtual copy of the plates so we can track composition

            # virtual_source = labware_copy(source_group[0].source)
            # virtual_source.remove(
            #     [virtual_source.wells[op.source_pos] for op in source_group],
            #     [op.volume for op in source_group],
            # )

            asp_count += 1

        for dest_group in dest_list.values():

            actions.append(("Dispense", dest_group, [op_tip[op] for op in dest_group]))

            # Add liquid to our virtual copy of the plates so we can track composition
            # virtual_dest = labware_copy(dest_group[0].destination)
            # compositions = [
            #     labware_copy(op.source).get_well_composition(
            #         op.source.wells[op.source_pos]
            #     )
            #     for op in dest_group
            # ]
            # virtual_dest.add(
            #     [virtual_dest.wells[op.dest_pos] for op in dest_group],
            #     [op.volume for op in dest_group],
            #     compositions=compositions,
            # )

            disp_count += 1

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats("program.cprof")

    cost = asp_count + disp_count + (wash_count * 2)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")

    return actions, cost, (asp_count, disp_count, wash_count)
