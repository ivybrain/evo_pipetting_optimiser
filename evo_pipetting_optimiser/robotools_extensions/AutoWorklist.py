from robotools import *
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from . import *
import itertools
import warnings
from collections import deque
import numpy as np
import math


class TransferOperation:
    """
    Internal class that stores all the information needed
    For a transfer from one well to another
    """

    op_id_counter = 0

    def __init__(
        self,
        source: liquidhandling.Labware,
        source_pos: Tuple[int, int],
        destination: AdvancedLabware,
        dest_pos: Tuple[int, int],
        volume: float,
        *,
        label: Optional[str] = None,
        on_underflow: Literal["debug", "warn", "raise"] = "raise",
        source_dep=None,
        dest_dep=None,
        liquid_class=None,
        **kwargs,
    ):
        self.id = TransferOperation.op_id_counter
        TransferOperation.op_id_counter += 1

        self.source = source
        self.source_pos = source_pos
        self.destination = destination
        self.dest_pos = dest_pos

        self.volume = volume
        self.label = label
        self.on_underflow = on_underflow

        self.source_dep = source_dep
        self.dest_dep = dest_dep

        self.selected_tip = None

        self.liquid_class = liquid_class

    def __str__(self):
        return f"{self.id}: {self.label} {self.source_pos} to {self.dest_pos}"

    def __repr__(self):
        return str(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class AutoWorklist(EvoWorklist):

    def __init__(
        self,
        *args,
        waste_location: Tuple[int, int] = None,
        cleaner_location: Tuple[int, int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.completed_ops = set()
        self.pending_ops = set()

        self.tips_used = [False] * 8
        self.tip_contents = [None] * 8

        self.currently_optimising = False
        self.silence_append_warning = False

        assert waste_location is not None, "Must define waste location grid and site"
        assert (
            cleaner_location is not None
        ), "Must define cleaner location grid and site"

        self.waste_location = waste_location
        self.cleaner_location = cleaner_location

    def auto_transfer(
        self,
        source: Union[AdvancedLabware, Trough],
        source_wells: Union[str, Sequence[str], np.ndarray],
        destination: AdvancedLabware,
        destination_wells: Union[str, Sequence[str], np.ndarray],
        volumes: Union[float, Sequence[float], np.ndarray],
        *,
        label: Optional[str] = "",
        on_underflow: Literal["debug", "warn", "raise"] = "raise",
        liquid_class: str = None,
        wash=True,
        **kwargs,
    ) -> None:
        """Transfer operation between two labwares with automatic pipetting order optimisation
        Ensures correct order of transfers to each particular well,
        While grouping transfers together for pipetting efficiency
        use .commit() to calculate the pipetting strategy and populate the worklist with the corresponding commands

        Parameters
        ----------
        source : AdvancedLabware or Trough (with location property set to (grid,site) tuple)
        source_wells : str or iterable
            List of source well ids
        destination : AdvancedLabware
            Destination labware
        destination_wells : str or iterable
            List of destination well ids
        volumes : float or iterable
            Volume(s) to transfer
        label : str
            Label of the operation to log into labware history
        liquid_class : str
            Liquid class to use for pipetting

        on_underflow
            What to do about volume underflows (going below ``vmin``) in non-empty wells.

            Options:

            - ``"debug"`` mentions the underflowing wells in a log message at DEBUG level.
            - ``"warn"`` emits an :class:`~robotools.liquidhandling.exceptions.VolumeUnderflowWarning`. This `can be captured in unit tests <https://docs.pytest.org/en/stable/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests>`_.
            - ``"raise"`` raises a :class:`~robotools.liquidhandling.exceptions.VolumeUnderflowError` about underflowing wells.
        """
        # reformat the convenience parameters
        source_wells = np.array(source_wells).flatten("F")
        destination_wells = np.array(destination_wells).flatten("F")
        volumes = np.array(volumes).flatten("F")
        nmax = max((len(source_wells), len(destination_wells), len(volumes)))

        if len(source_wells) == 1:
            source_wells = np.repeat(source_wells, nmax)
        if len(destination_wells) == 1:
            destination_wells = np.repeat(destination_wells, nmax)
        if len(volumes) == 1:
            volumes = np.repeat(volumes, nmax)
        lengths = (len(source_wells), len(destination_wells), len(volumes))
        assert (
            len(set(lengths)) == 1
        ), f"Number of source/destination/volumes must be equal. They were {lengths}"

        assert isinstance(source, AdvancedLabware) or isinstance(
            source, Trough
        ), "Source must be AdvancedLabware or Trough for auto_transfer"

        assert isinstance(
            destination, AdvancedLabware
        ), "Destination must be AdvancedLabware for auto_transfer"

        assert (
            liquid_class is not None
        ), "Liquid class must be speicified for auto_transfer"

        # Track if we currently have operations waiting to be optimised
        # That haven't been committed to the worklist. If so, we want to warn if the user
        # Tries to add anything else to the worklist.
        self.currently_optimising = True

        # source wells don't matter for a trough, set them all 0
        # then optimiser can arrange as needed
        if isinstance(source, Trough):
            source_wells = ["A01"] * len(source_wells)

        # Create a TransferOperation object with all the details of the transfer
        # For every pair of source,destination wells
        for i in range(len(source_wells)):

            # Check for large volume handling
            repeats = math.ceil(volumes[i] / self.max_volume)
            volume = volumes[i] / repeats
            source_dep = (
                source.last_op[source_wells[i]]
                if isinstance(source, AdvancedLabware)
                else None
            )
            dest_dep = (destination.last_op[destination_wells[i]],)

            for j in range(repeats):

                op = TransferOperation(
                    source,
                    source.indices[source_wells[i]],
                    destination,
                    destination.indices[destination_wells[i]],
                    volume,
                    label=label,
                    wash=wash,
                    on_underflow=on_underflow,
                    source_dep=source_dep,
                    dest_dep=dest_dep,
                    liquid_class=liquid_class,
                )

                # Set the source and dest dependencies of the next LVH repeat op to this op
                source_dep = op
                dest_dep = op

                # Append this op to the labware we're aspirating to
                # So that future transfers to the well know that they need to wait for this transfer first
                destination.op_tracking[destination_wells[i]].append(op)

                # Add this op to the pending operations set
                self.pending_ops.update([op])

    def append(self, *args, **kwargs):
        # If we have un-commited auto transfers and the user tries to append something else to the worklist
        # We have to optimise and commit these auto transfers first, or they'll appear after whatever the
        # user appends.
        # Warn the user, then commit the auto transfers automatically
        if self.currently_optimising and not self.silence_append_warning:
            warnings.warn(
                "Modifying worklist after auto_transfer without commit. Auto transfers will be committed now, before your modification"
            )
            self.commit()
        super().append(*args, **kwargs)

    # evo_aspirate, dispense, and wash overrides that we will use interally when committing optimisations.
    # Avoid warning the user in this case
    def _evo_aspirate(self, *args, silence_append_warning=False, **kwargs):
        self.silence_append_warning = silence_append_warning
        super().evo_aspirate(*args, **kwargs)
        self.silence_append_warning = False

    def _evo_dispense(self, *args, silence_append_warning=False, **kwargs):
        self.silence_append_warning = silence_append_warning
        super().evo_dispense(*args, **kwargs)
        self.silence_append_warning = False

    def _evo_wash(self, *args, silence_append_warning=False, **kwargs):
        self.silence_append_warning = silence_append_warning
        super().evo_wash(*args, **kwargs)
        self.silence_append_warning = False

    def group_movments_needed(self, op_set, field):
        """
        Group operations by the specified field (source or destination),
        The column, and the liquid class. We group like this because these are the constraints
        To aspirating (or dispensing) something in one step
        """
        group_dict = {}
        for op in op_set:

            if field == "source":
                key = (op.source.name, op.source_pos[1], op.liquid_class)
            else:
                key = (op.destination.name, op.dest_pos[1], op.liquid_class)
            if key not in group_dict:
                group_dict[key] = []

            group_dict[key].append(op)

        return group_dict

    def group_by(self, open_ops, primary="source"):
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
        primary_dict = self.group_movments_needed(open_ops, primary)

        # Track the groupings we've found so far
        best_groupings = []

        # Loop through all the primary groups we've found
        for selected_ops in primary_dict.values():

            # Group by secondary
            # i.e. when primary=source, find all the destination (labware, column) pairs we need to dispense to
            secondary_labware_col = self.group_movments_needed(selected_ops, secondary)
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
                primary_rows_group = [primary_pos(op)[0] for op in secondary_op_group]
                secondary_rows_group = [
                    secondary_pos(op)[0] for op in secondary_op_group
                ]

                primary_rows_mask = "".join(
                    ["t" if i in primary_rows_group else "f" for i in range(8)]
                )
                secondary_rows_mask = "".join(
                    [
                        "t" if i in secondary_rows_group else "f"
                        for i in range(
                            min(secondary_rows_group), max(secondary_rows_group) + 1
                        )
                    ]
                )

                # If the destionation rows, represented in a string (like tft for rows [5,7])
                # Are a substring of the source rows (like fffftftf)
                # We can aspirate this group in one shot
                # If the source is a trough, the rows are flexible, so we don't need this check
                # We also can't repeat secondary rows (without requiring an additional pipetting step)
                if (
                    isinstance(selected_ops[0].source, Trough)
                    or secondary_rows_mask in primary_rows_mask
                ) and len(secondary_rows_group) == len(set(secondary_rows_group)):

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
                first_row = group[1].index("t")
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

                    # If the combo is bigger than 1 group, and the primary isn't a trough source,
                    # We need to make sure none of the rows in the primary are repeated
                    # As this would require more than one aspirate/dispense

                    # Check that we don't have a conflict in primary mask
                    # Get all indices in the primary mask for each group
                    tip_indices = [
                        (np.array(list(group[1])) == "t").nonzero()[0].tolist()
                        for group in combination
                    ]

                    # If primary is a trough source, just assign one tip per op
                    if primary == "source" and isinstance(
                        next(iter(combination[0][0])).source, Trough
                    ):
                        op_count = sum([len(group[0]) for group in combination])
                        all_tips = list(range(op_count))
                    else:
                        all_tips = [
                            tip for group_tips in tip_indices for tip in group_tips
                        ]

                    # If there are repeats, skip this combo
                    if len(set(all_tips)) != len(all_tips):
                        continue

                    # Check we still have enough tips with offset restrictions
                    # If we don't violate any constraints, we want to use tip 1 for the first row included, etc
                    primary_offset = -min(all_tips)
                    tips_for_combo = []
                    for i in range(len(combination)):
                        group = combination[i]
                        exemplar_op = next(iter(group[0]))

                        primary_limit = getattr(
                            primary_labware(exemplar_op), "offset_limit", None
                        )
                        secondary_limit = getattr(
                            secondary_labware(exemplar_op), "offset_limit", None
                        )

                        first_tip = min(tip_indices[i]) + primary_offset
                        secondary_offset = secondary_pos(exemplar_op)[0] - first_tip

                        def check_limit(limit, offset):
                            # Check if we have exceeded the offset limit allowed
                            if limit is None:
                                return 0
                            if limit >= 0:
                                return offset - limit
                            return limit - offset

                        primary_check = check_limit(primary_limit, primary_offset)
                        secondary_check = check_limit(secondary_limit, secondary_offset)

                        if primary_check > 0 or secondary_check > 0:
                            primary_offset -= max(primary_check, secondary_check)

                    # List the tips needed for this group, adjusted by the offset
                    # This will be used to assign tips for pipetting later
                    tips_for_combo = [tip + primary_offset for tip in all_tips]

                    # remove ops that need more than tip 8 after offset is applied
                    ops_to_cut = len([tip for tip in tips_for_combo if tip > 8])
                    cut_from_start = len([tip for tip in tips_for_combo if tip < 0])

                    tips_for_combo = tips_for_combo[cut_from_start:]
                    if ops_to_cut > 0:
                        tips_for_combo = tips_for_combo[:-ops_to_cut]

                    tips_needed = max(tips_for_combo) + 1

                    op_count = 0
                    combination_cut = []
                    for group, _, _ in combination:
                        group_ops = []
                        for op in group:
                            op_count += 1
                            if op_count <= cut_from_start:
                                continue
                            group_ops.append(op)

                            if (
                                op_count
                                >= (len(tips_for_combo) + cut_from_start) - ops_to_cut
                            ):
                                break
                        if len(group_ops) > 0:
                            combination_cut.append(group_ops)

                    # Track the maximum tips we've seen for a combo
                    most_tips_achieved = max(most_tips_achieved, tips_needed)
                    # Append this combo to the valid list
                    valid_combinations.append(
                        (tips_needed, combination_cut, tips_for_combo)
                    )

                # If we've already found a combo that uses all 8 tips, stop searching
                if most_tips_achieved == 8:
                    break

            # Sort by the number of tips used by this combo
            valid_combinations.sort(reverse=True)

            # Extract the list of groups from this combo
            selected_groups = [
                (sorted(list(ops), key=lambda op: secondary_pos(op)[0]))
                for ops in valid_combinations[0][1]
            ]
            # Extract the list of ops among all groups in this combo
            selected_ops = [op for group in selected_groups for op in list(group)]
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

    def group_ops(self):

        # Get open ops - the pending operations that don't have an unfulfilled dependency
        # Ie all ops we can select from at this time point
        open_ops = [
            op
            for op in self.pending_ops
            if op.source_dep not in self.pending_ops
            and op.dest_dep not in self.pending_ops
        ]

        open_ops.sort()

        # Get the best groups of ops when grouping by source,
        # and grouping by destination
        # See group_by method for more details
        best_groupings = self.group_by(open_ops, "source")
        best_groupings += self.group_by(open_ops, "destination")

        # Sort the groups by best cost, then earliest op, then the most ops achieved
        def group_sort_key(group):
            (steps, _, ops, _, _) = group
            # Cost is the number of steps (aspirates + dispenses) needed for the group,
            # divided by the number of ops (well transfers) achieved
            cost = steps / len(ops)
            return (cost, min([op.id for op in ops]), -1 * len(ops))

        best_groupings.sort(key=group_sort_key)
        return best_groupings

    def make_plan(self):

        # Count the aspirates, dispenses, and washes we use
        # for performance tracking
        asp_count = 0
        disp_count = 0
        wash_count = 0

        # Repeat until we have no more operations pending
        while len(self.pending_ops) > 0:
            # Group the operations. See group_ops method for details
            best_groupings = self.group_ops()
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

            # Track which group we're looking at
            index = 0
            # While we haven't checked every group, and haven't used all our tips
            while tips_used < 8 and index < len(best_groupings):

                # Get the contents of this group
                (steps, group_type, ops, target_groups, tips_selected) = best_groupings[
                    index
                ]
                index += 1

                # Check the ops in previously selected groups and new group are disjoint - ie don't try and do the same op twice
                if len(set(selected_ops + ops)) != len(selected_ops) + len(ops):
                    continue

                tips_needed = max(tips_selected) + 1

                # Initial check this group won't use too many tips
                if tips_used + tips_needed > 8:
                    continue

                # Check that the cost of adding this group (and saving washes)
                # Isn't greater than the cost of the second best group with additional washes
                cost_of_adding = steps / len(ops)
                if cost_of_adding >= second_best_cost:
                    break

                # If this puts us over 8 tips, skip this group
                if tips_used + tips_needed > 8:
                    continue

                for i in range(len(ops)):
                    # Convert to 1-indexed tips for robotools commands
                    ops[i].selected_tip = tips_used + tips_selected[i] + 1

                tips_used += tips_needed

                # If no conflicts have occurred, add this group to the selected groups and ops
                selected_groups.append((group_type, ops, target_groups))
                selected_ops += ops

            if len(selected_ops) == 0:
                raise Exception("Error: No valid groups to select")

            # Process the groups into a list of sources to aspirate and a list of destinations to dispense
            source_list = []
            dest_list = []
            for group_type, ops, target_groups in selected_groups:

                if group_type == "source":
                    # If we've grouped by source, we can aspirate all ops in the group at once
                    source_list += [ops]
                    # Destinations will depend on the subgroups selected, one for each subgroup
                    dest_list += [group for group in target_groups]

                else:
                    # If we've grouped by destination, we can dispense all ops in the group at once
                    dest_list += [ops]
                    # Sources will depend on the subgroups selected, one for each subgroup
                    source_list += [group for group in target_groups]

            # Sort the source list and dest list by the tips used in each subgroup
            # This just makes sure the pipetting occurs in an order that's less confusing visually,
            # starting from tip 1 to tip 8
            sort_tip_key = lambda x: min([op.selected_tip for op in x])
            source_list.sort(key=sort_tip_key)

            dest_list.sort(key=sort_tip_key)

            # Loop through the source list, aspirating for each group
            for source_group in source_list:

                # Get the first op in the group
                source_op = next(iter(source_group))
                # Get the col of this source group
                source_col = source_op.source_pos[1]
                # Get the rows we need to aspirate from the source_pos attribute of each op
                source_rows = [op.source_pos[0] for op in source_group]

                # Get the volumes stored in the ops
                volumes = [op.volume for op in source_group]
                # Get the tips assigned to the ops in the previous step
                tips = [op.selected_tip for op in source_group]

                # Check from earlier troubleshooting
                if len(tips) != len(set(tips)):
                    raise Exception("Error in tip logic")

                # If we have a trough, just set source rows to whichever tips we're using
                if isinstance(source_op.source, Trough):
                    source_rows = [tip - 1 for tip in tips]

                # Sort volumes by tips used
                tips_volumes = list(zip(tips, volumes))
                tips_volumes.sort()
                tips, volumes = zip(*tips_volumes)

                # Perform the aspiration
                self._evo_aspirate(
                    source_op.source,
                    source_op.source.wells[source_rows, source_col],
                    source_op.source.location,
                    list(tips),
                    list(volumes),
                    liquid_class=source_op.liquid_class,
                    label=" + ".join(set([op.label for op in source_group]))
                    + ", ops: "
                    + ",".join([str(op.id) for op in source_group])
                    + f", offset: {source_rows[0] - (list(tips)[0] - 1)}",
                    silence_append_warning=True,
                )
                asp_count += 1

            for dest_group in dest_list:
                # Get the first op in the group
                dest_op = next(iter(dest_group))
                # Get the col of this source group
                dest_col = dest_op.dest_pos[1]

                # Get the rows we need to dispense to for the group
                dest_rows = [op.dest_pos[0] for op in dest_group]

                # Get the volume stored in each op
                volumes = [op.volume for op in dest_group]

                # Get the tips assigned in previous steps
                tips = [op.selected_tip for op in dest_group]
                # Get the composition from the source labware for each op
                compositions = [
                    op.source.get_well_composition(op.source.wells[op.source_pos])
                    for op in dest_group
                ]

                # Sort volumes by tips
                tips_volumes = list(zip(tips, volumes))
                tips_volumes.sort()
                tips, volumes = zip(*tips_volumes)

                # Perform the dispense op
                self._evo_dispense(
                    dest_op.destination,
                    dest_op.destination.wells[dest_rows, dest_col],
                    dest_op.destination.location,
                    list(tips),
                    list(volumes),
                    liquid_class=dest_op.liquid_class,
                    label=" + ".join(set([op.label for op in dest_group]))
                    + ", ops: "
                    + ",".join([str(op.id) for op in dest_group])
                    + f", offset: {dest_rows[0] - (list(tips)[0] - 1)}",
                    compositions=compositions,
                    silence_append_warning=True,
                )

                disp_count += 1

            # Wash after this group of ops
            self._evo_wash(
                tips=[op.selected_tip for op in selected_ops],
                waste_location=self.waste_location,
                cleaner_location=self.cleaner_location,
                silence_append_warning=True,
            )

            # Line after each group just to make worklist easier to read
            super().append("B;")

            wash_count += 1

            # Update the completed and pending ops sets
            self.pending_ops.difference_update(selected_ops)
            self.completed_ops.update(selected_ops)

        print(f"aspirates: {asp_count}, dispenses: {disp_count}, washes: {wash_count}")

        return

    def commit(self):
        # List the ops before any optimising
        # For debugging purposes
        for op in sorted(self.pending_ops, key=lambda x: x.id):
            print(op)

        # If we have ops pending, optimise and apply them
        if len(self.pending_ops) > 0:
            self.make_plan()
            self.pending_ops = set()
            self.completed_ops = set()
        self.currently_optimising = False
        self.append("B;")

    def __exit__(self, *args):

        # Commit to optimise and apply any pending ops before exiting
        self.commit()

        super().__exit__(*args)
