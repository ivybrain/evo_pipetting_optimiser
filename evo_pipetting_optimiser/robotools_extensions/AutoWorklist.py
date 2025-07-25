from robotools import *
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from . import *
import itertools
import warnings
from collections import deque
import numpy as np
from functools import reduce


class TransferOperation:

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
        """Transfer operation between two labwares."""
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

        self.currently_optimising = True

        # source wells don't matter for a trough, set them all 0
        # then optimiser can arrange as needed
        if isinstance(source, Trough):
            source_wells = ["A01"] * len(source_wells)

        # Append this op to all of the destination wells we touch
        for i in range(len(source_wells)):
            op = TransferOperation(
                source,
                source.indices[source_wells[i]],
                destination,
                destination.indices[destination_wells[i]],
                volumes[i],
                label=label,
                wash=wash,
                on_underflow=on_underflow,
                source_dep=(
                    source.last_op[source_wells[i]]
                    if isinstance(source, AdvancedLabware)
                    else None
                ),
                dest_dep=destination.last_op[destination_wells[i]],
                liquid_class=liquid_class,
            )

            destination.op_tracking[destination_wells[i]].append(op)

            self.pending_ops.update([op])

    def append(self, *args, **kwargs):
        if self.currently_optimising and not self.silence_append_warning:
            warnings.warn(
                "Modifying worklist after auto_transfer without commit. Auto transfers will be committed now, before your modification"
            )
            self.commit()
        super().append(*args, **kwargs)

    def evo_aspirate(self, *args, silence_append_warning=False, **kwargs):
        self.silence_append_warning = silence_append_warning
        super().evo_aspirate(*args, **kwargs)
        self.silence_append_warning = False

    def evo_dispense(self, *args, silence_append_warning=False, **kwargs):
        self.silence_append_warning = silence_append_warning
        super().evo_dispense(*args, **kwargs)
        self.silence_append_warning = False

    def evo_wash(self, *args, silence_append_warning=False, **kwargs):
        self.silence_append_warning = silence_append_warning
        super().evo_wash(*args, **kwargs)
        self.silence_append_warning = False

    def group_movments_needed(self, op_set, field, include_tip=False):
        """
        Group operations by the specified field (source or destination),
        The column, and the liquid class
        This effectively provides a minimum number of movements needed to
        complete these ops
        """
        group_dict = {}
        for tip, op in enumerate(op_set):
            if not op:
                continue

            if field == "source":
                key = (op.source.name, op.source_pos[1], op.liquid_class)
            else:
                key = (op.destination.name, op.dest_pos[1], op.liquid_class)
            if key not in group_dict:
                group_dict[key] = []

            if include_tip:
                group_dict[key].append((tip, op))
            else:
                group_dict[key].append(op)

        return group_dict

    def group_by(self, open_ops, primary="source"):

        if primary == "source":
            primary_pos = lambda op: op.source_pos
            secondary_pos = lambda op: op.dest_pos
            secondary = "destination"
        elif primary == "destination":
            primary_pos = lambda op: op.dest_pos
            secondary_pos = lambda op: op.source_pos
            secondary = "source"

        # Group available operations by source and source column
        # i.e. Group by what can be achieved in a single aspiration
        primary_dict = self.group_movments_needed(open_ops, primary)

        # Track the best groupings we've found
        best_groupings = []

        # While we still have groups to process
        for selected_ops in primary_dict.values():

            # Get the labware and columns needed among all the destinations
            secondary_labware_col = self.group_movments_needed(selected_ops, secondary)
            secondary_labware_col_queue = deque(
                [set(x) for x in secondary_labware_col.values()]
            )

            secondary_costs = []

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
                if (
                    isinstance(selected_ops[0].source, Trough)
                    or secondary_rows_mask in primary_rows_mask
                ):

                    # This means that the destinations line up with the source rows
                    secondary_labware_col_reachable.append(
                        (secondary_op_group, primary_rows_mask, secondary_rows_mask)
                    )
                else:
                    # Otherwise, we can't pipette these destination rows in one step. Split up to the smaller available subsets
                    # And add back to the queue
                    for op_group in itertools.combinations(
                        secondary_op_group, len(secondary_op_group) - 1
                    ):
                        if set(op_group) in secondary_labware_col_queue:
                            continue
                        secondary_labware_col_queue.append(set(op_group))

                secondary_costs.append(len(secondary_op_group))

            # Sort by biggest secondary groups
            def group_sort_key(group):
                # Want to sort by biggest
                size = -1 * len(group[0])
                first_tip = group[1].index("t")
                return (size, first_tip)

            secondary_labware_col_reachable.sort(key=group_sort_key)

            # All combinations of the groups
            valid_combinations = []
            most_tips_achieved = 0

            # Efficiency could be drastically improved with dynamic programming
            for combo_size in range(1, 8):
                for combination in itertools.combinations(
                    secondary_labware_col_reachable, combo_size
                ):
                    tips_needed = sum([len(group[2]) for group in combination])
                    # Check that we don't need too many tips
                    if tips_needed > 8:
                        continue

                    if combo_size > 1 and (
                        primary != "source"
                        or not isinstance(selected_ops[0].source, Trough)
                    ):
                        # Check that we don't have a conflict in primary mask
                        tip_indices = [
                            (np.array(list(group[1])) == "t").nonzero()[0].tolist()
                            for group in combination
                        ]
                        all_tips = [
                            tip for group_tips in tip_indices for tip in group_tips
                        ]
                        if len(set(all_tips)) != len(all_tips):
                            continue

                    most_tips_achieved = max(most_tips_achieved, tips_needed)
                    valid_combinations.append((tips_needed, combination))

                if most_tips_achieved == 8:
                    break

            valid_combinations.sort(reverse=True)
            # Now, we need to rationalise this group and select the best 8 ops we can do with our tips

            selected_groups = [
                (
                    sorted(list(ops), key=lambda op: secondary_pos(op)[0]),
                    primary_mask,
                    secondary_mask,
                )
                for (ops, primary_mask, secondary_mask) in valid_combinations[0][1]
            ]
            selected_ops = [op for group in selected_groups for op in list(group[0])]

            # Caclulate a cost for this group of operations
            # Ideal situation (cost 0) would be a single aspirate with 8 operations,
            # And a single dispense with the same 8 operations

            # Thus, the cost is how far we are from this ideal (8 - number_aspirated) + (8 - number_dispensed)

            steps_for_primary = 1
            steps_for_secondary = len(selected_groups)
            ops_achieved = len(selected_ops)

            grouping_cost = (steps_for_primary + steps_for_secondary) / ops_achieved

            # Add this group and its cost to the list
            best_groupings.append(
                (
                    grouping_cost,
                    primary,
                    selected_ops,
                    selected_groups,
                )
            )

        return best_groupings

    def group_ops(self):

        open_ops = [
            op
            for op in self.pending_ops
            if op.source_dep not in self.pending_ops
            and op.dest_dep not in self.pending_ops
        ]

        open_ops.sort()

        best_groupings = self.group_by(open_ops, "source")
        best_groupings += self.group_by(open_ops, "destination")

        def group_sort_key(group):
            (cost, _, ops, _) = group
            # With equal cost, bias those with earlier op id
            # To keep things in a more understandable order
            return (cost, min([op.id for op in ops]) - 1 * len(ops))

        # Sort the groups by their cost
        best_groupings.sort(key=group_sort_key)
        return best_groupings

    def make_plan(self):

        total_cost = 0
        asp_count = 0
        disp_count = 0
        wash_count = 0
        while len(self.pending_ops) > 0:
            best_groupings = self.group_ops()
            (cost, group_type, ops, target_groups) = best_groupings[0]
            total_cost += cost

            if group_type == "source":
                source_list = [ops]
                dest_list = [group[0] for group in target_groups]

                if isinstance(ops[0].source, Trough):
                    tip_counter = 0
                    for group, _, _ in target_groups:
                        for op in group:
                            op.selected_tip = tip_counter + 1
                            tip_counter += 1

            else:
                source_list = [group[0] for group in target_groups]
                dest_list = [ops]

            sort_tip_key = lambda x: min([op.selected_tip for op in x])
            source_list.sort(key=sort_tip_key)

            dest_list.sort(key=sort_tip_key)

            for source_group in source_list:

                source_op = next(iter(source_group))
                source_col = source_op.source_pos[1]
                source_rows = [op.source_pos[0] for op in source_group]

                volumes = [op.volume for op in source_group]

                tips = [op.selected_tip for op in source_group]

                if len(tips) != len(set(tips)):
                    raise Exception("Error in tip logic")

                # If we have a trough, source rows match the tips we have
                if isinstance(source_op.source, Trough):
                    source_rows = [tip - 1 for tip in tips]

                # Sort volumes by tips
                tips_volumes = list(zip(tips, volumes))
                tips_volumes.sort()
                tips, volumes = zip(*tips_volumes)

                self.evo_aspirate(
                    source_op.source,
                    source_op.source.wells[source_rows, source_col],
                    (source_op.source.grid, source_op.source.site),
                    list(tips),
                    list(volumes),
                    liquid_class=source_op.liquid_class,
                    label=" + ".join(set([op.label for op in source_group]))
                    + ", ops: "
                    + ",".join([str(op.id) for op in source_group]),
                    silence_append_warning=True,
                )
                asp_count += 1

            for dest_group in dest_list:
                dest_op = next(iter(dest_group))
                dest_col = dest_op.dest_pos[1]

                dest_rows = [op.dest_pos[0] for op in dest_group]

                volumes = [op.volume for op in dest_group]

                tips = [op.selected_tip for op in dest_group]
                compositions = [
                    op.source.get_well_composition(op.source.wells[op.source_pos])
                    for op in dest_group
                ]

                # Sort volumes by tips
                tips_volumes = list(zip(tips, volumes))
                tips_volumes.sort()
                tips, volumes = zip(*tips_volumes)

                self.evo_dispense(
                    dest_op.destination,
                    dest_op.destination.wells[dest_rows, dest_col],
                    (dest_op.destination.grid, dest_op.destination.site),
                    list(tips),
                    list(volumes),
                    liquid_class=dest_op.liquid_class,
                    label=" + ".join(set([op.label for op in dest_group]))
                    + ", ops: "
                    + ",".join([str(op.id) for op in dest_group]),
                    compositions=compositions,
                    silence_append_warning=True,
                )

                disp_count += 1

            # Wash after this group of ops
            self.evo_wash(
                tips=[op.selected_tip for op in source_group],
                waste_location=self.waste_location,
                cleaner_location=self.cleaner_location,
                silence_append_warning=True,
            )

            # Line after each group just to make worklist easier to read
            super().append("B;")

            wash_count += 1

            # Update the completed and pending ops sets
            self.pending_ops.difference_update(ops)
            self.completed_ops.update(ops)

        print(
            f"cost: {total_cost}, aspirates: {asp_count}, dispenses: {disp_count}, washes: {wash_count}"
        )

        return

    def commit(self):
        for op in sorted(self.pending_ops, key=lambda x: x.id):
            print(op)

        if len(self.pending_ops) > 0:
            self.make_plan()
            self.pending_ops = set()
            self.completed_ops = set()
        self.currently_optimising = False
        self.append("B;")

    def __exit__(self, *args):

        self.commit()

        super().__exit__(*args)
