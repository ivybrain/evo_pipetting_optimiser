from robotools import *
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from . import *
import itertools
import warnings
from collections import deque
import numpy as np


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

        self.liquid_class = liquid_class

    def __str__(self):
        return f"{self.id}: {self.label} {self.source_pos} to {self.dest_pos}"

    def __repr__(self):
        return str(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id


class AutoWorklist(EvoWorklist):

    def __init__(self, *args, wash_grid=None, wash_site=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.completed_ops = set()
        self.pending_ops = set()

        self.tips_used = [False] * 8
        self.tip_contents = [None] * 8

        self.currently_optimising = False

        assert wash_grid is not None, "Must define wash station grid"
        assert wash_site is not None, "Must define wash station site"

        self.wash_grid = wash_grid
        self.wash_site = wash_site

    def auto_transfer(
        self,
        source: Union[AdvancedLabware, Trough],
        source_wells: Union[str, Sequence[str], np.ndarray],
        destination: AdvancedLabware,
        destination_wells: Union[str, Sequence[str], np.ndarray],
        volumes: Union[float, Sequence[float], np.ndarray],
        *,
        label: Optional[str] = None,
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

    def transfer(self, *args, **kwargs):
        if self.currently_optimising:
            warnings.warn(
                "Using basic transfer after auto_transfer without commit. Auto transfers will be committed now"
            )
            self.commit()
        super().transfer(*args, **kwargs)

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
            elif field == "both":
                key = (
                    op.source.name,
                    op.source_pos[1],
                    op.destination.name,
                    op.dest_pos[1],
                    op.liquid_class,
                )
            else:
                key = (op.destination.name, op.dest_pos[1], op.liquid_class)
            if key not in group_dict:
                group_dict[key] = []

            if include_tip:
                group_dict[key].append((tip, op))
            else:
                group_dict[key].append(op)

        return group_dict

    def group_by_source(self, open_ops):
        """
        Group operations first by source (ie what can be aspirated in one step),
        Then by destination
        Along with a cost for each grouping
        """

        # Group available operations by source and source columnb
        # i.e. Group by what can be achieved in a single aspiration
        source_dict = self.group_movments_needed(open_ops, "source")

        # Queue to track the groups we have
        source_ops_consolidation = deque(source_dict.values())

        # Bool to track if we've added a new item to the queue while processing the current item
        deque_added = False

        # Track the best groupings we've found
        best_groupings = []

        # While we still have groups to process
        while len(source_ops_consolidation) > 0:
            # Get the operations that are part of this group
            ops = source_ops_consolidation.popleft()
            deque_added = False

            # Track the operations we can successfully perform in this group
            selected_ops = []

            # Track the rows needed for these operations
            source_rows = []

            for op in ops:
                row = op.source_pos[0]
                # If the row is already used in this op collection,
                # Pass it to the next collection
                # Because we can't pipette twice the same row in one operation
                # Unless the source is a trough, in which case we say we can pipette from 'row 0' up to 8 times
                if row not in source_rows or (
                    isinstance(op.source, Trough) and len(selected_ops) < 8
                ):
                    # If we don't have a row conflict, select this op
                    selected_ops.append(op)
                    source_rows.append(row)
                    continue

                # If we do have a row conflict, add the conflicting op to a new collection at the front of the queue
                if deque_added:
                    source_ops_consolidation[0].append(op)
                else:
                    # If we've already created such a new collection, add it ot that instead
                    source_ops_consolidation.appendleft([op])
                    deque_added = True

            # Get the labware and columns needed among all the destinations
            dest_labware_col = self.group_movments_needed(selected_ops, "destination")
            dest_labware_col_queue = deque(dest_labware_col.values())

            dest_costs = []

            # Track operation sets that are confirmed to be reachable in one dispense
            dest_labware_col_reachable = []

            # Track the tip we're up to if aspirating from a trough
            trough_tip_tracker = 0

            # Process each destination group of labware, column
            while len(dest_labware_col_queue) > 0:
                # Calculate the number of pipetting steps needed to satisfy this group
                # It will be one step if the tips can line up from the source and the dest
                # Otherwise more

                dest_op_group = dest_labware_col_queue.popleft()

                # Get the rows needed among the source and the destination
                source_rows_group = [op.source_pos[0] for op in dest_op_group]
                dest_rows_group = [op.dest_pos[0] for op in dest_op_group]

                source_rows_mask = "".join(
                    ["t" if i in source_rows_group else "f" for i in range(8)]
                )
                dest_rows_mask = "".join(
                    [
                        "t" if i in dest_rows_group else "f"
                        for i in range(min(dest_rows_group), max(dest_rows_group) + 1)
                    ]
                )

                # If the destionation rows, represented in a string (like tft for rows [5,7])
                # Are a substring of the source rows (like fffftftf)
                # We can aspirate this group in one shot
                if (
                    isinstance(selected_ops[0].source, Trough)
                    or dest_rows_mask in source_rows_mask
                ):

                    # Get the position of the destination mask within the source to calculate tip offset

                    # Calculate the tip needed for labware or trough
                    for i, op in enumerate(dest_op_group):
                        if isinstance(selected_ops[0].source, Trough):
                            op.source_tip = 1 + trough_tip_tracker
                            # op.source_pos = (trough_tip_tracker, op.source_pos[1])
                            trough_tip_tracker += 1
                        else:
                            offset = source_rows_mask.index(dest_rows_mask)
                            op.source_tip = offset + i + 1
                    # This means that the destinations line up with the source rows
                    dest_labware_col_reachable.append(dest_op_group)
                else:
                    # Otherwise, we can't pipette these destination rows in one step. Split up to the smaller available subsets
                    # And add back to the queue
                    for op_group in itertools.combinations(
                        dest_op_group, len(dest_op_group) - 1
                    ):
                        dest_labware_col_queue.append(op_group)

                dest_costs.append(len(dest_op_group))

            dest_labware_col_reachable.sort(reverse=True, key=lambda x: len(x))

            # Caclulate a cost for this group of operations
            # Ideal situation (cost 0) would be a single aspirate with 8 operations,
            # And a single dispense with the same 8 operations

            # Thus, the cost is how far we are from this ideal (8 - number_aspirated) + (8 - number_dispensed)

            steps_for_primary = 1
            steps_for_seccondary = len(dest_labware_col_reachable)
            ops_achieved = len(selected_ops)

            grouping_cost = (steps_for_primary + steps_for_seccondary) / ops_achieved
            # Add this group and its cost to the list
            best_groupings.append(
                (
                    grouping_cost,
                    "source",
                    selected_ops,
                    dest_labware_col_reachable,
                )
            )

        return best_groupings

    def group_by_dest(self, open_ops):
        # Group available operations by source and source columnb
        # i.e. Group by what can be achieved in a single aspiration
        primary_dict = self.group_movments_needed(open_ops, "destination")

        # Queue to track the groups we have
        primary_ops_consolidation = deque(primary_dict.values())

        # Bool to track if we've added a new item to the queue while processing the current item
        deque_added = False

        # Track the best groupings we've found
        best_groupings = []

        # While we still have groups to process
        while len(primary_ops_consolidation) > 0:
            # Get the operations that are part of this group
            ops = primary_ops_consolidation.popleft()
            deque_added = False

            # Track the operations we can successfully perform in this group
            selected_ops = []

            # Track the rows needed for these operations
            primary_rows = []

            for op in ops:
                row = op.dest_pos[0]
                # If the row is already used in this op collection,
                # Pass it to the next collection
                # Because we can't pipette twice the same row in one operation
                # Unless the source is a trough, in which case we say we can pipette from 'row 0' up to 8 times
                if row not in primary_rows:
                    # If we don't have a row conflict, select this op
                    selected_ops.append(op)
                    primary_rows.append(row)
                    continue

                # If we do have a row conflict, add the conflicting op to a new collection at the front of the queue
                if deque_added:
                    primary_ops_consolidation[0].append(op)
                else:
                    # If we've already created such a new collection, add it ot that instead
                    primary_ops_consolidation.appendleft([op])
                    deque_added = True

            # Get the labware and columns needed among all the destinations
            seccondary_labware_col = self.group_movments_needed(selected_ops, "source")
            seccondary_labware_col_queue = deque(seccondary_labware_col.values())

            seccondary_costs = []

            # Track operation sets that are confirmed to be reachable in one dispense
            seccondary_labware_col_reachable = []

            # Process each destination group of labware, column
            while len(seccondary_labware_col_queue) > 0:
                # Calculate the number of pipetting steps needed to satisfy this group
                # It will be one step if the tips can line up from the source and the dest
                # Otherwise more
                trough_tip_tracker = 0
                seccondary_op_group = seccondary_labware_col_queue.popleft()

                # Get the rows needed among the source and the destination
                primary_rows_group = [op.dest_pos[0] for op in seccondary_op_group]
                seccondary_rows_group = [op.source_pos[0] for op in seccondary_op_group]

                primary_rows_mask = "".join(
                    ["t" if i in primary_rows_group else "f" for i in range(8)]
                )
                seccondary_rows_mask = "".join(
                    [
                        "t" if i in seccondary_rows_group else "f"
                        for i in range(
                            min(seccondary_rows_group), max(seccondary_rows_group) + 1
                        )
                    ]
                )

                # If the destionation rows, represented in a string (like tft for rows [5,7])
                # Are a substring of the source rows (like fffftftf)
                # We can aspirate this group in one shot
                if (
                    isinstance(selected_ops[0].source, Trough)
                    or seccondary_rows_mask in primary_rows_mask
                ):

                    for i, op in enumerate(seccondary_op_group):
                        if isinstance(selected_ops[0].source, Trough):
                            op.dest_tip = 1 + trough_tip_tracker
                            # op.source_pos = (trough_tip_tracker, op.source_pos[1])
                            trough_tip_tracker += 1
                        else:
                            offset = primary_rows_mask.index(seccondary_rows_mask)
                            op.dest_tip = offset + i + 1
                    # This means that the destinations line up with the source rows
                    seccondary_labware_col_reachable.append(seccondary_op_group)
                else:
                    # Otherwise, we can't pipette these destination rows in one step. Split up to the smaller available subsets
                    # And add back to the queue
                    for op_group in itertools.combinations(
                        seccondary_op_group, len(seccondary_op_group) - 1
                    ):
                        seccondary_labware_col_queue.append(op_group)

                seccondary_costs.append(len(seccondary_op_group))

            seccondary_labware_col_reachable.sort(reverse=True, key=lambda x: len(x))

            # Caclulate a cost for this group of operations
            # Ideal situation (cost 0) would be a single aspirate with 8 operations,
            # And a single dispense with the same 8 operations

            # Thus, the cost is how far we are from this ideal (8 - number_aspirated) + (8 - number_dispensed)

            steps_for_primary = 1
            steps_for_seccondary = len(seccondary_labware_col_reachable)
            ops_achieved = len(selected_ops)

            grouping_cost = (steps_for_primary + steps_for_seccondary) / ops_achieved

            # Add this group and its cost to the list
            best_groupings.append(
                (
                    grouping_cost,
                    "dest",
                    selected_ops,
                    seccondary_labware_col_reachable,
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

        best_groupings = self.group_by_source(open_ops)
        best_groupings += self.group_by_dest(open_ops)

        def group_sort_key(group):
            (cost, _, ops, _) = group
            # With equal cost, bias those with earlier op id
            # To keep things in a more understandable order
            return (cost, min([op.id for op in ops]) - 1 * len(ops))

        # Sort the groups by their cost
        best_groupings.sort(key=group_sort_key)
        return best_groupings

    def make_plan(self):
        # initial_node = TransferNode(
        #     "_", None, [None] * 8, [False] * 8, 0, self.completed_ops, self.pending_ops
        # )
        # print("Initial", initial_node)

        # plan = self.astar(initial_node)

        # return plan

        total_cost = 0
        asp_count = 0
        disp_count = 0
        while len(self.pending_ops) > 0:
            best_groupings = self.group_ops()
            (cost, group_type, ops, target_groups) = best_groupings[0]
            total_cost += cost

            if group_type == "source":
                source_list = [ops]
                dest_list = target_groups
            else:
                source_list = target_groups
                dest_list = [ops]

            for source_group in source_list:
                source_op = next(iter(source_group))
                source_col = source_op.source_pos[1]
                source_rows = [op.source_pos[0] for op in source_group]
                volumes = [op.volume for op in source_group]

                tips = [
                    op.source_tip if group_type == "source" else op.dest_tip
                    for op in source_group
                ]

                if len(tips) != len(set(tips)):
                    raise Exception("Error in tip logic")

                self.evo_aspirate(
                    source_op.source,
                    source_op.source.wells[source_rows, source_col],
                    (source_op.source.grid, source_op.source.site),
                    tips,
                    volumes,
                    liquid_class=source_op.liquid_class,
                    label=" + ".join(set([op.label for op in ops])),
                )
                asp_count += 1

            for dest_group in dest_list:
                dest_op = next(iter(dest_group))
                dest_col = dest_op.dest_pos[1]

                dest_rows = [op.dest_pos[0] for op in dest_group]

                volumes = [op.volume for op in dest_group]

                tips = [
                    op.source_tip if group_type == "source" else op.dest_tip
                    for op in dest_group
                ]
                compositions = [
                    op.source.get_well_composition(op.source.wells[op.source_pos])
                    for op in dest_group
                ]
                self.evo_dispense(
                    dest_op.destination,
                    dest_op.destination.wells[dest_rows, dest_col],
                    (dest_op.destination.grid, dest_op.destination.site),
                    tips,
                    volumes,
                    liquid_class=dest_op.liquid_class,
                    label=" + ".join(set([op.label for op in ops])),
                    compositions=compositions,
                )

                disp_count += 1

            # Update the completed and pending ops sets
            self.pending_ops.difference_update(ops)
            self.completed_ops.update(ops)

        print(f"cost: {total_cost}, aspirates: {asp_count}, dispenses: {disp_count}")

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
