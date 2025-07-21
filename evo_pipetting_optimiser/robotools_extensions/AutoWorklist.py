from robotools import *
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from . import *
import itertools
import warnings
from sortedcontainers import SortedSet
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
        wash_scheme: Literal[1, 2, 3, 4, "flush", "reuse"] = 1,
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
        self.wash_scheme = wash_scheme
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


def group_movments_needed(op_set, field, include_tip=False):
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


class AutoWorklist(EvoWorklist):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.completed_ops = set()
        self.pending_ops = set()

        self.tips_used = [False] * 8
        self.tip_contents = [None] * 8

        self.currently_optimising = False

    def auto_transfer(
        self,
        source: Union[AdvancedLabware, Trough],
        source_wells: Union[str, Sequence[str], np.ndarray],
        destination: AdvancedLabware,
        destination_wells: Union[str, Sequence[str], np.ndarray],
        volumes: Union[float, Sequence[float], np.ndarray],
        *,
        label: Optional[str] = None,
        wash_scheme: Literal[1, 2, 3, 4, "flush", "reuse"] = 1,
        on_underflow: Literal["debug", "warn", "raise"] = "raise",
        liquid_class: str = None,
        **kwargs,
    ) -> None:
        """Transfer operation between two labwares."""
        # reformat the convenience parameters
        source_wells = np.array(source_wells).flatten("F")
        destination_wells = np.array(destination_wells).flatten("F")
        volumes = np.array(volumes).flatten("F")
        nmax = max((len(source_wells), len(destination_wells), len(volumes)))

        # Deal with deprecated behavior
        if wash_scheme is None:
            warnings.warn(
                "wash_scheme=None is deprecated. For tip reuse pass 'reuse'.",
                DeprecationWarning,
                stacklevel=2,
            )
            wash_scheme = "reuse"

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

        # if isinstance(source, Trough):
        #     return

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
                wash_scheme=wash_scheme,
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

    def group_ops(self):
        open_dependencies = set(
            # [
            #     op.source_dep
            #     for op in self.pending_ops
            #     if op.source_dep and op.source_dep in self.pending_ops
            # ]
        )

        open_ops = [
            op
            for op in self.pending_ops
            if op.source_dep not in open_dependencies
            and op.source_dep not in self.pending_ops
            and op.dest_dep not in self.pending_ops
        ]

        open_ops.sort()

        source_dict = group_movments_needed(open_ops, "source")
        source_ops_consolidation = deque(source_dict.values())
        deque_added = False

        best_groupings = []

        while len(source_ops_consolidation) > 0:
            ops = source_ops_consolidation.popleft()
            deque_added = False
            selected_ops = []
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
                    selected_ops.append(op)
                    source_rows.append(row)
                    continue

                if deque_added:
                    source_ops_consolidation[0].append(op)
                else:
                    source_ops_consolidation.appendleft([op])
                    deque_added = True

                continue

            # Get the labware and columns needed among all the destinations
            dest_labware_col = group_movments_needed(selected_ops, "destination")
            dest_labware_col_queue = deque(dest_labware_col.values())

            dest_costs = []

            dest_labware_col_reachable = []

            trough_tip_tracker = 0

            while len(dest_labware_col_queue) > 0:
                # Calculate the number of pipetting steps needed to satisfy this group
                # It will be one step if the tips can line up from the source and the dest
                # Otherwise more

                dest_op_group = dest_labware_col_queue.popleft()

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

                if dest_rows_mask in source_rows_mask:

                    offset = source_rows_mask.index(dest_rows_mask)
                    for i, op in enumerate(dest_op_group):
                        if isinstance(selected_ops[0].source, Trough):
                            op.tip = i + 1 + trough_tip_tracker
                            op.source_pos = (trough_tip_tracker, op.source_pos[1])
                            trough_tip_tracker += 1
                        else:
                            op.tip = offset + i + 1
                    # This means that the destinations line up with the source rows
                    dest_labware_col_reachable.append(
                        (len(dest_op_group), dest_op_group)
                    )
                else:
                    # Otherwise, we can't pipette these destination rows in one step. Split up to the smaller available subsets
                    # And add back to the queue
                    for op_group in itertools.combinations(
                        dest_op_group, len(dest_op_group) - 1
                    ):
                        dest_labware_col_queue.append(op_group)

                dest_costs.append(len(dest_op_group))

            dest_labware_col_reachable.sort(reverse=True, key=lambda x: x[0])

            source_max = max(8, len(selected_ops))
            grouping_cost = (8 - source_max) + (8 - dest_labware_col_reachable[0][0])

            best_groupings.append(
                (
                    grouping_cost,
                    "source",
                    selected_ops,
                    dest_labware_col_reachable,
                )
            )

        def group_sort_key(group):
            (cost, _, ops, _) = group
            return (cost, -1 * len(ops))

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
                source_op = next(iter(ops))

                source_col = source_op.source_pos[1]
                source_rows = [op.source_pos[0] for op in ops]
                volumes = [op.volume for op in ops]

                tips = [op.tip for op in ops]

                if len(tips) != len(set(tips)):
                    raise Exception("Tip logic still not good oops")

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

                for _, target in target_groups:
                    target_op = next(iter(target))

                    target_col = target_op.dest_pos[1]

                    target_rows = [op.dest_pos[0] for op in target]

                    volumes = [op.volume for op in target]

                    tips = [op.tip for op in target]
                    compositions = [
                        op.source.get_well_composition(op.source.wells[op.source_pos])
                        for op in target
                    ]
                    self.evo_dispense(
                        target_op.destination,
                        target_op.destination.wells[target_rows, target_col],
                        (target_op.destination.grid, target_op.destination.site),
                        tips,
                        volumes,
                        liquid_class=target_op.liquid_class,
                        label=" + ".join(set([op.label for op in ops])),
                        compositions=compositions,
                    )

                    disp_count += 1

            self.pending_ops.difference_update(ops)
            self.completed_ops.update(ops)
            continue

        print(f"cost: {total_cost}, aspirates: {asp_count}, dispenses: {disp_count}")

        return

    def commit(self):
        # for op in sorted(self.pending_ops, key=lambda x: x.id):
        #     print(op)
        if len(self.pending_ops) > 0:
            self.make_plan()
            self.pending_ops = set()
            self.completed_ops = set()
            self.currently_optimising = False
        self.append("B;")

    def __exit__(self, *args):

        self.commit()

        super().__exit__(*args)
