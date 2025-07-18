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


class TransferNode:
    """
    Node to track the state of tips and operations in the planning search
    """

    def __init__(
        self,
        command,
        parent,
        tip_ops,
        tip_used,
        parent_cost,
        completed_ops,
        pending_ops,
    ):
        self.parent = parent
        self.command = command
        self.tip_ops = tip_ops
        self.tip_used = tip_used
        self.completed_ops = completed_ops.copy()
        self.pending_ops = pending_ops.copy()

        # Dictionary of (destination labware, column) : (tip number, operation)
        dests_cols = group_movments_needed(
            self.tip_ops, "destination", include_tip=True
        )

        self.required_dispenses = []

        for (_, col, _), tips_ops in dests_cols.items():
            dest = tips_ops[0][1].destination
            fulfilled_ops = []
            offset = None

            for tip, op in tips_ops:
                # If this is the first op, or if we have another op that doesn't line up with the existing chosen offset
                # We need a new offset, and a new operation
                if offset == None or tip + offset != op.dest_pos[0]:
                    # The offset required to reach the row in the first op
                    offset = op.dest_pos[0] - tip
                    fulfilled_ops = []
                    fulfilled_ops.append(op)
                    self.required_dispenses.append(
                        (dest.name, col, offset, fulfilled_ops)
                    )
                    continue

                # If the op is already possible with the same offset (the tip is lined up, add that to this dispense)
                if tip + offset == op.dest_pos[0]:
                    fulfilled_ops.append(op)

        # Number of ops we have in the tips currently
        selected_ops = [op for op in self.tip_ops if op]

        # Add selected ops to completed
        self.completed_ops.update(selected_ops)
        self.pending_ops.difference_update(selected_ops)

        # Cost is partent cost + 1 aspirate + the number of dispenses we require
        self.cost = parent_cost + len(self.required_dispenses) + 1

        self.cols_needed = group_movments_needed(self.pending_ops, "both")

        heuristic = len(self.cols_needed) - (
            len(self.completed_ops) / (len(self.completed_ops) + len(self.pending_ops))
        )

        self.fscore = self.cost + 2 * len(self.pending_ops)

    @property
    def open_dependencies(self):
        """
        Pending transfers that are dependent on a particular op
        If any are still open, we can't do a further op onto the dependent well
        """
        return set(
            [
                op.source_dep
                for op in self.pending_ops
                if op.source_dep and op.source_dep in self.pending_ops
            ]
        )

    def __str__(self):
        return f"({self.command}, {[op.id if op else 'x' for op in self.tip_ops]}, {self.cost}, {len(self.required_dispenses)}, {self.fscore}, {len(self.completed_ops)}, {len(self.pending_ops)})"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, b):
        return self.fscore < b.fscore

    def __gt__(self, b):
        return self.fscore > b.fscore

    def __eq__(self, other):

        return (
            self.completed_ops == other.completed_ops
            and self.pending_ops == other.pending_ops
            and self.fscore == other.fscore
        )

    def __hash__(self):
        return hash(
            f"C{str(self.completed_ops)};P{str(self.pending_ops)};F{self.fscore}"
        )


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

        assert (
            isinstance(source, AdvancedLabware) or isinstance(source, Trough),
            "Source must be AdvancedLabware or Trough for auto_transfer",
        )

        assert (
            isinstance(destination, AdvancedLabware),
            "Destination must be AdvancedLabware for auto_transfer",
        )

        assert (
            liquid_class is not None,
            "Liquid class must be speicified for auto_transfer",
        )

        # SKIP TROUGHS FOR NOW
        if isinstance(source, Trough):
            return

        self.comment(label)

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
        warnings.warn(
            "Using basic transfer after auto_transfer without commit. Auto transfers will be committed now"
        )
        self.commit()
        super().transfer(*args, **kwargs)

    def valid_moves(self, node):

        moves = []

        open_dependencies = node.open_dependencies

        # Dict of Tuple(source_id, source_col): [rows that can be aspirated]
        valid_sources = {}

        for op in node.pending_ops:
            # If this op depends on a source that hasn't been pipetted yet, can't conduct this op
            if op.source_dep in open_dependencies:
                continue

            # Group ops by source and col and liquid class (as can only aspirate or disp one liquid class at a time with
            # advanced worklist commands
            source_col = (id(op.source), op.source_pos[1], op.liquid_class)
            if source_col not in valid_sources:
                valid_sources[source_col] = []
            valid_sources[source_col].append(op)

        for (source_id, col, _), ops in valid_sources.items():
            source = ops[0].source

            ops.sort(key=lambda op: op.source_pos[0])

            # If source is a trough, we can aspirate from any and all virtual rows

            if isinstance(source, Trough):
                continue
                # Choose 8 ops from this source to assign to the tips. Any 8
                ops_choices = list(itertools.combinations(ops, 8))

                for ops in ops_choices:
                    tip_ops = node.tip_ops.copy()
                    tip_used = node.tip_used.copy()
                    for tip, op in enumerate(ops):
                        tip_ops[tip] = op
                        tip_used[tip] = True
                    moves.append(
                        TransferNode(
                            "A",
                            tip_ops,
                            tip_used,
                            1,
                            self.completed_ops,
                            self.pending_ops,
                        )
                    )

                continue

            # Possible moves aspirating from an AdvancedLabware source

            # Get the unique rows we can aspirate from for all ops, and the ops that can proceed from that row
            ops_at_row = {}
            for row, ops in itertools.groupby(ops, lambda op: op.source_pos[0]):
                ops_at_row[row] = list(ops)

            # print(
            #     source.name,
            #     col,
            #     {
            #         f"Row {row}: {[op.id for op in ops]}"
            #         for row, ops in ops_at_row.items()
            #     },
            # )

            # Get the possible assignments of tips to rows
            # Don't take into account spreading for now
            # We get with all 8 tips to all 8 rows, or offset by -7 (upwards, tip 8 goes to row A)
            # or +7 (downwards, tip 1 goes to row H)

            for offset in range(-7, 8):
                tip_at_row_0 = 0 - offset

                # Iterate through the pairs of tips and rows that the tips are at, at this offset
                combo = []
                tips = []
                for row, tip in enumerate(range(tip_at_row_0, 8)):
                    if tip < 0:
                        continue
                    if row > 7:
                        break

                    if row not in ops_at_row:
                        continue
                    combo.append((tip, row))
                    tips.append(tip)

                if len(combo) == 0:
                    continue
                if len(combo) == 1:
                    tip, row = combo[0]
                    tip_op_combos = [[(tip, op)] for op in ops_at_row[row]]
                else:

                    tip_op_combos = list(
                        itertools.product(*[ops_at_row[row] for (tip, row) in combo]),
                    )

                    tip_op_combos = [
                        list(zip(tips, op_assignments))
                        for op_assignments in tip_op_combos
                    ]

                for assignment in tip_op_combos:
                    tip_ops = [None] * 8
                    tip_used = [False] * 8
                    for tip, op in assignment:
                        tip_ops[tip] = op
                        tip_used[tip] = True
                    moves.append(
                        TransferNode(
                            "A",
                            node,
                            tip_ops,
                            tip_used,
                            node.cost,
                            node.completed_ops,
                            node.pending_ops,
                        )
                    )

        return moves

    def group_ops(self):
        open_dependencies = set(
            [
                op.source_dep
                for op in self.pending_ops
                if op.source_dep and op.source_dep in self.pending_ops
            ]
        )

        open_ops = [
            op for op in self.pending_ops if op.source_dep not in open_dependencies
        ]

        source_dict = group_movments_needed(open_ops, "source")
        source_ops_consolidation = deque(source_dict.values())
        deque_added = False

        best_groupings = []

        while len(source_ops_consolidation) > 0:
            ops = source_ops_consolidation.popleft()
            deque_added = False
            source_rows = {}
            for op in ops:
                row = op.source_pos[0]
                # If the row is already used in this op collection,
                # Pass it to the next collection
                # Because we can't pipette twice the same row in one operation
                if row in source_rows:
                    if deque_added:
                        source_ops_consolidation[0].append(op)
                    else:
                        source_ops_consolidation.appendleft([op])
                        deque_added = True

                    continue

                source_rows[row] = op

            # Get the labware and columns needed among all the destinations
            dest_labware_col = group_movments_needed(ops, "destination")
            dest_labware_col_queue = deque(dest_labware_col.values())

            dest_costs = []

            dest_labware_col_reachable = []

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

            source_max = max(8, len(source_rows))
            grouping_cost = (8 - source_max) + (8 - dest_labware_col_reachable[0][0])

            best_groupings.append(
                (
                    grouping_cost,
                    "source",
                    set(source_rows.values()),
                )
            )

        def group_sort_key(group):
            (cost, _, ops) = group
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

        while len(self.pending_ops) > 0:
            best_groupings = self.group_ops()
            (cost, _, ops) = best_groupings[0]
            total_cost += cost

            self.pending_ops.difference_update(ops)
            self.completed_ops.update(ops)
            continue

        return

    def astar(self, initial_node):
        self.open_nodes = SortedSet()
        self.open_nodes.add(initial_node)

        plan = []
        counter = 0
        while len(self.open_nodes) > 0:
            node = self.open_nodes.pop(0)

            print(counter, len(self.open_nodes))

            # If no more ops are pending, we're done
            if len(node.pending_ops) == 0 or counter >= 20:

                plan.append(node)
                next_node = node.parent
                while next_node:
                    plan.append(next_node)
                    next_node = next_node.parent

                break

            moves = self.valid_moves(node)
            moves.sort()
            self.open_nodes = self.open_nodes.union(moves)

            counter += 1

        plan.reverse()

        print(counter)

        return plan

    def commit(self):
        self.make_plan()
        self.append("B;")

    def __exit__(self, *args):

        for op in sorted(self.pending_ops, key=lambda x: x.id):
            print(op)

        self.commit()

        super().__exit__(*args)
