from robotools import *
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from . import *
import itertools
import warnings

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

    def __str__(self):
        return f"{self.id}: {self.label} {self.source_pos} to {self.dest_pos}"

    def __repr__(self):
        return self.__str__()


class TransferNode:
    """
    Node to track the state of tips and operations in the planning search
    """

    def __init__(self, command, tip_ops, tip_used, cost, completed_ops, pending_ops):
        self.command = command
        self.tip_ops = tip_ops
        self.tip_used = tip_used
        self.cost = cost
        self.completed_ops = completed_ops.copy()
        self.pending_ops = pending_ops.copy()

        self.fscore = cost + self.heuristic()

    def heuristic(self):
        return (len(self.pending_ops) - len([op for op in self.tip_ops if op])) / 8

    def __str__(self):
        return f"({self.command}, {[op.id if op else 'x' for op in self.tip_ops]}, {self.cost}, {self.fscore}, {len(self.completed_ops)}, {len(self.pending_ops)})"

    def __repr__(self):
        return self.__str__()


class AutoWorklist(EvoWorklist):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.completed_ops = []
        self.pending_ops = []

        self.tips_used = [False] * 8
        self.tip_contents = [None] * 8

        self.currently_optimising = False

    @property
    def open_dependencies(self):
        """
        Pending transfers that are dependent on a particular op
        If any are still open, we can't do a further op onto the dependent well
        """
        return set([op.source_dep for op in self.pending_ops if op.source_dep])

    def auto_transfer(
        self,
        source: AdvancedLabware,
        source_wells: Union[str, Sequence[str], np.ndarray],
        destination: AdvancedLabware,
        destination_wells: Union[str, Sequence[str], np.ndarray],
        volumes: Union[float, Sequence[float], np.ndarray],
        *,
        label: Optional[str] = None,
        wash_scheme: Literal[1, 2, 3, 4, "flush", "reuse"] = 1,
        on_underflow: Literal["debug", "warn", "raise"] = "raise",
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
            )

            destination.op_tracking[destination_wells[i]].append(op)

            self.pending_ops.append(op)

    def transfer(self, *args, **kwargs):
        warnings.warn(
            "Using basic transfer after auto_transfer without commit. Auto transfers will be committed now"
        )
        self.commit()
        super().transfer(*args, **kwargs)

    def valid_moves(self, node):

        moves = []

        open_dependencies = self.open_dependencies

        # Dict of Tuple(source_id, source_col): [rows that can be aspirated]
        valid_sources = {}

        for op in self.pending_ops:
            # If this op depends on a source that hasn't been pipetted yet, can't conduct this op
            if op.source_dep in open_dependencies:
                continue

            # Group ops by source and col
            source_col = (id(op.source), op.source_pos[1])
            if source_col not in valid_sources:
                valid_sources[source_col] = []
            valid_sources[source_col].append(op)

        for (source_id, col), ops in valid_sources.items():
            source = ops[0].source

            ops.sort(key=lambda op: op.source_pos[0])

            # If source is a trough, we can aspirate from any and all virtual rows

            if isinstance(source, Trough):
                # Choose 8 ops from this source to assign to the tips. Any 8
                # ops_choices = list(itertools.product(*[ops for tip in range(8)]))
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
                    tip_ops = node.tip_ops.copy()
                    tip_used = node.tip_used.copy()
                    for tip, op in assignment:
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

        return moves

    def make_plan(self):
        initial_node = TransferNode(
            "_", [None] * 8, [False] * 8, 0, self.completed_ops, self.pending_ops
        )
        print("Initial", initial_node)
        moves = self.valid_moves(initial_node)
        print(moves)

    def commit(self):
        self.make_plan()
        self.append("B;")

    def __exit__(self, *args):

        for op in self.pending_ops:
            print(op)

        self.commit()

        super().__exit__(*args)
