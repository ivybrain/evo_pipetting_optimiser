from robotools import *
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from . import *
import itertools

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
        return f"{self.id}: {self.label}, Dep: {self.source_dep.id if self.source_dep else 'None'}, {self.dest_dep.id if self.dest_dep else 'None'}"

    def __repr__(self):
        return self.__str__()


class TransferNode:
    pass


class AutoWorklist(EvoWorklist):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.completed_ops = []
        self.pending_ops = []

        self.tips_used = [False] * 8
        self.tip_contents = [None] * 8

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

    def valid_moves(self):

        moves = []

        open_dependencies = self.open_dependencies

        # Dict of Tuple(source_id, source_col): [rows that can be aspirated]
        valid_sources = {}

        id_to_source = {}

        for op in self.pending_ops:
            # If this op depends on a source that hasn't been pipetted yet, can't conduct this op
            if op.source_dep in open_dependencies:
                continue

            id_to_source[id(op.source)] = op.source

            source_col = (id(op.source), op.source_pos[1])
            if source_col not in valid_sources:
                valid_sources[source_col] = []
            valid_sources[source_col].append(op.source_pos[0])

        for (source_id, col), rows in valid_sources.items():
            source = id_to_source[source_id]
            # print(source.name, col, rows)

            # If source is a trough, can aspirate any tips we want
            if isinstance(source, Trough):
                # How many tips we can aspirate from this trough. Max of 8 tips, or however many transfers we actually need from trough
                max_needed = min(8, len(rows))
                tip_use_values = [0, 1]
                tip_combos = itertools.product(tip_use_values, repeat=8)

                for combo in tip_combos:
                    if sum(combo) > max_needed:
                        continue
                    moves.append(("A", source.name, combo))

        return moves

    def make_plan(self):
        print(self.valid_moves())

    def commit(self):
        self.make_plan()
        self.append("B;")

    def __exit__(self, *args):
        for op in self.pending_ops:
            print(op)

        self.commit()

        super().__exit__(*args)
