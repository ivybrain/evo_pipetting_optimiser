from robotools import *
from typing import Literal, Optional, Tuple
from robotools.liquidhandling.composition import combine_composition
from . import AdvancedLabware


class TransferOperation:
    """
    Internal class that stores all the information needed
    For a transfer from one well to another
    """

    # Track ids of each op
    _op_id_counter = 0
    op_index = {}

    def __init__(
        self,
        source: AdvancedLabware,
        source_pos: Tuple[int, int],
        destination: AdvancedLabware,
        dest_pos: Tuple[int, int],
        volume: float,
        *,
        label: Optional[str] = None,
        on_underflow: Literal["debug", "warn", "raise"] = "raise",
        liquid_class=None,
        wash_scheme: Literal["D", 1, None, "smart"] = 1,
        tip_index=None,
        **kwargs,
    ):
        self.id = TransferOperation._op_id_counter
        TransferOperation._op_id_counter += 1
        TransferOperation.op_index = self

        self.source = source
        self.source_pos = source_pos
        self.destination = destination
        self.dest_pos = dest_pos

        self.volume = volume
        self.label = label
        self.on_underflow = on_underflow

        self.wash_scheme = wash_scheme

        self.liquid_class = liquid_class

        self.dep_children = []

        source_well = source.wells[source_pos]
        dest_well = destination.wells[dest_pos]

        self.source_dep = source.last_op[source_well]
        self.dest_dep = destination.last_op[dest_well]

        self.source_volume = source.latest_volume(source_well) - volume
        source_composition = source.latest_composition(source_well)

        self.dest_volume = destination.latest_volume(dest_well) + volume

        self.dest_composition = combine_composition(
            volume,
            source_composition,
            destination.volumes[dest_pos],
            destination.latest_composition(dest_well),
        )

        self.tip_index = tip_index

        return

    def __str__(self):
        source_label = f"{self.source.name}-{'any' if isinstance(self.source, Trough) else self.source.wells[self.source_pos]}"
        dest_label = f"{self.destination.name}-{self.destination.wells[self.dest_pos]}"
        return "{0:>3}: {1:>20} to {2:<20} {3:3.1f}ul\t{4}".format(
            self.id, source_label, dest_label, self.volume, self.label
        )

    def __repr__(self):
        return str(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id

    def __eq__(self, other):
        if not isinstance(other, TransferOperation):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
