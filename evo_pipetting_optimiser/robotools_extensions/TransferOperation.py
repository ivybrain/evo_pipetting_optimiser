from robotools import *
from typing import Literal, Optional, Tuple
from . import *


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
        wash_scheme: Literal["D", 1, None] = 1,
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

        self.wash_scheme = wash_scheme

        self.liquid_class = liquid_class

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
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
