from robotools import *
from typing import Literal, Optional, Tuple
from . import *


class TransferOperation:
    """
    Internal class that stores all the information needed
    For a transfer from one well to another
    """

    # Track ids of each op
    _op_id_counter = 0

    _loc_id = {}
    _id_loc = []
    _loc_counter = 0

    _lc_counter = 0
    _lc_id = {}
    _id_lc = []

    @property
    def loc_counter(self):
        return TransferOperation._loc_counter

    @property
    def lc_counter(self):
        return TransferOperation._lc_counter

    def get_loc_code(labware, column):
        if (labware.name, column) in TransferOperation._loc_id:
            return TransferOperation._loc_id[(labware.name, column)]

        TransferOperation._id_loc.append((labware, column))
        TransferOperation._loc_id[(labware.name, column)] = (
            TransferOperation._loc_counter
        )
        TransferOperation._loc_counter += 1

        return TransferOperation._loc_counter - 1

    def get_lc_code(liquid_class):
        if liquid_class in TransferOperation._lc_id:
            return TransferOperation._lc_id[liquid_class]

        TransferOperation._id_lc.append(liquid_class)
        TransferOperation._lc_id[liquid_class] = TransferOperation._lc_counter
        TransferOperation._lc_counter += 1

        return TransferOperation._lc_counter - 1

    def reset_counters():
        TransferOperation._op_id_counter = 0
        TransferOperation._loc_counter = 0
        TransferOperation._loc_id = {}
        TransferOperation._id_loc = []
        TransferOperation._lc_counter = 0
        TransferOperation._lc_id = {}
        TransferOperation._id_lc = []

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
        wash_scheme: Literal["D", 1, None, "smart"] = 1,
        **kwargs,
    ):
        self.id = TransferOperation._op_id_counter
        TransferOperation._op_id_counter += 1

        self.source = source
        self.source_pos = source_pos
        self.destination = destination
        self.dest_pos = dest_pos

        self.volume = volume
        self.label = label
        self.on_underflow = on_underflow

        self.source_dep = source_dep
        self.dest_dep = dest_dep

        self.wash_scheme = wash_scheme

        self.liquid_class = liquid_class

        self.source_loc = TransferOperation.get_loc_code(source, source_pos[1])
        self.dest_loc = TransferOperation.get_loc_code(destination, dest_pos[1])

        self.lc_id = TransferOperation.get_lc_code(liquid_class)

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
