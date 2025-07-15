from robotools import *
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from . import *

import numpy as np


class TransferOperation:

    op_id_counter = 0

    def __init__(
        self,
        source: liquidhandling.Labware,
        source_wells: Union[str, Sequence[str], np.ndarray],
        destination: AdvancedLabware,
        destination_wells: Union[str, Sequence[str], np.ndarray],
        volumes: Union[float, Sequence[float], np.ndarray],
        *,
        label: Optional[str] = None,
        wash_scheme: Literal[1, 2, 3, 4, "flush", "reuse"] = 1,
        on_underflow: Literal["debug", "warn", "raise"] = "raise",
        **kwargs,
    ):
        self.id = TransferOperation.op_id_counter
        TransferOperation.op_id_counter += 1

        self.source = source
        self.source_wells = source_wells
        self.destination_wells = destination_wells

        self.volumes = volumes
        self.label = label
        self.wash_scheme = wash_scheme
        self.on_underflow = on_underflow

    def __str__(self):
        return f"{self.id}: {self.label}"

    def __repr__(self):
        return self.__str__()


class AutoWorklist(EvoWorklist):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.completed_ops = []
        self.pending_ops = []

    def auto_transfer(
        self,
        source: liquidhandling.Labware,
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

        # the label applies to the entire transfer operation and is not logged at individual aspirate/dispense steps

        op = TransferOperation(
            source,
            source_wells,
            destination,
            destination_wells,
            volumes,
            label=label,
            wash_scheme=wash_scheme,
            on_underflow=on_underflow,
        )

        self.comment(op.label)

        # Append this op to all of the destination wells we touch
        for well in destination_wells:
            destination.op_tracking[well].append(op)

        self.pending_ops.append(op)

    def __exit__(self, *args):
        for op in self.pending_ops:
            print(op)

        self.commit()

        super().__exit__(*args)
