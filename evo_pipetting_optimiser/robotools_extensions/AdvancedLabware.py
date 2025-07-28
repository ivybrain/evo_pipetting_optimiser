import robotools
import numpy as np
from . import LiquidNode


class AdvancedLabware(robotools.Labware):
    """
    Extended labware class to allow pipetting order optimisation
    """

    @property
    def last_op(self):
        return {
            well: (
                self.op_tracking[well][-1] if len(self.op_tracking[well]) > 0 else None
            )
            for well in self.wells.flatten("F")
        }

    def __init__(self, *args, location=None, offset_limit=None, **kwargs):

        super().__init__(*args, **kwargs)

        if not location:
            raise ValueError("Grid and Site must be specified")
        self.location = location

        self.offset_limit = offset_limit

        self.op_tracking = {well: [] for well in self.wells.flatten("F")}
