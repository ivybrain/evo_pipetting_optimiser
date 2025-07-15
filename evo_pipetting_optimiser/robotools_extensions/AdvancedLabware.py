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

    def __init__(self, *args, grid=None, site=None, **kwargs):

        super().__init__(*args, **kwargs)

        if not grid or not site:
            raise ValueError("Grid and Site must be specified")
        self.grid = grid
        self.site = site

        self.op_tracking = {well: [] for well in self.wells.flatten("F")}
