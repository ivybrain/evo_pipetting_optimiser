from . import AdvancedLabware

id_counter = 0


class LiquidNode:
    def __init__(self, labware: AdvancedLabware = None, well=None, complete=False):
        """
        A node representing a liquid
        """
        self.id = id_counter
        id_counter += 1

        self.complete = complete
        self.Labware = labware
        self.well = well

        # List of other LiquidNodes that must be completed before this one is completed
        self._dependencies = []
