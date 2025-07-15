import robotools


class AdvancedLabware(robotools.Labware):
    """
    Extended labware class to allow pipetting order optimisation
    """

    def __init__(self, *args, grid=None, site=None, **kwargs):
        if not grid or not site:
            raise ValueError("Grid and Site must be specified")
        self.grid = grid
        self.site = site

        super().__init__(*args, **kwargs)
