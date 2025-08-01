import robotools


class AdvancedLabware(robotools.Labware):
    """
    Labware object with added capability to track AutoWorklist operations for pipetting optimisation
    and offset limit support
    """

    @property
    def last_op(self):
        """
        A dictionary of {well: TransferOperation} which tracks the last TransferOperation which touched each well
        Used to check if a pipetting operation has any dependencies
        """
        return {
            well: (
                self.op_tracking[well][-1] if len(self.op_tracking[well]) > 0 else None
            )
            for well in self.wells.flatten("F")
        }

    def __init__(self, *args, location=None, offset_limit=None, **kwargs):
        """Creates an `AdvancedLabware` object for AutoWorklist pipetting optimisation

        Parameters
        ----------
        name : str
            Label that the labware is identified by.
        rows : int
            Number of rows in the labware
        columns : int
            Number of columns in the labware
        min_volume : float
            Filling volume that must remain after an aspirate operation.
        max_volume : float
            Maximum volume that must not be exceeded after a dispense.
        initial_volumes : float, array-like, optional
            Initial filling volume of the wells (default: 0)
        offset_limit : int, [-7, 7], optional, default=None
            Limits the vertical translation of tips accessing this Labware (when used with AutoWorklist.auto_transfer)
            Useful for labware near the top or bottom of the evo deck, where not every row can be accessed by every tip
            If negative, limits the upwards translation of the tips: -4 means the tips can shift no more than 4 rows above
            their default position. I.e. tip 5 can access row A, but tip 6 cannot
            If positive, limits the downward translation of the tips.
            If 0, no offsets are allowed
            If none, no offset limit and any offsets are permitted
            Setting a limit will reduce optimisation effectiveness and increase pipetting time
        virtual_rows : int, optional
            When specified to a positive number, the `Labware` is treated as a trough.
            Must be used in combination with `rows=1`.
            For example: A `Labware` with virtual rows can be accessed with 6 Tips,
            but has just one row in the `volumes` array.
        component_names : dict, optional
            A dictionary that names the content of non-empty real wells for composition tracking.
        """

        super().__init__(*args, **kwargs)

        if not location:
            raise ValueError("Grid and Site must be specified")
        self.location = location

        self.offset_limit = offset_limit

        self.op_tracking = {well: [] for well in self.wells.flatten("F")}
