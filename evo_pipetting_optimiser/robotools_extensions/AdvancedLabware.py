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

    def latest_volume(self, well):
        # Default volume if we have no ops
        vol = self.volumes[self.indices[well]]
        dest_op = None

        # If we have an op that uses this labware as a destination, get volume from that
        if self.op_tracking[well]:
            dest_op = self.op_tracking[well][-1]
            vol = dest_op.dest_volume

        # If we have an op that uses this labware as a source, and it's more recent than the dest op, use that volume
        if self.source_op_tracking[well] and (
            (not dest_op) or self.source_op_tracking[well][-1] > dest_op
        ):
            vol = self.source_op_tracking[well][-1].source_volume

        return vol

    def latest_composition(self, well):
        if self.op_tracking[well]:
            op = self.op_tracking[well][-1]
            return op.dest_composition

        return self.get_well_composition(well)

    def reset_op_tracking(self):
        self.op_tracking = {well: [] for well in self.wells.flatten("F")}
        self.source_op_tracking = {well: [] for well in self.wells.flatten("F")}

    def __init__(
        self,
        *args,
        location=None,
        offset_limit_up=None,
        offset_limit_down=None,
        **kwargs,
    ):
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
        offset_limit_up : int, optional, default=None
            Limits the vertical translation upwards (towards the back of the evo) of tips accessing this Labware
            (when used with AutoWorklist.auto_transfer)
            Useful for labware near the top of the evo deck, where not every row can be accessed by every tip
            If set to an int, limits the upwards translation of the tips: 4 means the tips can shift no more than 4 rows above
            their default position. I.e. tip 5 can access row A, but tip 6 cannot
            If 0, no upwards shift is allowed
            If none, no offset limit and any offsets are permitted
            Setting a limit will reduce optimisation effectiveness and increase pipetting time
        offset_limit_down : int, optional, default=None
            Same as offset_limit_up, but limits tip translation downwards
            Note: a 12-row labware needs a least a down offset of 4 for tip 8 to access row 12
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
        self.offset_limit_up = offset_limit_up if offset_limit_up is not None else 7
        self.offset_limit_down = (
            offset_limit_down if offset_limit_down is not None else self.n_rows - 1
        )

        self.reset_op_tracking()
