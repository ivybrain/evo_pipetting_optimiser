import robotools
from robotools.liquidhandling.composition import get_trough_component_names
from . import AdvancedLabware


class AdvancedTrough(AdvancedLabware, robotools.Trough):
    """
    Special type of labware that can be accessed by many pipette tips in parallel.
    With with added capability to track AutoWorklist operations for pipetting optimisation
    and offset limit support
    """

    def __init__(
        self,
        *args,
        location=None,
        offset_limit_up=None,
        offset_limit_down=None,
        **kwargs
    ):
        """Creates an `AdvancedTrough` object for AutoWorklist pipetting optimisation

        Parameters
        ----------
        name : str
            Label that the labware is identified by.
        virtual_rows : int, optional
            Number of tips that may access the trough in parallel.
            For example: A `Labware` with virtual rows can be accessed with 6 Tips,
            but has just one row in the `volumes` array.
        columns : int
            Number of columns in the labware
        min_volume : float
            Filling volume that must remain after an aspirate operation.
        max_volume : float
            Maximum volume that must not be exceeded after a dispense.
        initial_volumes : float, array-like, optional
            Initial filling volume of the wells (default: 0)
        column_names : array-like, optional
            A list/tuple of names for the column-wise contents of the troughs.
            If provided, these names are used for composition tracking.
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
        """

        robotools.Trough.__init__(self, *args, **kwargs)

        if not location:
            raise ValueError("Grid and Site must be specified")
        self.location = location

        self.offset_limit_up = offset_limit_up
        self.offset_limit_down = offset_limit_down

        self.op_tracking = {well: [] for well in self.wells.flatten("F")}
