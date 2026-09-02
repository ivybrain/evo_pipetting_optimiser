from robotools import EvoWorklist, Trough, Tip
from robotools.evotools.commands import evo_get_selection
from typing import Literal, Optional, Sequence, Tuple, Union
from . import *
import warnings
import numpy as np
import math
from .TransferOperation import TransferOperation
from ..utils.utils import *
from ..optimisation import make_plan, OptimisationError
from concurrent.futures import ThreadPoolExecutor
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)


class AutoWorklist(EvoWorklist):

    all_labware = set()

    def __init__(
        self,
        worklist_path: str,
        waste_location: Tuple[int, int] = None,
        cleaner_location: Tuple[int, int] = None,
        diti_mode: bool = False,
        default_tip: int = None,
        tip_discard_location: Tuple[int, int] = None,
        arm: int = 0,
        max_concurrent_ops: int = None,
        *args,
        **kwargs,
    ):
        """
        worklist_path
            Optional filename/filepath to write when the context is exited (must include a .gwl extension)
        waste_location:  (int, int)
            (grid, site) tuple for the waste station location on the evo. Used for automatic washes if diti_mode=False. If diti_mode=True,
            this is the location to drop ditis
        cleaner_location: (int, int)
            (grid, site) tuple for the cleaner station location on the evo. Used for automatic washes if diti_mode=False
        diti_mode: bool
            Set to true for an EVO with disposable tips instead of fixed
        default_tip: int
            Tip ID in evoware for the default disposable tip type to use when a specific tip type is not specified in a transfer command.
            Must be set if diti_mode = True
        tip_discard_location: Tuple(grid, site)
            In Diti mode, the location to drop used DiTis with the DropDiti command
        arm: int
            LiHa number. Default=0
        max_volume : int
            Maximum aspiration volume in µL
        max_concurrent_ops: int
            Maximum number of ops to optimise over at once. If there are more pipetting operations than this, they will be split into sections
            and optimised seperately, so that optimisation does not take a long time. Default None = no split.

        """
        super().__init__(worklist_path, *args, **kwargs)
        self.pending_ops = []

        self.tips_used = [False] * 8
        self.tip_contents = [None] * 8

        self.currently_optimising = False
        self.silence_append_warning = False

        self.processing = False
        self.max_concurrent_ops = max_concurrent_ops

        self.wash_params = {}
        self.set_wash_parameters(waste_location=waste_location, cleaner_location=cleaner_location, tip_discard_location=tip_discard_location)
        self.arm = arm

        self.diti_mode = diti_mode
        self.tips_mounted = None
        if diti_mode and (default_tip is None):
            raise ValueError("Default tip index must be specified in DiTi Mode")

        if diti_mode and (tip_discard_location is None):
            raise ValueError("Tip discard location must be set in Diti mode")

        if not diti_mode and not (cleaner_location and waste_location):
            raise ValueError(
                "Cleaner and Waste locations must be specified if diti_mode=False"
            )

        self.default_tip = default_tip

    def set_wash_parameters(
        self,
        cleaner_location=None,
        waste_location=None,
        decon_troughs: Union[Trough, Sequence[Trough]] = None,
        waste_vol: float = None,
        waste_delay: int = None,
        cleaner_vol: float = None,
        cleaner_delay: int = None,
        airgap: int = None,
        airgap_speed: int = None,
        retract_speed: int = None,
        fastwash: int = None,
        low_volume: int = None,
        decon_excess_volume: int = None,
        decon_liquid_class: str = None,
        decon_delay: int = None,
        max_tip_uses: int = None,
        tip_discard_location: Tuple[int, int] = None
    ):
        """
        Set the parameters that will be passed to evo_wash for washes triggered by auto_transfer().
        Normally, these are fine to leave as defaults
        Params not explicity passed to this method will stay as their previous value
        You can only set wash parameters once per optimisation session. If you need different wash parameters for different transfers,
        call commit() first to optimise transfers with the old wash parameters, before changing parameters for the next session
        Params starting with decon_ determine the behaviour of an aspiration from a decontamination trough
        Which occurs before the wash when auto_transfer() is called with wash_scheme="D"

        Parameters
        ----------
        waste_location : tuple
            Tuple with grid position (1-67) and site number (0-127) of waste as integers
        cleaner_location : tuple
            Tuple with grid position (1-67) and site number (0-127) of cleaner as integers
        decon_troughs : Trough or list of Troughs
            Troughs to aspirate for deontamination. If one trough, that trough will be used regardless. If a list of Troughs, the first Trough
            will be used until its min_volume is reached, then the next trough will be used. Troughs must have the .location parameter set
        arm : int
            number of the LiHa performing the action: 0 = LiHa 1, 1 = LiHa 2. Default 0
        waste_vol: float
            Volume in waste in mL (0-100). Default 3.0
        waste_delay : int
            Delay before closing valves in waste in ms (0-1000). Default 500
        cleaner_vol: float
            Volume in cleaner in mL (0-100). Default 4.0
        cleaner_delay : int
            Delay before closing valves in cleaner in ms (0-1000). Default 500
        airgap : int
            Volume of airgap in µL which is aspirated after washing the tips (system trailing airgap) (0-100). Default 10
        airgap_speed : int
            Speed of airgap aspiration in µL/s (1-1000). Default 70
        retract_speed : int
            Retract speed in mm/s (1-100). Default 30
        fastwash : int
            Use fast-wash module = 1, don't use it = 0. Default 1
        low_volume : int
            Use pinch valves = 1, don't use them = 0. Default 0
        decon_excess_volume : int
            Decon will aspirate however much liquid was present in a given tip, plus this excess value. Default 20ul
        decon_liquid_class : str
            Liquid class to use for decon aspiration
        decon_delay : int
            Delay to soak the tips after aspirating decon solution. Default 1000ms
        max_tip_uses:
            When reusing tips with wash_scheme = "smart" or None, this is the maximum number of uses of a tip before a wash is done anyway.
            This is useful to prevent a loss of accuracy and 'reset' the liquid class. Set to None for no limit. Default 5
        tip_discard_location: Tuple(grid, site)
            In Diti mode, the location to drop used DiTis with the DropDiti command
        """

      
        param_defaults = {
            "cleaner_location": None,
            "waste_location": None,
            "decon_troughs": None,
            "waste_vol": 3.0,
            "waste_delay": 500,
            "cleaner_vol": 4.0,
            "cleaner_delay": 500,
            "airgap": 10,
            "airgap_speed": 70,
            "retract_speed": 30,
            "fastwash": 1,
            "low_volume": 0,
            "decon_excess_volume": 20,
            "decon_liquid_class": None,
            "decon_delay": 1000,
            "max_tip_uses": 5,
            "tip_discard_location": None
        }

        for key, default in param_defaults.items():

            # If the passed parameter is none and the key is already set, don't overwrite it
            passed_value = locals()[key]
            if key not in self.wash_params:
                self.wash_params[key] = default

            if passed_value is None:
                continue

            self.wash_params[key] = passed_value

    def get_diti(self, tips, tip_index, retry=False):
        """
        Get Ditis when diti mode
        """

        if not self.diti_mode:
            raise Exception("Cannot get Ditis unless in DiTi mode")

        # Calculate the 8 bit mask for the selected tips
        # ie tips 1,3,5,7 == 10101010 == 170
        tip_selection = 0
        for tip in tips:
            if isinstance(tip, int):
                tip_selection += 2 ** (tip - 1)
            elif isinstance(tip, Tip):
                tip_selection += tip.value

        self.tips_mounted = tip_index
        self.append(
            f"B;GetDiti({tip_selection}, {tip_index}, {int(retry)}, {self.arm})"
        )

    def drop_diti(self, tips):

        if not self.diti_mode:
            raise Exception("Cannot drop Ditis unless in DiTi mode")

        tip_selection = 0
        for tip in tips:
            if isinstance(tip, int):
                tip_selection += 2 ** (tip - 1)
            elif isinstance(tip, Tip):
                tip_selection += tip.value

        self.tips_mounted = None

        grid = self.wash_params["tip_discard_location"][0]
        site = self.wash_params["tip_discard_location"][1]

        self.append(
            f'B;DropDiTi({tip_selection}, {grid}, {site-1}, {self.wash_params["airgap"]}, {self.wash_params["airgap_speed"]}, {self.arm})'
        )

    def liha_flush(self):
        '''
        Flushes the liha (dispenses any remaining liquid) at the waste location
        Useful if any liquid class leaves an excess
        Used in decontamination wash to remove any excess before aspirating decon
        '''

        if not self.wash_params["waste_location"]:
            raise ValueError("Waste location not specified")

        grid = self.wash_params["waste_location"][0]
        site = self.wash_params["waste_location"][1]

        # Well selection for all 8 wells of waste
        selection = evo_get_selection(8, 1, np.ones((8,1)))

        # Move the Liha to the waste position, z dispense
        self.append(f'B;MoveLiha(255,{grid},{site - 1},1,"{selection}",{0},{1},0,10,0,0);')


        # Firmware commands taken from logs when executing the 'Liha Safe Flush' direct command

        # Position valve logical. PVL 0 sets plunger valve to outlet port
        self.append('B;Command("C5PVL0,0,0,0,0,0,0,0",1,1,,,2,2,0);')

        # SEP is set end speed for plunger movement, in half-steps per second
        self.append('B;Command("C5SEP5400,5400,5400,5400,5400,5400,5400,5400",1,1,,,2,2,0);')

        # SPP is plunger stop speed in half-steps per second
        self.append('B;Command("C5SPP2700,2700,2700,2700,2700,2700,2700,2700",1,1,,,2,2,0);')

        # PPA = move plunger to absolute positoin
        # Position 0 (fully closed) for all tips
        self.append('B;Command("C5PPA0,0,0,0,0,0,0,0",1,1,,,2,2,0);')

        # Move the Liha back to global z travel above waste position
        self.append(f'B;MoveLiha(255,{grid},{site - 1},1,"{selection}",{0},{4},0,10,0,0);')


    def auto_transfer(
        self,
        source: Union[AdvancedLabware, Trough],
        source_wells: Union[str, Sequence[str], np.ndarray],
        destination: AdvancedLabware,
        destination_wells: Union[str, Sequence[str], np.ndarray],
        volumes: Union[float, Sequence[float], np.ndarray],
        *,
        label: Optional[str] = "",
        on_underflow: Literal["debug", "warn", "raise"] = "raise",
        liquid_class: str = None,
        wash_scheme: Literal["D", 1, None, "smart"] = 1,
        tips: Union[int, Sequence[int], Sequence[Tip]] = None,
        tip_index=None,
    ) -> None:
        """Transfer operation between two labwares with automatic pipetting order optimisation
        Ensures correct order of transfers to each particular well,
        While grouping transfers together for pipetting efficiency
        use .commit() to calculate the pipetting strategy and populate the worklist with the corresponding commands

        Parameters
        ----------
        source : AdvancedLabware or Trough (with location property set to (grid,site) tuple)
        source_wells : str or iterable
            List of source well ids
        destination : AdvancedLabware
            Destination labware
        destination_wells : str or iterable
            List of destination well ids
        volumes : float or iterable
            Volume(s) to transfer
        label : str
            Label of the operation to log into labware history
        liquid_class : str
            Liquid class to use for pipetting
        wash_scheme : "D", 1, "smart" or None
            Desired wash behaviour. None will not insert any washes. 1 (default) will conduct a wash for each tip that is used,
            before it is used again. It does this by calling evo_wash. Parameters passed to evo_wash can be set by calling set_wash_parameters
            "D" will conduct a decontamination wash.
            "smart" will wash a tip if it has touched any liquid with a composition other than the one it is going to handle
            i.e. will skip washes if it is only touching a single liquid type
        tip: int or iterable of (ints or Tips), or None
            Restrict the pipetting operations to specific tips. If None, any tip can be used. If a single number, will use that tip for all operations.
            If a list, will use the tips in the list for the corresponding operations
        """

        # add the labware to the set of all labware we've used
        self.all_labware.add(source)
        self.all_labware.add(destination)

        # reformat the convenience parameters
        source_wells = np.array(source_wells).flatten("F")
        destination_wells = np.array(destination_wells).flatten("F")
        volumes = np.array(volumes).flatten("F")
        nmax = max((len(source_wells), len(destination_wells), len(volumes)))

        # If in diti mode and tip not specified, assign the default tip
        if self.diti_mode and (tip_index is None):
            tip_index = self.default_tip

        # If diti mode, append the tip index to the liquid class so that ops are grouped by tip type as well
        if self.diti_mode:
            liquid_class += f";{tip_index}"

        # Reformat parameters where a single value has been entered into arrays
        if len(source_wells) == 1:
            source_wells = np.repeat(source_wells, nmax)
        if len(destination_wells) == 1:
            destination_wells = np.repeat(destination_wells, nmax)
        if len(volumes) == 1:
            volumes = np.repeat(volumes, nmax)
        if tips is None or isinstance(tips, int) or len(tips) == 1:
            tips = np.repeat(tips, nmax)
        lengths = (len(source_wells), len(destination_wells), len(volumes), len(tips))
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Number of source/destination/volumes/tips must be equal (or 1). They were {lengths}"
            )

        if not (isinstance(source, AdvancedLabware) or isinstance(source, Trough)):
            raise TypeError("Source must be AdvancedLabware or Trough for auto_transfer")

        if not isinstance(destination, AdvancedLabware):
            raise TypeError("Destination must be AdvancedLabware for auto_transfer")

        if liquid_class is None:
            raise ValueError("Liquid class must be speicified for auto_transfer")

        if wash_scheme not in ["D", "smart", 1, None]:
            raise ValueError('Wash schemes supported for auto_transfer are "D", 1, None, or "smart"')

        if wash_scheme == "D" and self.wash_params["decon_troughs"] is None:
            raise ValueError(
                "Using decon wash without decon_troughs specified. Call set_wash_parameters() to set it."
            )
        if wash_scheme == "D" and self.wash_params["decon_liquid_class"] is None:
            raise ValueError(
                "Using decon wash without decon_liquid_class specified. Call set_wash_parameters() to set it."
            )

        # Track if we currently have operations waiting to be optimised
        # That haven't been committed to the worklist. If so, we want to warn if the user
        # Tries to add anything else to the worklist.
        self.currently_optimising = True

        # source wells don't matter for a trough, set them all 0
        # then optimiser can arrange as needed
        if isinstance(source, Trough):
            source_wells = ["A01"] * len(source_wells)

        # Create a TransferOperation object with all the details of the transfer
        # For every pair of source,destination wells
        for i in range(len(source_wells)):

            # Check for large volume handling
            repeats = math.ceil(volumes[i] / self.max_volume)
            volume = volumes[i] / repeats

            # Rationalise to A01 format
            source_well = source_wells[i]
            if len(source_well) == 2:
                source_well = source_well[0] + "0" + source_well[1]
            dest_well = destination_wells[i]
            if len(dest_well) == 2:
                dest_well = dest_well[0] + "0" + dest_well[1]

            for j in range(repeats):

                op = TransferOperation(
                    source,
                    source.indices[source_well],
                    destination,
                    destination.indices[dest_well],
                    volume,
                    label=label,
                    wash_scheme=wash_scheme,
                    on_underflow=on_underflow,
                    liquid_class=liquid_class,
                    tip=tips[i],
                    tip_index=tip_index,
                    worklist=self,
                )

                # Append this op to the labware we're aspirating to
                # So that future transfers to the well know that they need to wait for this transfer first
                source.source_op_tracking[source_well].append(op)
                destination.op_tracking[dest_well].append(op)

                # Add this op to the pending operations
                self.pending_ops.append(op)

    def append(self, *args, **kwargs):
        # If we have un-commited auto transfers and the user tries to append something else to the worklist
        # We have to optimise and commit these auto transfers first, or they'll appear after whatever the
        # user appends.
        # Warn the user, then commit the auto transfers automatically
        if self.currently_optimising and not self.silence_append_warning:

            warnings.warn(
                "Modifying worklist after auto_transfer without commit. Auto transfers will be committed now, before your modification"
            )
            self.commit()
        super().append(*args, **kwargs)

    def wash(self, scheme=1):
        '''
        Replace the default evo wash with one that accepts the decontamintation 'D' scheme
        '''
        if scheme not in [1, 2, 3, 4, 'D']:
            raise ValueError("Invalid Wash Scheme")
        
        self.append(f"W{scheme};")

    # evo_aspirate, dispense, and wash overrides that we will use interally when committing optimisations.
    # Avoid warning the user in this case
    def _evo_aspirate(self, *args, silence_append_warning=True, **kwargs):
        self.silence_append_warning = silence_append_warning
        super().evo_aspirate(*args, arm=self.arm, **kwargs)
        self.silence_append_warning = False

    def _evo_dispense(self, *args, silence_append_warning=True, **kwargs):
        self.silence_append_warning = silence_append_warning
        super().evo_dispense(*args, arm=self.arm, **kwargs)
        self.silence_append_warning = False

    

    def _auto_wash(
        self,
        *args,
        silence_append_warning=True,
        tips=[],
        wash_schemes=[],
        tip_volumes=[],
        **kwargs,
    ):
        """
        Internal method called by auto_transfer to conduct a wash, with or without decon

        """

        self.silence_append_warning = silence_append_warning

        if not (len(tips) == len(wash_schemes) == len(tip_volumes)):
            raise OptimisationError("Internal wash assignment error")

        if self.diti_mode:
            self.drop_diti(tips)
            self.silence_append_warning = False
            return

        # Specify decon tips and volumes
        decon_mask = [scheme == "D" for scheme in wash_schemes]
        decon_tips = [tips[i] for i in range(len(tips)) if decon_mask[i]]
        decon_wells = [tip - 1 for tip in decon_tips]
        decon_volumes = [
            min(
                self.max_volume,
                tip_volumes[i] + self.wash_params["decon_excess_volume"],
            )
            for i in range(len(tips))
            if decon_mask[i]
        ]

        decon_sorted = sorted(list(zip(decon_tips, decon_volumes)))
        if decon_sorted:
            decon_tips, decon_volumes = zip(*decon_sorted)
        if len(decon_tips) > 0:
            self.comment("Decontaminating")

            # Flush any existing liquid out of the tips before aspirating decontaminant
            self.liha_flush()

            needed_volume = sum(decon_volumes)
            # If we have a single trough, just use that
            if isinstance(self.wash_params["decon_troughs"], Trough):
                decon_trough = self.wash_params["decon_troughs"]
            else:
                # If we have a list of decon troughs, find the first with sufficient volume
                decon_trough_index = 0
                decon_trough = self.wash_params["decon_troughs"][decon_trough_index]
                while (
                    decon_trough.volumes[0, 0] - needed_volume
                    <= decon_trough.min_volume
                    and decon_trough_index < len(self.wash_params["decon_troughs"]) - 1
                ):
                    decon_trough_index += 1
                    decon_trough = self.wash_params["decon_troughs"][decon_trough_index]


            # Aspirate decon liquid
            self._evo_aspirate(
                decon_trough,
                decon_trough.wells[decon_wells, 0],
                decon_trough.location,
                tips=list(decon_tips),
                volumes=list(decon_volumes),
                liquid_class=self.wash_params["decon_liquid_class"],
            )

            self.silence_append_warning = silence_append_warning
            # Run timer to soak the tips for the specified delay
            self.append('B;StartTimer("1");')
            duration_str = "{0:.2f}".format(self.wash_params["decon_delay"] / 1000)
            self.append(f'B;WaitTimer("1","{duration_str}");')

        # Conduct standard wash sequence
        wash_mask = [scheme is not None for scheme in wash_schemes]
        wash_tips = [tips[i] for i in range(len(tips)) if wash_mask[i]]

        # Collect the generic wash parameters to pass to the evo_wash function
        wash_params = {
            key: value
            for key, value in self.wash_params.items()
            if (not key.startswith("decon_")) and key not in ["max_tip_uses", "tip_discard_location"]
        }

        if len(wash_tips) > 0:
            super().evo_wash(tips=wash_tips, arm=self.arm, **wash_params)
        self.silence_append_warning = False

    def execute_plan(self, plan, counts):

        decon_at_end = False
        
        for i, (action, ops, tips) in enumerate(plan):

            # If the plan ends at this step, track if we need to decontaminate for the final wash
            decon_at_end = False

            if action == "Aspirate":

                # Get the first op in the group
                source_op = ops[0]
                # Get the col of this source group
                source_col = source_op.source_pos[1]
                # Get the rows we need to aspirate from the source_pos attribute of each op
                source_rows = [op.source_pos[0] for op in ops]

                # Get the volumes stored in the ops
                volumes = [op.volume for op in ops]

                # Mount tips if we don't have any
                if self.diti_mode and not self.tips_mounted:

                    # Find the next 'wash' action and include all tips used up until that action
                    search_index = i
                    get_tips = []
                    while search_index < len(plan):
                        a, o, t = plan[search_index]
                        if a == "Aspirate":
                            get_tips += t
                        if a == "Wash":
                            break
                        search_index += 1

                    self.silence_append_warning = True
                    self.get_diti(get_tips, ops[0].tip_index, retry=True)
                    self.silence_append_warning = False

                # Check that tip type index matches for all ops
                if self.diti_mode:
                    for op in ops:
                        if self.tips_mounted != op.tip_index:
                            raise OptimisationError(
                                "Error, attempting to aspirate with wrong tip type mounted"
                            )

                # Check from earlier troubleshooting
                if len(tips) != len(set(tips)):
                    raise OptimisationError("Error in tip logic")

                # If we have a trough, just set source rows to whichever tips we're using
                if (getattr(source_op.source, "virtual_rows", 0) or 0) > max(tips) - 1:
                    source_rows = [tip - 1 for tip in tips]

                # Check that the tip-row offset is consistent - i.e. that Evoware will actually do this in one move
                offset = source_rows[0] - (tips[0] - 1)
                for i in range(len(source_rows)):
                    if source_rows[i] - (tips[i] - 1) != offset:
                        raise OptimisationError("Tip assignment offset inconsistency")

                if not (
                    offset <= (getattr(source_op.source, "offset_limit_down", None) or 10000)
                    and offset >= -(getattr(source_op.source, "offset_limit_up", None) or 10000)
                ):
                    raise OptimisationError("Offset limit violated")

                # If in diti mode, extract the liquid class only from the liquid class + tip string
                liquid_class = source_op.liquid_class
                if self.diti_mode:
                    liquid_class = liquid_class.split(";")[0]

                # Perform the aspiration
                self._evo_aspirate(
                    source_op.source,
                    source_op.source.wells[source_rows, source_col],
                    source_op.source.location,
                    list(tips),
                    list(volumes),
                    liquid_class=liquid_class,
                    label=" + ".join(set([op.label for op in ops]))
                    + ", ops: "
                    + ",".join([str(op.id) for op in ops]),
                    on_underflow=source_op.on_underflow,
                )

            if action == "Dispense":

                # Get the first op in the group
                dest_op = ops[0]
                # Get the col of this source group
                dest_col = dest_op.dest_pos[1]

                # Get the rows we need to dispense to for the group
                dest_rows = [op.dest_pos[0] for op in ops]

                # If we have a trough, just set source rows to whichever tips we're using
                if (
                    getattr(source_op.destination, "virtual_rows", 0)
                    or 0 > max(tips) - 1
                ):
                    dest_rows = [tip - 1 for tip in tips]

                # Get the volume stored in each op
                volumes = [op.volume for op in ops]

                # Get the composition from the source labware for each op
                compositions = [
                    op.source.get_well_composition(op.source.wells[op.source_pos])
                    for op in ops
                ]

                # If we have a trough, just set rows to whichever tips we're using
                if getattr(dest_op.destination, "virtual_rows", 0) or 0 > max(tips) - 1:
                    dest_rows = [tip - 1 for tip in tips]

                # Check that the tip-row offset is consistent for all tips-row combinations. If not, Evoware can't do this in one move.
                offset = dest_rows[0] - (tips[0] - 1)
                for i in range(len(dest_rows)):
                    if dest_rows[i] - (tips[i] - 1) != offset:
                        raise OptimisationError("Tip offest issue")

                if not (
                    offset <= (dest_op.destination.offset_limit_down or 10000)
                    and offset >= -(dest_op.destination.offset_limit_up or 10000)
                ):
                    raise OptimisationError("Offset limit exceeded")

                # If in diti mode, extract the liquid class only from the liquid class + tip string
                liquid_class = dest_op.liquid_class
                if self.diti_mode:
                    liquid_class = liquid_class.split(";")[0]

                # Perform the dispense op
                self._evo_dispense(
                    dest_op.destination,
                    dest_op.destination.wells[dest_rows, dest_col],
                    dest_op.destination.location,
                    list(tips),
                    list(volumes),
                    liquid_class=liquid_class,
                    label=" + ".join(set([op.label for op in ops]))
                    + ", ops: "
                    + ",".join([str(op.id) for op in ops]),
                    compositions=compositions,
                )

                decon_at_end = any(op.wash_scheme == "D" for op in ops)
                end_wash_volumes = list(volumes)

            if action == "Wash":

                # Wash after this group of ops
                self._auto_wash(
                    tips=tips,
                    tip_volumes=[op.volume for op in ops],
                    wash_schemes=[op.wash_scheme for op in ops],
                )
                # Add a break after each wash to improve readability
                super().append("B;")

        # Wash at the end for safety
        # Wash all 8 tips. If we have less than 8 volumes recorded, wash the max volume for all 8
        if len(end_wash_volumes) < 8:
            end_wash_volumes = [max(end_wash_volumes)] * 8
        self._auto_wash(
            tips=list(range(1, 9)), tip_volumes=end_wash_volumes, wash_schemes=["D" if decon_at_end else 1] * 8
        )
        super().append(
            f"C;Optimisation session complete. aspirates: {counts[0]}, dispenses: {counts[1]}, washes: {counts[2]}"
        )
        self.asp_count = counts[0]
        self.disp_count = counts[1]
        self.wash_count = counts[2] + 1
        super().append("B;")

        return

    def report_ops(self):
        """
        Prints the report directly to stdout
        """
        print(self.report)

    @property
    def report(self):
        """
        String report of all operations queued for the current optimisation session
        """
        return "\n".join(
            [str(op) for op in sorted(self.pending_ops, key=lambda x: x.id)]
        )

    def commit(self, report=True):

        # If we have ops pending, optimise and apply them
        if len(self.pending_ops) > 0 and not self.processing:
            self.processing = True

            # Split the ops into sections to optimise for performance
            if self.max_concurrent_ops:
                segments = range((len(self.pending_ops) // self.max_concurrent_ops) + 1)
            else:
                segments = [0]


            def create_plans(progress=None):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    segment_ops = []
                    tasks = []
                    for s in segments:

                        if self.max_concurrent_ops:
                            ops = self.pending_ops[
                                s
                                * self.max_concurrent_ops : (s + 1)
                                * self.max_concurrent_ops
                            ]
                        else:
                            ops = self.pending_ops

                        segment_ops.append(ops)

                        if progress:
                            tasks.append(progress.add_task("Optimising...", total=len(ops)))
                        else:
                            tasks.append(None)

                    # Execute optimisation to create plan
                    results = executor.map(
                        make_plan, segment_ops, [progress] * len(segments), tasks
                    )

                    return results


            if report:
                with Progress(
                    TextColumn("[bright_magenta]{task.description}"),
                    BarColumn(finished_style="bar.complete"),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                ) as progress:
                    results = create_plans(progress)
            else:
                results = create_plans()
                    

                
            # Execute each plan in order (multiple plans occur if optimisation was split into batches)
            for plan, cost, (asp_count, disp_count, wash_count) in results:

                self.execute_plan(plan, (asp_count, disp_count, wash_count))

            # Reset the pending ops
            self.pending_ops = []

            # Reset the op tracking on all labware
            for labware in self.all_labware:
                if isinstance(labware, AdvancedLabware):
                    labware.reset_op_tracking()

            self.all_labware = set()
            self.processing = False
            if report:
                print(
                    f"Optimisation complete. aspirates: {self.asp_count}, dispenses: {self.disp_count}, washes: {self.wash_count}"
                )
        self.currently_optimising = False
        super().append("B;")

    def __enter__(self) -> "AutoWorklist":
        # Redefine to give correct type hint
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        # Commit to optimise and apply any pending ops before exiting, but only if we haven't hit an exception
        if exc_type is None:
            self.commit()

            # Reset transfer operation counter
            TransferOperation._op_id_counter = 0

        super().__exit__(exc_type, exc_value, traceback)
