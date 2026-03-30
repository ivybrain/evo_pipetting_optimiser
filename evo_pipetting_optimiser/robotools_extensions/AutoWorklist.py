from robotools import *
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from . import *
import warnings
import numpy as np
import math
from .TransferOperation import TransferOperation
from ..utils.utils import *
from ..optimisation import group_ops, make_plan
import time


class OffsetLimitException(Exception):
    pass


class AutoWorklist(EvoWorklist):

    all_labware = set()

    def __init__(
        self,
        worklist_path,
        waste_location: Tuple[int, int],
        cleaner_location: Tuple[int, int],
        *args,
        **kwargs,
    ):
        """
        worklist_path
            Optional filename/filepath to write when the context is exited (must include a .gwl extension)
        waste_location:  (int, int)
            (grid, site) tuple for the waste station location on the evo. Used for automatic washes
        cleaner_location: (int, int)
            (grid, site) tuple for the cleaner station location on the evo. Used for automatic washes
        max_volume : int
            Maximum aspiration volume in µL

        """
        super().__init__(worklist_path, *args, **kwargs)
        self.pending_ops = set()

        self.tips_used = [False] * 8
        self.tip_contents = [None] * 8

        self.currently_optimising = False
        self.silence_append_warning = False

        self.processing = False

        self.wash_params = {
            "waste_location": waste_location,
            "cleaner_location": cleaner_location,
        }
        self.set_wash_parameters()

    def set_wash_parameters(
        self,
        cleaner_location=None,
        waste_location=None,
        decon_troughs: Union[Trough, Sequence[Trough]] = None,
        waste_vol: float = 3.0,
        waste_delay: int = 500,
        cleaner_vol: float = 4.0,
        cleaner_delay: int = 500,
        airgap: int = 10,
        airgap_speed: int = 70,
        retract_speed: int = 30,
        fastwash: int = 1,
        low_volume: int = 0,
        decon_excess_volume: int = 20,
        decon_liquid_class: str = None,
        decon_delay: int = 1000,
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
            number of the LiHa performing the action: 0 = LiHa 1, 1 = LiHa 2
        waste_vol: float
            Volume in waste in mL (0-100)
        waste_delay : int
            Delay before closing valves in waste in ms (0-1000)
        cleaner_vol: float
            Volume in cleaner in mL (0-100)
        cleaner_delay : int
            Delay before closing valves in cleaner in ms (0-1000)
        airgap : int
            Volume of airgap in µL which is aspirated after washing the tips (system trailing airgap) (0-100)
        airgap_speed : int
            Speed of airgap aspiration in µL/s (1-1000)
        retract_speed : int
            Retract speed in mm/s (1-100)
        fastwash : int
            Use fast-wash module = 1, don't use it = 0
        low_volume : int
            Use pinch valves = 1, don't use them = 0
        decon_excess_volume : int
            Decon will aspirate however much liquid was present in a given tip, plus this excess value. Default 20ul
        decon_liquid_class : str
            Liquid class to use for decon aspiration
        decon_delay : int
            Delay to soak the tips after aspirating decon solution. Default 1000ms
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
        }

        for key, default in param_defaults.items():

            # If the passed parameter is the default value, and the property is already set, don't overwrite it
            passed_value = locals()[key]
            if passed_value == default and key in self.wash_params:
                continue

            self.wash_params[key] = passed_value

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
        wash_scheme: Literal["D", 1, None] = 1,
        **kwargs,
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


        on_underflow
            What to do about volume underflows (going below ``vmin``) in non-empty wells.

            Options:

            - ``"debug"`` mentions the underflowing wells in a log message at DEBUG level.
            - ``"warn"`` emits an :class:`~robotools.liquidhandling.exceptions.VolumeUnderflowWarning`. This `can be captured in unit tests <https://docs.pytest.org/en/stable/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests>`_.
            - ``"raise"`` raises a :class:`~robotools.liquidhandling.exceptions.VolumeUnderflowError` about underflowing wells.
        """

        # add the labware to the set of all labware we've used
        self.all_labware.add(source)
        self.all_labware.add(destination)

        # reformat the convenience parameters
        source_wells = np.array(source_wells).flatten("F")
        destination_wells = np.array(destination_wells).flatten("F")
        volumes = np.array(volumes).flatten("F")
        nmax = max((len(source_wells), len(destination_wells), len(volumes)))

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

        assert isinstance(source, AdvancedLabware) or isinstance(
            source, Trough
        ), "Source must be AdvancedLabware or Trough for auto_transfer"

        assert isinstance(
            destination, AdvancedLabware
        ), "Destination must be AdvancedLabware for auto_transfer"

        assert (
            liquid_class is not None
        ), "Liquid class must be speicified for auto_transfer"

        assert wash_scheme in [
            "D",
            "smart",
            1,
            None,
        ], "Wash schemes supported for auto_transfer are D, 1, or None"

        assert (
            wash_scheme != "D" or self.wash_params["decon_troughs"] is not None
        ), "Using decon wash without decon_troughs specified. Call set_wash_parameters() to set it."
        assert (
            wash_scheme != "D" or self.wash_params["decon_liquid_class"] is not None
        ), "Using decon wash without decon_liquid_class specified. Call set_wash_parameters() to set it."

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
                )

                # Append this op to the labware we're aspirating to
                # So that future transfers to the well know that they need to wait for this transfer first
                destination.op_tracking[destination_wells[i]].append(op)

                # Add this op to the pending operations set
                self.pending_ops.update([op])

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

    # evo_aspirate, dispense, and wash overrides that we will use interally when committing optimisations.
    # Avoid warning the user in this case
    def _evo_aspirate(self, *args, silence_append_warning=True, **kwargs):
        self.silence_append_warning = silence_append_warning
        super().evo_aspirate(*args, **kwargs)
        self.silence_append_warning = False

    def _evo_dispense(self, *args, silence_append_warning=True, **kwargs):
        self.silence_append_warning = silence_append_warning
        super().evo_dispense(*args, **kwargs)
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

        assert (
            len(tips) == len(wash_schemes) == len(tip_volumes)
        ), "Internal wash assignment error"
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

        wash_mask = [scheme is not None for scheme in wash_schemes]
        wash_tips = [tips[i] for i in range(len(tips)) if wash_mask[i]]

        wash_params = {
            key: value
            for key, value in self.wash_params.items()
            if not key.startswith("decon_")
        }

        if len(wash_tips) > 0:
            super().evo_wash(tips=wash_tips, **wash_params)
        self.silence_append_warning = False

    def execute_plan(self, plan, counts):

        for action, ops, tips in plan:

            if action == "Aspirate":

                # Get the first op in the group
                source_op = ops[0]
                # Get the col of this source group
                source_col = source_op.source_pos[1]
                # Get the rows we need to aspirate from the source_pos attribute of each op
                source_rows = [op.source_pos[0] for op in ops]

                # Get the volumes stored in the ops
                volumes = [op.volume for op in ops]

                # Check from earlier troubleshooting
                if len(tips) != len(set(tips)):
                    raise Exception("Error in tip logic")

                # If we have a trough, just set source rows to whichever tips we're using
                if getattr(source_op.source, "virtual_rows", 0) or 0 > max(tips) - 1:
                    source_rows = [tip - 1 for tip in tips]

                # Check that the tip-row offset is consistent - i.e. that Evoware will actually do this in one move
                offset = source_rows[0] - (tips[0] - 1)
                for i in range(len(source_rows)):
                    assert (
                        source_rows[i] - (tips[i] - 1) == offset
                    ), "Tip assignment offset inconsistency"

                assert offset <= (
                    getattr(source_op.source, "offset_limit_down", None) or 10000
                ) and offset >= -(
                    getattr(source_op.source, "offset_limit_up", None) or 10000
                ), "Offset limit violated"

                # Perform the aspiration
                self._evo_aspirate(
                    source_op.source,
                    source_op.source.wells[source_rows, source_col],
                    source_op.source.location,
                    list(tips),
                    list(volumes),
                    liquid_class=source_op.liquid_class,
                    label=" + ".join(set([op.label for op in ops]))
                    + ", ops: "
                    + ",".join([str(op.id) for op in ops]),
                    on_underflow=source_op.on_underflow,
                )

            if action == "Dispense":

                # Get the first op in the group
                dest_op = next(iter(ops))
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

                # Check that the tip-row offset is consistent - i.e. that Evoware will actually do this in one move
                offset = dest_rows[0] - (tips[0] - 1)
                for i in range(len(dest_rows)):
                    assert dest_rows[i] - (tips[i] - 1) == offset, "Tip offest issue"

                assert offset <= (
                    dest_op.destination.offset_limit_down or 10000
                ) and offset >= -(
                    dest_op.destination.offset_limit_up or 10000
                ), "Offset limit exceeded"

                # Perform the dispense op
                self._evo_dispense(
                    dest_op.destination,
                    dest_op.destination.wells[dest_rows, dest_col],
                    dest_op.destination.location,
                    list(tips),
                    list(volumes),
                    liquid_class=dest_op.liquid_class,
                    label=" + ".join(set([op.label for op in ops]))
                    + ", ops: "
                    + ",".join([str(op.id) for op in ops]),
                    compositions=compositions,
                )

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
        self._auto_wash(
            tips=list(range(1, 9)), tip_volumes=[950] * 8, wash_schemes=[1] * 8
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

    def commit(self, report=False):

        # If we have ops pending, optimise and apply them
        if len(self.pending_ops) > 0 and not self.processing:
            self.processing = True
            plan, cost, (asp_count, disp_count, wash_count) = make_plan(
                self.pending_ops
            )
            self.execute_plan(plan, (asp_count, disp_count, wash_count))
            self.pending_ops = set()

            # Reset the op tracking on all labware
            for labware in self.all_labware:
                if isinstance(labware, AdvancedLabware):
                    labware.reset_op_tracking()

            self.all_labware = set()
            self.processing = False
            if report:
                print(
                    f"Optimisation complete. aspirates: {asp_count}, dispenses: {disp_count}, washes: {wash_count}"
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

        super().__exit__(exc_type, exc_value, traceback)
