import robotools
import json

from evo_pipetting_optimiser import AdvancedLabware, AutoWorklist, AdvancedTrough

"""
This example shows using disposable tips (DiTis)
"""


diluent = AdvancedTrough(
    "Diluent",
    8,
    1,
    min_volume=100,
    max_volume=100_000,
    initial_volumes=80_000,
    location=(8, 3),
)

microplate = AdvancedLabware(
    "Microplate", 8, 12, min_volume=20, max_volume=340, location=(32, 1)
)

DWP = AdvancedLabware(
    "DWP", 8, 12, min_volume=50, max_volume=2000, initial_volumes=0, location=(38, 3)
)

tubes = AdvancedLabware(
    "SampleTubes",
    8,
    12,
    min_volume=200,
    max_volume=50_000,
    initial_volumes=50_000,
    location=(9, 1),
)


def basic_row_column(worklist):

    # Use LiHa 2 (arm=1)
    with AutoWorklist(
        worklist, diti_mode=True, default_tip=16, arm=1, waste_location=(25, 3)
    ) as wl:

        # wl.set_wash_parameters(max_tip_reuse=2)

        sample_count = 4

        # Transfer neat sample to first column
        # Wash scheme is default (1) so new tips will be fetched then discarded
        wl.auto_transfer(
            tubes,
            tubes.wells[:sample_count, 0],
            DWP,
            DWP.wells[:sample_count, 0],
            500,
            liquid_class="Ivy Training",
            label="Get neat sample",
        )

        # Wash scheme is 'smart', so same tips will be used for all diluent
        # But different tips will be used where concentration differs
        for i in range(5):
            # Transfer diluent
            wl.auto_transfer(
                diluent,
                diluent.wells[:sample_count],
                DWP,
                DWP.wells[:sample_count, i + 1],
                250,
                liquid_class="Ivy training",
                label="Dispense diluent",
                wash_scheme="smart",
            )
            # Series dilute from this column to next
            wl.auto_transfer(
                DWP,
                DWP.wells[:sample_count, i],
                DWP,
                DWP.wells[:sample_count, i + 1],
                250,
                liquid_class="Ivy Training",
                label=f"Series diluting into column {i+1}",
                wash_scheme="smart",
            )

        # Print the operations we have registered, so we can sanity check
        print(wl.report)

        wl.commit(report=True)


if __name__ == "__main__":
    basic_row_column("example_ditis.gwl")
