import robotools
import json

from evo_pipetting_optimiser import AdvancedLabware, AutoWorklist, AdvancedTrough

"""
This example shows using disposable tips (DiTis) with two different liquid handling arms
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


def example_multi_arm(worklist):

    # Start with LiHa 1 (arm 0)
    with AutoWorklist(
        worklist, diti_mode=True, default_tip=16, arm=0, tip_discard_location=(25, 3), max_volume=4950, # Max volume for 5000ul diluters
    ) as wl:


        sample_count = 8

        # Dispense small sample volume to microplate
        wl.auto_transfer(
            tubes,
            tubes.wells[:sample_count, 0],
            microplate,
            microplate.wells[:sample_count, 0],
            100,
            liquid_class="Ivy Training",
            label="Dispense low volume sample",
            tip_index = 12 # e.g. 200ul tip
        )

        # Dispense diluent to DWP columns
        # none in col 1, 500 ul in col 2, 1000ul in col 3
        # use arm 2 (5ml diluter) with 2ml tips
        wl.auto_transfer( 
            diluent, 
            None,
            DWP,
            DWP.wells[:sample_count, 1:3],
            ([1600] * 8) + ([1800] * 8),
            liquid_class="Ivy Training",
            label="Dispense diluent",
            tip_index = 18 # 2ml tip
        )

        # Dispense sample to DWP columns
        # 2ml in col 1, 400ul in col 2, 200ul in col 3

        wl.auto_transfer(
            tubes,
            tubes.wells[:sample_count, 0],
            DWP,
            DWP.wells[:sample_count, 0],
            2000,
            liquid_class="Ivy Training",
            label="Dispense neat sample",
            tip_index = 18 # 2ml tip
        )

        wl.auto_transfer(
            tubes,
            tubes.wells[:sample_count, 0],
            DWP,
            DWP.wells[:sample_count, 1],
            400,
            liquid_class="Ivy Training",
            label="Dispense neat sample",
            tip_index = 14 # 500 ul
        )

        wl.auto_transfer(
            tubes,
            tubes.wells[:sample_count, 0],
            DWP,
            DWP.wells[:sample_count, 2],
            200,
            liquid_class="Ivy Training",
            label="Dispense neat sample",
            tip_index = 12 # 200ul tip
        )

        # Print the operations we have registered, so we can sanity check
        print(wl.report)


        # Commit ops with volumes less than 800ul  to arm 0 (small volume)
        wl.commit(arm=0, filter=lambda op: op.volume < 800, report=True)
        # We see from the progress bar that we have committed 8 ops out of 48

        # print the remaining ops after this commit
        print(wl.report)

        # commit ops with volumes >=800ul to arm 1 (large volume)
        wl.commit(arm=1, filter=lambda op: op.volume>=800, report=True)
        # We see from the progress bar we have committed 24 ops out of 40

        # print the remaining ops after this commit
        print(wl.report)

        #OUTPUT:
        # 32:      SampleTubes-A01 to DWP-A02              400.0ul       Dispense neat sample
        # 33:      SampleTubes-B01 to DWP-B02              400.0ul       Dispense neat sample
        # 34:      SampleTubes-C01 to DWP-C02              400.0ul       Dispense neat sample
        # 35:      SampleTubes-D01 to DWP-D02              400.0ul       Dispense neat sample
        # 36:      SampleTubes-E01 to DWP-E02              400.0ul       Dispense neat sample
        # 37:      SampleTubes-F01 to DWP-F02              400.0ul       Dispense neat sample
        # 38:      SampleTubes-G01 to DWP-G02              400.0ul       Dispense neat sample
        # 39:      SampleTubes-H01 to DWP-H02              400.0ul       Dispense neat sample
        # 40:      SampleTubes-A01 to DWP-A03              200.0ul       Dispense neat sample
        # 41:      SampleTubes-B01 to DWP-B03              200.0ul       Dispense neat sample
        # 42:      SampleTubes-C01 to DWP-C03              200.0ul       Dispense neat sample
        # 43:      SampleTubes-D01 to DWP-D03              200.0ul       Dispense neat sample
        # 44:      SampleTubes-E01 to DWP-E03              200.0ul       Dispense neat sample
        # 45:      SampleTubes-F01 to DWP-F03              200.0ul       Dispense neat sample
        # 46:      SampleTubes-G01 to DWP-G03              200.0ul       Dispense neat sample
        # 47:      SampleTubes-H01 to DWP-H03              200.0ul       Dispense neat sample

        # These ops were not completed in the first commit, because they dispense to the DWP after the diluent
        # has been added - so they were blocked until the diluent was dispensed by the larger arm
        # doing arm 1 first would have let it take only 2 commits, but this demonstrates the behaviour
        
        # Commit any remaining volumes < 800 to arm 0
        wl.commit(arm=0, filter=lambda op: op.volume < 800, report=True)

        # Double check we haven't missed anything!
        if len(wl.pending_ops) > 0:
            raise Exception("We missed some ops!")


if __name__ == "__main__":
    example_multi_arm("example_multi_arm.gwl")
