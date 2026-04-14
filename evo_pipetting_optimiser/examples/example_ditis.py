import robotools
import json

from evo_pipetting_optimiser import AdvancedLabware, AutoWorklist, AdvancedTrough

# Define a rack of sample matrix tubes as sample plate. It is a 96-well microplate, so 8 rows, 12 columns


WB_trough = AdvancedTrough(
    "WB",
    8,
    1,
    min_volume=100,
    max_volume=100_000,
    initial_volumes=80_000,
    location=(8, 1),
)
# Define dilution plate. It is a 96-well microplate, so 8 rows, 12 columns

microplate = AdvancedLabware(
    "Microplate", 8, 12, min_volume=20, max_volume=340, location=(32, 1)
)

DWP = AdvancedLabware(
    "DWP", 8, 12, min_volume=50, max_volume=2000, initial_volumes=2000, location=(38, 1)
)


def basic_row_column(worklist):

    with AutoWorklist(
        worklist, diti_mode=True, default_tip=9, arm=1, waste_location=(25, 3)
    ) as wl:

        wl.auto_transfer(
            WB_trough,
            WB_trough.wells[0],
            microplate,
            microplate.wells[2, 0],
            50,
            liquid_class="Water free dispense",
        )
        wl.auto_transfer(
            WB_trough,
            WB_trough.wells[0],
            microplate,
            microplate.wells[7, 0],
            50,
            liquid_class="Water free dispense",
        )

        wl.commit()

        wl.auto_transfer(
            DWP,
            DWP.wells[2:4, 0],
            microplate,
            microplate.wells[2:4, 0],
            50,
            liquid_class="Water free dispense",
        )

        # Print the operations we have registered, so we can sanity check
        print(wl.report)

        wl.commit(report=True)


if __name__ == "__main__":
    basic_row_column(r"\\PME1SVM001L.lab.net\rla\FreedomEvo200_L6\optimiser_ditis.gwl")
