import robotools


from evo_pipetting_optimiser.robotools_extensions import *

# Define a 12 tube holder for sample tubes
# Often, the LiHa can't go beyond the edges of a tube holder
# offset_limit_up = 0 means tip 1 can't go above the first tube
# offset_limit_down = 4 means tip 8 can't go below the 12th tube
# If you set offset_limit_down = 0, only the first 8 tubes can be accessed
sample_tubes = AdvancedLabware(
    "smp",
    12,
    1,
    min_volume=20,
    max_volume=10000,
    initial_volumes=5000,
    component_names={
        "A01": "sample1",
        "B01": "sample2",
        "C01": "sample3",
        "D01": "sample4",
    },
    location=(2, 1),
    offset_limit_up=0,
    offset_limit_down=4,
)


# Define dilution plate. It is a 96-well microplate, so 8 rows, 12 columns
# Limit the LiHa offset to 2 upwards - ie, tip 8 can reach row F but not row E
dilution_plate = AdvancedLabware(
    "dilplate",
    8,
    12,
    min_volume=20,
    max_volume=5000,
    initial_volumes=1000,
    location=(32, 2),
)


def basic_row_column(worklist):

    with AutoWorklist(worklist, waste_location=(1, 2), cleaner_location=(1, 3)) as wl:

        # Transfer from sample plate to dilution plate

        wl.auto_transfer(
            sample_tubes,
            sample_tubes.wells[:4, 0],
            dilution_plate,
            dilution_plate.wells[:4, 0],
            50,
            liquid_class="Water free dispense",
        )

        wl.auto_transfer(
            sample_tubes,
            sample_tubes.wells[:4, 0],
            dilution_plate,
            dilution_plate.wells[:4, 1],
            50,
            liquid_class="Water free dispense",
        )

        # Print the operations we have registered, so we can sanity check
        wl.report_ops()

        # Optimise these steps and add to worklist
        wl.commit()


if __name__ == "__main__":
    basic_row_column("example_tube_holder.gwl")
