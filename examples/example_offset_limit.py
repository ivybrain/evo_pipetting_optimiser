import robotools


from evo_pipetting_optimiser import *

# Define a rack of sample matrix tubes as sample plate. It is a 96-well microplate, so 8 rows, 12 columns
# Limit the LiHa offset to 4 downwards - ie, tip 1 can reach row E but not row F
sample_plate = AdvancedLabware(
    "smp",
    8,
    12,
    min_volume=20,
    max_volume=10000,
    initial_volumes=5000,
    component_names={
        "A01": "sample1",
        "B01": "sample2",
        "C01": "sample3",
        "D01": "sample4",
    },
    location=(41, 3),
    offset_limit=(4),
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
    offset_limit=-2,
)


def basic_row_column(worklist):

    with AutoWorklist(worklist, waste_location=(1, 2), cleaner_location=(1, 3)) as wl:

        # Transfer from sample plate to dilution plate

        wl.auto_transfer(
            sample_plate,
            sample_plate.wells[:, 0],
            dilution_plate,
            dilution_plate.wells[:4, :2],
            50,
            liquid_class="Water free dispense",
        )

        # Transfer from dilution plate to sample plate
        wl.auto_transfer(
            dilution_plate,
            dilution_plate.wells[:, 0],
            sample_plate,
            sample_plate.wells[:4, :2],
            50,
            liquid_class="Water free dispense",
        )

        # Print the operations we have registered, so we can sanity check
        wl.report_ops()

        # Optimise these steps and add to worklist
        wl.commit()


if __name__ == "__main__":
    basic_row_column("example_offset_limit.gwl")
