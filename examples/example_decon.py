import robotools


from ..evo_pipetting_optimiser.robotools_extensions import *

# Define a rack of sample matrix tubes as sample plate. It is a 96-well microplate, so 8 rows, 12 columns
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
)


# Define dilution plate. It is a 96-well microplate, so 8 rows, 12 columns
dilution_plate = AdvancedLabware(
    "dilplate",
    8,
    12,
    min_volume=20,
    max_volume=5000,
    initial_volumes=0,
    location=(32, 2),
)

# Define our tecon trough
decon_trough = robotools.Trough(
    "Decon",
    8,
    1,
    min_volume=3000,
    max_volume=100_000,
    initial_volumes=80_000,
)
decon_trough.location = (2, 1)


def basic_row_column(worklist):

    with AutoWorklist(worklist, waste_location=(1, 2), cleaner_location=(1, 3)) as wl:

        wl.set_wash_parameters(
            decon_troughs=decon_trough, decon_liquid_class="Water free dispense"
        )

        # Transfer as in example_basic_row_column - wash scheme not specified, so using default '1' - standard wash
        wl.auto_transfer(
            sample_plate,
            sample_plate.wells[[0, 1, 2, 3, 5, 6, 7], 0],
            dilution_plate,
            dilution_plate.wells[0, [0, 1, 2, 3, 5, 6, 7]],
            1250,
            liquid_class="Water free dispense",
        )

        # Example as in example_basic_row_column, but this time with wash_scheme="D" - will use decon wash
        wl.auto_transfer(
            sample_plate,
            sample_plate.wells[0, [0, 1, 2, 3, 5, 6, 7]],
            dilution_plate,
            dilution_plate.wells[[0, 1, 2, 3, 5, 6, 7], 0],
            1250,
            liquid_class="Water free dispense",
            wash_scheme="D",
        )

        # Print the operations we have registered, so we can sanity check
        wl.report_ops()

        wl.commit()


if __name__ == "__main__":
    basic_row_column("example_basic_row_column.gwl")
