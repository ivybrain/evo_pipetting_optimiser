import robotools


from evo_pipetting_optimiser import AdvancedLabware, AutoWorklist

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


def basic_row_column(worklist):

    with AutoWorklist(worklist, waste_location=(1, 2), cleaner_location=(1, 3)) as wl:

        # Basic: Transfer col 0 to row 0
        wl.auto_transfer(
            sample_plate,
            sample_plate.wells[:, 0],
            dilution_plate,
            dilution_plate.wells[0, :8],
            50,
            liquid_class="Water free dispense",
        )

        wl.auto_transfer(
            sample_plate,
            sample_plate.wells[0, :8],
            dilution_plate,
            dilution_plate.wells[:, 0],
            50,
            liquid_class="Water free dispense",
        )

        # # Optimise these steps and add to worklist
        wl.commit()

        # Situation where we transfer a column to a row, with a range across 8 wells but excluding a middle well
        # Will aspirate in one op, leaving tip 5 empty
        # Unlike default Evoware, which would aspirate rows A-D with tips 1-4 then rows F-H with tips 5-7
        # Large volume (>950) means we repeat each op twoce

        wl.auto_transfer(
            sample_plate,
            sample_plate.wells[[0, 1, 2, 3, 5, 6, 7], 0],
            dilution_plate,
            dilution_plate.wells[0, [0, 1, 2, 3, 5, 6, 7]],
            1250,
            liquid_class="Water free dispense",
        )

        # Same situation but transferring row to column
        # Will do several aspirates, but group to one dispense
        wl.auto_transfer(
            sample_plate,
            sample_plate.wells[0, [0, 1, 2, 3, 5, 6, 7]],
            dilution_plate,
            dilution_plate.wells[[0, 1, 2, 3, 5, 6, 7], 0],
            1250,
            liquid_class="Water free dispense",
        )

        # Print the operations we have registered, so we can sanity check
        wl.report_ops()

        wl.commit()


if __name__ == "__main__":
    basic_row_column("example_basic_row_column.gwl")
