from evo_pipetting_optimiser import AdvancedLabware, AutoWorklist, AdvancedTrough

"""
This example demonstrates a range of 'tricky' transfers between two plates that standard transfers do not perform optimally
e.g. transfering duplicates, transfering between row layouts and column layouts, and transferring non continuous groups of rows
"""

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
    offset_limit_down=2,
    offset_limit_up=2,
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
    offset_limit_down=2,
    offset_limit_up=2
)

test_trough = AdvancedTrough(
    "Trough1",
    8,
    1,
    min_volume=100,
    max_volume=100_000,
    initial_volumes=80_000,
    location=(2, 1),
)


def basic_row_column(worklist):

    with AutoWorklist(worklist, waste_location=(1, 2), cleaner_location=(1, 3)) as wl:

        # wl.auto_transfer(
        #     sample_plate,
        #     [sample_plate.wells[i // 2, 0] for i in range(8)],
        #     dilution_plate,
        #     dilution_plate.wells[:, 0],
        #     50,
        #     liquid_class="Water free dispense",
        #     label="Transfer duplicates",
        # )

        # # transfer sampel to dilution plate
        # wl.auto_transfer(
        #     sample_plate,
        #     sample_plate.wells[1:7, 0],
        #     dilution_plate,
        #     dilution_plate.wells[2:, 0],
        #     volumes=50,
        #     liquid_class="Water free dispense",
        #     label="Transfer mismatching rows",
        # )

        # # Transfer diluent to dilution plate
        # wl.auto_transfer(
        #     test_trough,
        #     test_trough.wells,
        #     dilution_plate,
        #     dilution_plate.wells[:8, 0],
        #     50,
        #     liquid_class="Water free dispense",
        #     label="Transfer diluent",
        # )

        # Transfer col 0 to row 0
        # wl.auto_transfer(
        #     sample_plate,
        #     sample_plate.wells[:, 0],
        #     dilution_plate,
        #     dilution_plate.wells[0, :8],
        #     50,
        #     liquid_class="Water free dispense",
        #     label="Transfer column to row",
        # )

        # wl.auto_transfer(
        #     sample_plate,
        #     sample_plate.wells[0, :8],
        #     dilution_plate,
        #     dilution_plate.wells[:, 0],
        #     50,
        #     liquid_class="Water free dispense",
        #     label="transfer row to column",
        # )

        # # Now transfer back to trough
        # wl.auto_transfer(
        #     dilution_plate,
        #     dilution_plate.wells[:, 0],
        #     test_trough,
        #     test_trough.wells,
        #     50,
        #     liquid_class="Water free dispense",
        #     label="transfer to trough",
        # )

        # Situation where we transfer a column to a row, with a range across 8 wells but excluding a middle well
        # Will aspirate in one op, leaving tip 5 empty
        # Unlike default Evoware, which would aspirate rows A-D with tips 1-4 then rows F-H with tips 5-7
        # Large volume (>950) means we repeat each op twice

        wl.auto_transfer(
            sample_plate,
            sample_plate.wells[[0, 2], 0],
            dilution_plate,
            dilution_plate.wells[0, [ 1, 2,]],
            600,
            liquid_class="Water free dispense",
            label="transfer noncontinuous rows to col",
        )

        wl.auto_transfer(
            sample_plate,
            sample_plate.wells[1, 2:4],
            dilution_plate,
            dilution_plate.wells[0:2, 8],
            600,
            liquid_class="Water free dispense",
            label="transfer noncontinuous rows to col",
        )

        # Same situation but transferring one row to many cols
        # Will do several aspirates, but group to one dispense
        # wl.auto_transfer(
        #     sample_plate,
        #     sample_plate.wells[0, [0, 1, 2, 3, 5, 6, 7]],
        #     dilution_plate,
        #     dilution_plate.wells[[0, 1, 2, 3, 5, 6, 7], 0],
        #     600,
        #     liquid_class="Water free dispense",
        #     label="Transfer noncontinuous cols to rows",
        # )

        # Print the operations we have registered, so we can sanity check
        print(wl.report)

        wl.commit(report=False)


if __name__ == "__main__":
    basic_row_column("example_basic_row_column.gwl")
