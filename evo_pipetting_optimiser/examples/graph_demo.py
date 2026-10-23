from evo_pipetting_optimiser import AdvancedLabware, AutoWorklist, AdvancedTrough

"""
Used to generate the main graph shown in the optimisation_details.md file
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


        wl.auto_transfer(
            sample_plate,
            sample_plate.wells[[0,2], 0],
            dilution_plate,
            dilution_plate.wells[[0,1], [ 0]],
            600,
            liquid_class="Water free dispense",
            label="transfer noncontinuous rows to col",
        )

        wl.auto_transfer(
            sample_plate,
            sample_plate.wells[1, [2,3]],
            dilution_plate,
            dilution_plate.wells[[0,1], [ 1]],
            600,
            liquid_class="Water free dispense",
            label="transfer noncontinuous rows to col",
        )

       
        print(wl.report)

        wl.commit(report=True)


if __name__ == "__main__":
    basic_row_column("example_basic_row_column.gwl")
