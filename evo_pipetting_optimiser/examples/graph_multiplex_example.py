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
    offset_limit_up=6,
    offset_limit_down=0
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
    offset_limit_up=4,
    offset_limit_down=0,
    
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


        # Example restricting the tips - the 8 wells in the loop will each be assigned the 8 tips in order
        input_wells = sample_plate.wells[:3, :3].flatten("F")[:8]
        #input_wells[3] = "D02"
        wl.auto_transfer(
            sample_plate,
            input_wells,
            dilution_plate,
            dilution_plate.wells[:4, :2].flatten("F"),
            50,
            liquid_class="Water free dispense",
            label="Transfer duplicates",
        )

        print(wl.report)

        wl.commit(report=True)


if __name__ == "__main__":
    basic_row_column("graph_multiplex_example.gwl")
