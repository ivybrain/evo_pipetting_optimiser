from evo_pipetting_optimiser import AdvancedLabware, AutoWorklist, AdvancedTrough

"""
This example demonstrates selecting specific tips, or limiting tips used
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

test_trough = AdvancedTrough(
    "Trough1",
    8,
    1,
    min_volume=100,
    max_volume=100_000,
    initial_volumes=80_000,
    location=(2, 1),
)


def example_tip_constraint(worklist):

    with AutoWorklist(worklist, waste_location=(1, 2), cleaner_location=(1, 3)) as wl:

       

        # transfer sample to dilution plate
        # row B and C tip 1
        # row C and D tip 2
        # row E and F tip 3
        wl.auto_transfer(
            sample_plate,
            sample_plate.wells[1:7, 0],
            dilution_plate,
            dilution_plate.wells[2:, 0],
            volumes=50,
            tips=[0,0,1,1,2,2],
            liquid_class="Water free dispense",
            label="Transfer mismatching rows",
        )

        wl.commit()

        # Transfer diluent to dilution plate
        # Optimiser chooses tips from the allowed list
        wl.auto_transfer(
            test_trough,
            test_trough.wells,
            dilution_plate,
            dilution_plate.wells[:8, 0],
            50,
            allowed_tips = [0,2,3,7],
            liquid_class="Water free dispense",
            label="Transfer diluent",
        )

        
        # Print the operations we have registered, so we can sanity check
        print(wl.report)

        wl.commit(report=False)


if __name__ == "__main__":
    example_tip_constraint("example_tip_constraint.gwl")