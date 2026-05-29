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


def test_incomplete_rows():

    with AutoWorklist(
        "test_incomplete_rows.gwl", waste_location=(1, 2), cleaner_location=(1, 3)
    ) as wl:

        wl.auto_transfer(
            sample_plate,
            list(sample_plate.wells[:, 0]) * 9,
            dilution_plate,
            dilution_plate.wells[:6, :],
            50,
            liquid_class="Water free dispense",
            label="Transfer duplicates",
        )

        wl.commit(report=True)

        assert wl.asp_count == 9
        assert wl.disp_count == 21
        assert wl.wash_count == 10


if __name__ == "__main__":
    test_incomplete_rows()
