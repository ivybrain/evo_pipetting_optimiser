from evo_pipetting_optimiser import AdvancedLabware, AutoWorklist, AdvancedTrough

# Define a rack of sample matrix tubes as sample plate. It is a 96-well microplate, so 8 rows, 12 columns


diluent = AdvancedTrough(
    "Diluent",
    8,
    1,
    min_volume=100,
    max_volume=100_000,
    initial_volumes=80_000,
    location=(8, 3),
)
# Define dilution plate. It is a 96-well microplate, so 8 rows, 12 columns

microplate = AdvancedLabware(
    "Microplate", 8, 12, min_volume=20, max_volume=340, location=(32, 1)
)

DWP = AdvancedLabware(
    "DWP", 8, 12, min_volume=50, max_volume=2000, initial_volumes=0, location=(38, 3)
)

tubes = AdvancedLabware(
    "SampleTubes",
    8,
    12,
    min_volume=200,
    max_volume=50_000,
    initial_volumes=50_000,
    location=(9, 1),
)


def test_ditis():
    """
    Test behaviour works as expected with DiTis and smart wash. Same as examples/example_ditis.py
    """

    with AutoWorklist(
        "test_ditis.gwl", diti_mode=True, default_tip=16, arm=1, tip_discard_location=(25, 3)
    ) as wl:

        sample_count = 4

        # Transfer neat sample to first column
        wl.auto_transfer(
            tubes,
            tubes.wells[:sample_count, 0],
            DWP,
            DWP.wells[:sample_count, 0],
            500,
            liquid_class="Ivy Training",
            label="Get neat sample",
        )

        for i in range(5):
            # Transfer diluent
            wl.auto_transfer(
                diluent,
                diluent.wells[:sample_count],
                DWP,
                DWP.wells[:sample_count, i + 1],
                250,
                liquid_class="Ivy training",
                label="Dispense diluent",
                wash_scheme="smart",
            )
            # Series dilute from this column to next
            wl.auto_transfer(
                DWP,
                DWP.wells[:sample_count, i],
                DWP,
                DWP.wells[:sample_count, i + 1],
                250,
                liquid_class="Ivy Training",
                label=f"Series diluting into column {i+1}",
                wash_scheme="smart",
            )

        wl.commit(report=True)

        # Ensure it completes this task efficiently as expected
        assert wl.asp_count == 9
        assert wl.disp_count == 11
        assert wl.wash_count == 5


if __name__ == "__main__":
    test_ditis()
