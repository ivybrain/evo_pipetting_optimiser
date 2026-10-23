from evo_pipetting_optimiser import AdvancedLabware, AutoWorklist

"""
This example shows aliquoting samples for various assays and also retains.
It makes use of 'smart wash' to avoid repeated washing between handling of the sample

"""

Source_Tubes = AdvancedLabware(
    "Source-Tubes",
    16,
    1,
    min_volume=50,
    max_volume=15000,
    initial_volumes=15000,
    location=(2, 1),
)

Source_Diluent = AdvancedLabware(
    "Source-Diluent",
    16,
    1,
    min_volume=50,
    max_volume=15000,
    initial_volumes=15000,
    location=(3, 1),
)

Dest_1 = AdvancedLabware(
    "Dest-1", 8, 12, min_volume=10, max_volume=1400, initial_volumes=0, location=(10, 1)
)

Dest_DilutedRetains = AdvancedLabware(
    "Dest-DilutedRetains",
    8,
    12,
    min_volume=10,
    max_volume=1400,
    initial_volumes=0,
    location=(10, 2),
)

Dest_NeatRetains = AdvancedLabware(
    "Dest-NeatRetains",
    8,
    12,
    min_volume=10,
    max_volume=1400,
    initial_volumes=0,
    location=(10, 3),
)

Dest_AURA = AdvancedLabware(
    "Dest-AURA",
    8,
    12,
    min_volume=10,
    max_volume=340,
    initial_volumes=0,
    location=(19, 3),
)

Dest_lunatic = AdvancedLabware(
    "Dest-lunatic",
    8,
    12,
    min_volume=10,
    max_volume=50,
    initial_volumes=0,
    location=(19, 4),
)

Dest_DilutionBlock = AdvancedLabware(
    "Dest-DilutionBlock",
    8,
    12,
    min_volume=0,
    max_volume=2000,
    initial_volumes=0,
    location=(28, 2),
)


def test_dispensing():

    # Assume first source is fully filled, then second, etc
    n_formulations = 6

    # aliquot volumes to fill in bulk plate
    aliquot_volumes = [100, 100, 100, 1000, 150, 500, 150, 150, 150]

    neat_liquid_class = "LC03 100 mg/mL Separate tips"
    dilute_liquid_class = "LC01 0-10 mg/mL Separate tips"

    with AutoWorklist(
        "test_Dispenser.gwl",
        waste_location=(1, 2),
        cleaner_location=(1, 3),
    ) as wl:

        # Transfer triplicates to lunatic plate
        wl.auto_transfer(
            Source_Tubes,
            list(Source_Tubes.wells[:n_formulations]) * 3,
            Dest_lunatic,
            Dest_lunatic.wells[:n_formulations, 0:3].flatten("F"),
            4,
            liquid_class=neat_liquid_class,
            wash_scheme="smart",
            label="Dispense triplicates",
        )

        # Dispense 900 diluent into dilution block
        wl.auto_transfer(
            Source_Diluent,
            Source_Diluent.wells.flatten("F")[:n_formulations],
            Dest_DilutionBlock,
            Dest_DilutionBlock.wells.flatten("F")[:n_formulations],
            900,
            liquid_class="Water free dispense",
            wash_scheme="smart",
            label="Dispense diluent",
        )

        for i, aliquot in enumerate(aliquot_volumes):
            wl.auto_transfer(
                Source_Tubes,
                Source_Tubes.wells[:n_formulations],
                Dest_1,
                Dest_1.wells[:n_formulations, i],
                aliquot,
                liquid_class=neat_liquid_class,
                wash_scheme="smart",
                label="Dispense bulk aliquots",
            )

        # Transfer to individual plates
        wl.auto_transfer(
            Source_Tubes,
            Source_Tubes.wells[:n_formulations],
            Dest_AURA,
            Dest_AURA.wells[:n_formulations, 0],
            300,
            liquid_class=neat_liquid_class,
            wash_scheme="smart",
        )

        # Transfer sample to dilution block
        wl.auto_transfer(
            Source_Tubes,
            Source_Tubes.wells[:n_formulations],
            Dest_DilutionBlock,
            Dest_DilutionBlock.wells[:n_formulations, 0],
            100,
            liquid_class=neat_liquid_class,
            wash_scheme="smart",
        )

        # Diluted retains
        for i in range(7):
            wl.auto_transfer(
                Dest_DilutionBlock,
                Dest_DilutionBlock.wells.flatten("F")[:n_formulations],
                Dest_DilutedRetains,
                Dest_DilutedRetains.wells[:n_formulations, i],
                100,
                liquid_class=dilute_liquid_class,
                wash_scheme="smart",
                label=f"Diluted retains {i}",
            )
        # Neat retains
        for i in range(7):
            wl.auto_transfer(
                Source_Tubes,
                Source_Tubes.wells[:n_formulations],
                Dest_NeatRetains,
                Dest_NeatRetains.wells[:n_formulations, i],
                130,
                liquid_class=neat_liquid_class,
                wash_scheme="smart",
                label=f"Neat retains {i}",
            )

        wl.commit(report=True)
        assert wl.asp_count == 30
        assert wl.disp_count == 30
        assert wl.wash_count == 9


if __name__ == "__main__":
    test_dispensing()
