from evo_pipetting_optimiser import AdvancedLabware, AutoWorklist

"""
This example shows bulk pipetting samples into various plates for future use.
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

Dest_DilutedBulk = AdvancedLabware(
    "Dest-DilutedBulk",
    8,
    12,
    min_volume=10,
    max_volume=1400,
    initial_volumes=0,
    location=(10, 2),
)

Dest_NeatBulk = AdvancedLabware(
    "Dest-NeatBulk",
    8,
    12,
    min_volume=10,
    max_volume=1400,
    initial_volumes=0,
    location=(10, 3),
)

Dest_Plate1 = AdvancedLabware(
    "Dest-Plate1",
    8,
    12,
    min_volume=10,
    max_volume=300,
    initial_volumes=0,
    location=(19, 3),
)

Dest_Plate2 = AdvancedLabware(
    "Dest-Plate2",
    8,
    12,
    min_volume=10,
    max_volume=80,
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

# Assume first source is fully filled, then second, etc
n_samples = 6

# sample volumes to fill in bulk plate
sample_volumes = [100, 100, 100, 1000, 150, 500, 150, 150, 150]


neat_liquid_class = "LC03 100 mg/mL Separate tips"
dilute_liquid_class = "LC01 0-10 mg/mL Separate tips"


with AutoWorklist(
    "Dispenser.gwl",
    waste_location=(1, 2),
    cleaner_location=(1, 3),
) as wl:

    # Transfer triplicates plate 2
    wl.auto_transfer(
        Source_Tubes,
        list(Source_Tubes.wells[:n_samples]) * 3,
        Dest_Plate2,
        Dest_Plate2.wells[:n_samples, 0:3].flatten("F"),
        4,
        liquid_class=neat_liquid_class,
        wash_scheme="smart",
        label="Dispense triplicates",
    )

    # Dispense 900 diluent into dilution block
    wl.auto_transfer(
        Source_Diluent,
        Source_Diluent.wells.flatten("F")[:n_samples],
        Dest_DilutionBlock,
        Dest_DilutionBlock.wells.flatten("F")[:n_samples],
        900,
        liquid_class="Water free dispense",
        wash_scheme="smart",
        label="Dispense diluent",
    )

    for i, sample in enumerate(sample_volumes):
        wl.auto_transfer(
            Source_Tubes,
            Source_Tubes.wells[:n_samples],
            Dest_1,
            Dest_1.wells[:n_samples, i],
            sample,
            liquid_class=neat_liquid_class,
            wash_scheme="smart",
            label="Dispense bulk samples",
        )

    # Transfer to individual plates
    wl.auto_transfer(
        Source_Tubes,
        Source_Tubes.wells[:n_samples],
        Dest_Plate1,
        Dest_Plate1.wells[:n_samples, 0],
        300,
        liquid_class=neat_liquid_class,
        wash_scheme="smart",
    )

    # Transfer sample to dilution block
    wl.auto_transfer(
        Source_Tubes,
        Source_Tubes.wells[:n_samples],
        Dest_DilutionBlock,
        Dest_DilutionBlock.wells[:n_samples, 0],
        100,
        liquid_class=neat_liquid_class,
        wash_scheme="smart",
    )

    # Diluted bulk
    for i in range(7):
        wl.auto_transfer(
            Dest_DilutionBlock,
            Dest_DilutionBlock.wells.flatten("F")[:n_samples],
            Dest_DilutedBulk,
            Dest_DilutedBulk.wells[:n_samples, i],
            100,
            liquid_class=dilute_liquid_class,
            wash_scheme="smart",
        )
    # Neat Bulk
    for i in range(7):
        wl.auto_transfer(
            Source_Tubes,
            Source_Tubes.wells[:n_samples],
            Dest_NeatBulk,
            Dest_NeatBulk.wells[:n_samples, i],
            130,
            liquid_class=neat_liquid_class,
            wash_scheme="smart",
        )

    print(wl.report)
    wl.commit(report=True)
