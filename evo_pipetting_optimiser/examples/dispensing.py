from evo_pipetting_optimiser import AdvancedLabware, AutoWorklist


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

Dest_Caliper = AdvancedLabware(
    "Dest-Caliper",
    8,
    12,
    min_volume=10,
    max_volume=1400,
    initial_volumes=0,
    location=(19, 1),
)

Dest_SEC = AdvancedLabware(
    "Dest-SEC",
    8,
    12,
    min_volume=10,
    max_volume=1400,
    initial_volumes=0,
    location=(19, 2),
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


sources = [Source_Tubes, Source_Diluent]
dests = [
    Dest_1,
    Dest_Caliper,
    Dest_SEC,
    Dest_AURA,
    Dest_lunatic,
    Dest_DilutionBlock,
    Dest_NeatRetains,
    Dest_DilutedRetains,
]

# Assume first source is fully filled, then second, etc
n_formulations = 6

# Plates to fill up with aliquots column-wise
compact_dests = [
    Dest_1,
]
# aliquot volumes to fill in these plates
compact_aliquot_volumes = [100, 100, 100, 1000, 150, 500, 150, 150, 150]
# ph, osmo, charge variant, DIPA, Amino Acids, Viscosity, Poloxamer, PS80V1, PS80V2


# Plates that will receive a single copy of each formulation
# Lunatic need triplicates
perplate_dests = [Dest_AURA, Dest_DilutionBlock]
# Volume to dispense to each corresponding plate
perplate_aliquot_volumes = [300, 100]
# Lunatic liquid class "Lunatic Dispense"
# Diltion blockm 100 neat, 900 diluent


liquid_class = "Water free dispense"


with AutoWorklist(
    "Dispenser.gwl",
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
        liquid_class=liquid_class,
        wash_scheme="smart",
        label="Dispense lunatic triplicates",
    )

    # Dispense 900 diluent into dilution block
    wl.auto_transfer(
        Source_Diluent,
        Source_Diluent.wells.flatten("F")[:n_formulations],
        Dest_DilutionBlock,
        Dest_DilutionBlock.wells.flatten("F")[:n_formulations],
        900,
        liquid_class=liquid_class,
        wash_scheme="smart",
        label="Dispense diluent",
    )

    # First, process the 'compact' dispenses

    # Track the number of formulations we've processed
    n_processed = 0

    # List of tuple (source, well, aliquot_volumes) for every source well we need to process
    source_wells = []

    # Loop through each source to create a list of every formulation
    for source in sources:

        # Wells in this particular source that have formulation
        this_source_wells = source.wells.flatten("F")[: n_formulations - n_processed]
        source_wells += [
            (source, well, compact_aliquot_volumes.copy()) for well in this_source_wells
        ]
        # Set starting formulation number for this source
        source.form_number_start = n_processed
        n_processed += len(this_source_wells)

        # Check if we've run out of formulations to dispense, and if so, ignore any other source plates
        if n_processed >= n_formulations:
            break

    # List of dest columns, across all plates, in the order we want to fill them
    # In the form of tuples (dest plate, number of rows, column number)
    dest_cols = []
    for dest in compact_dests:
        for col in range(dest.shape[1]):
            dest_cols.append((dest, dest.n_rows, col))

    # Loop through dest columns, and find the appropriate sources to dispense to them
    for dest, n_rows, col in dest_cols:

        # Loop through rows of this dest column
        for row in range(n_rows):

            # If there isn't a corresponding source well for this row, skip
            if row >= len(source_wells):
                continue

            # Get the next recorded aliquod volume for the source well corresponding to this row
            (source, well, aliquot_volumes) = source_wells[row]
            volume = aliquot_volumes.pop(0)
            # Transfer from source to dest
            wl.auto_transfer(
                source,
                well,
                dest,
                dest.wells[row, col],
                volume,
                liquid_class=liquid_class,
                wash_scheme="smart",
            )
        # Remove any source wells from the list that have no aliquot volumes left
        source_wells = [
            (source, well, aliquot_volumes)
            for source, well, aliquot_volumes in source_wells
            if aliquot_volumes
        ]

    # Now, process any per plate dispenses
    # One rack per assay mode
    # Loop through each destination rack to give it a full copy of all formulations
    for dest, volume in zip(perplate_dests, perplate_aliquot_volumes):

        # Track the number of formulations we've processed
        n_processed = 0

        # Loop through each source to get a copy of each formulation
        for source in sources:
            # Check if we've run out of formulations to dispense, and if so get the last well with formulation
            if n_processed >= n_formulations:
                break

            final_well = n_formulations - n_processed
            source_wells = source.wells.flatten("F")[:final_well]

            # Just fill the dest plate column-wise
            # Starting from the next empty well
            dest_wells = dest.wells.flatten("F")[
                n_processed : n_processed + len(source_wells)
            ]

            # Transfer from this source to the dest
            wl.auto_transfer(
                source,
                source_wells,
                dest,
                dest_wells,
                volume,
                liquid_class=liquid_class,
                wash_scheme="smart",
            )

            # Count how many formulations we've processed
            n_processed += len(source_wells)

        # Process one

    # Transfer to Caliper plate Non-reducing then reducing
    diluentblock_wells = (
        list(Dest_DilutionBlock.wells.flatten("F")[:n_formulations]) * 2
    )
    wl.auto_transfer(
        Dest_DilutionBlock,
        diluentblock_wells,
        Dest_Caliper,
        Dest_Caliper.wells[:n_formulations, 0:2].flatten("F"),
        100,
        liquid_class=liquid_class,
        wash_scheme="smart",
        label=f"Caliper diluted",
    )
    # Transfer to SEC plate
    wl.auto_transfer(
        Dest_DilutionBlock,
        Dest_DilutionBlock.wells.flatten("F")[:n_formulations],
        Dest_SEC,
        Dest_SEC.wells.flatten("F")[:n_formulations],
        100,
        label=f"SEC diluted",
        liquid_class=liquid_class,
        wash_scheme="smart",
    )
    # Transfer diluent to aura plate
    wl.auto_transfer(
        Source_Diluent,
        Source_Diluent.wells[:n_formulations],
        Dest_AURA,
        Dest_AURA.wells[:n_formulations, 1],
        300,
        liquid_class=liquid_class,
        wash_scheme="smart",
        label="Add diluent to aura plate",
    )

    # Diluted retains 7 * 100 ul
    for i in range(7):
        wl.auto_transfer(
            Dest_DilutionBlock,
            Dest_DilutionBlock.wells.flatten("F")[:n_formulations],
            Dest_DilutedRetains,
            Dest_DilutedRetains.wells[:n_formulations, i],
            100,
            liquid_class=liquid_class,
            wash_scheme="smart",
            label=f"Diluted retains {i}",
        )
    # Neat retains 130 ul * 7
    for i in range(7):
        wl.auto_transfer(
            Source_Tubes,
            Source_Tubes.wells[:n_formulations],
            Dest_NeatRetains,
            Dest_NeatRetains.wells[:n_formulations, i],
            130,
            liquid_class=liquid_class,
            wash_scheme="smart",
            label=f"Neat retains {i}",
        )

    print(wl.report)
    wl.commit(report=True)
