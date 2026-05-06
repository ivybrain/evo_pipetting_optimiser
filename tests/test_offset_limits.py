from evo_pipetting_optimiser import *
from evo_pipetting_optimiser import OffsetLimitException
from itertools import permutations
from random import sample
from .fixtures.invalid_offsets import invalid_offsets


def test_offset_limits():
    # Stress test offset limits by using every possible combination of limits for these transfers
    # The optimiser has internal assertions to check offsets aren't violated, so rely on those for test fails

    limit_values = [None, 0, 1, 2, 3, 4, 5, 6, 7]

    # All possible up and down offsets for smp plate and dilution plate
    possible_offsets = list(permutations(limit_values, 4))

    # Randomly pick 30 offsets to test - good sample, ensures we get some valid and some invalid
    testing_offsets = sample(possible_offsets, k=30)

    for smp_up, smp_down, dil_up, dil_down in testing_offsets:

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
            offset_limit_up=smp_up,
            offset_limit_down=smp_down,
        )

        # Define dilution plate. It is a 96-well microplate, so 8 rows, 12 columns
        # Limit the LiHa offset to 2 upwards - ie, tip 8 can reach row F but not row E
        dilution_plate = AdvancedLabware(
            "dilplate",
            8,
            12,
            min_volume=20,
            max_volume=5000,
            initial_volumes=1000,
            location=(32, 2),
            offset_limit_up=dil_up,
            offset_limit_down=dil_down,
        )

        with AutoWorklist(
            "offset_test.gwl", waste_location=(1, 2), cleaner_location=(1, 3)
        ) as wl:

            # Transfer from sample plate to dilution plate

            wl.auto_transfer(
                sample_plate,
                sample_plate.wells[:, 0],
                dilution_plate,
                dilution_plate.wells[:4, :2],
                50,
                liquid_class="Water free dispense",
            )

            # Transfer from dilution plate to sample plate
            wl.auto_transfer(
                dilution_plate,
                dilution_plate.wells[:, 0],
                sample_plate,
                sample_plate.wells[:4, :2],
                50,
                liquid_class="Water free dispense",
            )

            try:
                # Optimise these steps and add to worklist
                wl.commit(report=False)

                # If the optimisation succeeds, this combo should not be in the invalid offsets list
                assert (smp_up, smp_down, dil_up, dil_down) not in invalid_offsets, f"{(smp_up, smp_down, dil_up, dil_down)} should not be possible"
            # Some transfers are not possible with offset limits that are too restrictive
            # With no offsets allowed on either plate, you can only transfer between identical rows
            except OffsetLimitException:
                assert (smp_up, smp_down, dil_up, dil_down) in invalid_offsets, f"{(smp_up, smp_down, dil_up, dil_down)} should be possible"


if __name__ == "__main__":
    test_offset_limits()
