# Evo_Pipetting_Optimiser

Published by CSL Innovation PTY LTD under the GNU Affero General Public License v3.0. CSL is a global biotechnology leader that develops and delivers innovative lifesaving medicines, protects public health and helps people with life-threatening medical conditions live full lives.

## Introduction
[robotools](https://github.com/JuBiotech/robotools) provides a simple python interface for writing Tecan Liquid Handler worklists to perform pipetting workflows. Most commonly used is the `EvoWorklist.transfer()` method, which produces Tecan EVO worklists that use basic worklist commands (`A`, `D`, and `W`). These commands are sufficient for defining pipetting workflows, but have several limitations.

A frequent challenge is the flawed optimisation that EVOware conducts for basic worklist commands (`A` and `D`),
as produced by `EvoWorklist.transfer()`. The strategy it selects can often make pipetting slower, meaning that `B;` commands 
(produced by `EvoWorklist.commit()`) must be inserted between commands to stop the EVO from rearranging them inefficiently. It can even cause intermittent errors, where, if sample positions change, it may attempt to pipette a well with a tip which cannot reach that well, if the labware is near the top or bottom of the deck.

Additionally, a common use case for robotools is dilution/processing/preparation of several samples in a plate. In such cases,
the easiest way to translate the process to Python and robotools is to loop through each sample, and define the pipetting steps which 
must be conducted for that sample to be processed. Unfortunately, this leads to inefficient worklist commands, which pipette one sample at
a time. Previously, careful design and more complicated code was employed to pre-calculate the operations required for all samples,
then make fewer calls to `transfer()`, ideally only one (covering all samples) for each step in the process. This allowed several samples to
be pipetted at once, at the cost of making code more difficult to write and understand. Additionally, it could still fall victim to poor
optimisation from the EVOware side, and miss potential speed.

Thus an optimiser which tracks all operations a user wishes to conduct, before automatically grouping them into the most efficient order,
is desired. This package provides this, by providing an `auto_transfer` method which records the desired operations for later optimisation. When `commit` is called or the worklist is saved, the recorded operations are optimised and added to the worklist as advanced worklist commands.

WARNING: Like any advanced worklist commands, commands produced by the optimiser can cause damage if not specified correctly. Ensure that all grid and site positions are defined correctly, and that this matches the deck layout set in the EVOware script.

### In scope
Optimisation of pipetting operations on the Tecan Evo.
8-tip LiHa arms with either fixed or disposable tips are supported.

Provision of an interface with minimal differences to existing robotools `EvoWorklist` and `transfer` funtionality so that operation is
understandable to existing users.

Maintenance of a given well's pipetting order - operations on a particular well are conducted in the order that they're specified in code.

Support for the existing composition tracking and reporting functionality in robotools.

### Out of scope
Other liquid handlers or arms.

Optimisation by using tip spreading - tips will only move with the minimum defined spread for each labware. Spreading tips to access seperated wells with subsequent tips would add significant complexity and is far more likely to cause errors from trying to move tips outside the bounds of the EVO.

## Overview and usage

### Installation

This package can be installed by:

1. `cd evo_pipetting_optimiser`
2. `pip install .`

### Deinstallation

You can just uninstall the package with:
 ```
 pip uninstall evo_pipetting_optimiser
 ```

### Usage
First import AutoWorklist and AdvancedLabware, and AdvancedTrough (if needed):
```
from evo_pipetting_optimiser import AdvancedLabware, AutoWorklist, AdvancedTrough
```

Create AdvancedLabware just as you would Labware, but with the added `location` parameter set to a tuple (grid, site) of the location of the labware on the EVO deck:
```
sample_plate = AdvancedLabware(
    "smp",
    8,
    12,
    min_volume=20,
    max_volume=10000,
    initial_volumes=5000,
    location=(41, 3),
)
```

The same applies for troughs:

```
reducing_trough = AdvancedTrough(
        "Reducing",
        8,
        1,
        min_volume=50,
        max_volume=25_000,
        initial_volumes=25_000,
        location=(4,2)
    )
```

Then, create an `AutoWorklist`. This is the same as `EvoWorklist`, but with some extra arguments relating to washes.
If using fixed tips, waste_location and cleaner_location must be specified as (grid, site) tuples.
```
with AutoWorklist("test.gwl", waste_location=(1, 2), cleaner_location=(1, 3),) as wl:
```

If using disposable tips (DiTis), set diti_mode to True, set default_tip to the integer used to represent the desired tip type in evoware,
and set `tip_discard_location` as the (grid, site) tuple where used tips should be discarded. 
```
with AutoWorklist("example.gwl", diti_mode=True, default_tip=16, arm=1, tip_discard_location=(25, 3))
```

All `EvoWorklist` commands can be used as normal. For optimisation, the `auto_transfer` method must be used. This is nearly identical to `transfer` in use. The only difference is in the wash_scheme argument; "D", "smart", 1, and None are supported. Additionally, if using DiTis, the tip_index argument can be used. This will set the tip type to a specific index in evoware for that operation. If not specified, the default_tip set in the AutoWorklist class will be used.
```
wl.auto_transfer(
            sample_plate,
            sample_plate.wells[0, 1],
            dilution_plate,
            dilution_plate.wells[1, 0],
            1250,
            wash_scheme="smart",
            tip_index=9,
            liquid_class="Water free dispense",
)
```

When you are ready to optimise a set of operations and write them to the worklist, use 'wl.commit()'. Alternatively, exiting the 'with block' will also optimise and write operations to the worklist.

#### Large volume handling
Large volume handling is supported just as in EvoWorklist. AutoWorklist can be passed a max_volume argument (default 950ul), and any volumes greater than that will be split into several operations with volumes less than the maximum. Note: operations created this way may not execute sequentially - they are optimised just like any other operations.


#### Washes
Washes are added with advanced worklist commands, rather than the usual basic W command - thus, seperate numbered schemes as defined in evoware are not available. Instead, wash parameters can be set with the set_wash_parameters method, i.e. 
`wl.set_wash_parameters(waste_vol=5.0)`. All arguments available to the `evo_wash` method can be set as parameters here, and will be applied to all washes generated by `auto_transfer`.

Additionally, there is a `wash_scheme="smart"`. This uses the composition tracking of robotools to check the list of compositions a tip has touched since it was last washed. If it has only touched liquids with the same composition (concentrations of components may vary up to 20%), it will skip a wash and touch that liquid again, as there is little risk of contamination. The optimiser will attempt to arrange operations to avoid washes, i.e. to handle the same liquid many times with a tip before washing. This is very useful when doing bulk dispenses of material.

If you do not want washes, you can use `auto_transfer` with `wash_scheme=None`. This will avoid creating wash commands for the operations. This functionality is unreliable with basic worklists, so providing it here gives additional utility. However, use this feature cautiously as it prevents tips from being washed even when they touch different liquids.

In DiTi mode, GetDiti and DropDiti commands will be generated instead of wash commands. These commands will use the tip_discard_location set in AutoWorklist to drop tips, and the airgap and airgap_speed parameters from set_wash_parameters. `None` or `"smart"` also work here to skip tip replacements.

When using `None` or `"smart"` washes, it is prudent to check your liquid class can handle repeated aspirates and dispenses without system liquid contamination. Often it is useful to insert a wash anyway between every n aspirates/dispenses (say, 10) to reset the airgap and ensure performance remains consistent. This can be set with `wl.set_wash_parameters(max_tip_uses=10)`. This will ensure a tip is used at most 10 times (10 aspirate-dispense cycles) before it is washed.

Decontamination washes are supported, through adding one or more decon troughs.
To set decon washes, specify decon troughts using `wl.set_wash_parameters(decon_troughs=[trough1, trough2])` then use transfers with `wash_scheme="D"`. The first trough in the list will be used until the volume tracking determines it does not have sufficient volume, then the next trough will be used. Thus, the switchover point will depend on the `initial_volumes` that the user has set for that Trough. Additional decon wash parameters can also be set with `set_wash_parameters`:
- `decon_liquid_class` - The liquid class for aspirating decon solution. Must be set before any decon is used.
- `decon_delay` - Delay to apply after aspirating the decon solution, to soak the insides of the tips
- `decon_excess_volume` - Volume of decon aspirated by each tip will be set equal to the volume that was aspirated by that tip in the transfer step, plus this excess volume. This is superior to EVOware decon, where only a single fixed volume can be set, and thus the maximum desired must be set and will be used for all decons.

Only one set of parameters can be applied per optimisation. If you wish to change wash parameters, first apply the previous optimisation with `commit()`, then, change the parameters for the coming operations.

Likewise, commands other than `auto_transfer` cannot be added while optimisation is pending and has not been commited, as any other commands would be added to the worklist before the pending commands. Call `commit()` before using any other commands. If you do not call commit, it will be called automatically and generate a warning.

#### Offsets
To prevent the optimiser trying to move tips further than is physically possible on the deck, `offset_limit_up` and `offset_limit_down` can be set as keyword arguments of `AdvancedLabware`.

`offset_limit_up` Limits the vertical translation upwards (towards the back of the evo) of tips accessing the labware, when used with `auto_transfer`
This is used for labware near the top of the evo deck, where not every row can be accessed by every tip
If set to an int, limits the upwards translation of the tips: 4 means the tips can shift no more than 4 rows above
their default position. I.e. tip 5 can access row A, but tip 6 cannot
If 0, no upwards shift is allowed
If none, no offset limit and any offsets are permitted
Setting a limit will reduce optimisation effectiveness and increase pipetting time.

`offset_limit_down` likewise limits the movement of the tips downwards, useful for labware near the bottom of the evo deck.


#### Reporting
After several `auto_transfer` operations but before a commit, you can use `AutoWorklist.report` to see a list of operations that are pending.
```
print(wl.report)
```

A report will look like this, with the operation ID, the name and well of the source and destination, and the volume.
```
16:              smp-A01 to dilplate-A01         625.0ul
17:              smp-A01 to dilplate-A01         625.0ul
18:              smp-B01 to dilplate-A02         625.0ul
19:              smp-B01 to dilplate-A02         625.0ul
```

After a commit, the volumes and compositions of labware will reflect the operations applied, just as if `transfer` had been used. This, along with the `report` method of labware, can be used to validate pipetting was conducted as expected.

If you want to know the volume or composition of a labware after pending operations but before you have committed operations, use the `latest_volume` and `latest_composition` methods of `AdvancedLabware`.

Once optimisation is complete, it will print how many aspirates, dispenses, and washes were calculated. If you do not want to see a progress bar or this message printed, call `wl.commit()` with argument `report=False`.

#### Calculation Time
The optimiser can take some time to calculate a pipetting order. By default, it will show a progress bar and a timer (hidden if `report=False`). The time taken will depend both on the number of operations, and the number of labwares/columns touched. Typical running times are ~12 seconds for 180 operations, extending up to ~10 minutes for 768 operations. If faster running time is desired, operations can be 'batched' by creating the `AutoWorklist` with the `max_concurrent_ops` argument. Setting this to 200 will split the operations into batches of 200 - all ops within each batch will be optimised, but the batches will dispense in order. This makes the optimiser run much faster, but sacrifices some optimality, as operations in seperate batches cannot be grouped. 

#### Further Examples
Complete examples are available in the examples folder. Start with [example_basic_row_column.py](examples/example_basic_row_column.py)


## Architecture
This package is designed first to track the operations which a user wishes to conduct on all defined Labware, then rearrange the order of these operations
to achieve maximum pipetting efficiency, before adding them to a worklist with Advanced Worklist Commands (Aspirate and Dispense). This allows us to 
precisely define which tips should be used in which wells, and prevent EVOware's built in optimiser from changing our optimised order.

Some additional properties must be defined to allow us to automatically generate advanced worklist commands. E.g. each labware in use must have a location tuple
(grid, site) defined. Thus `AdvancedLabware` and equivalent `AdvancedTrough` classes have been created which require the inclusion of this location. They also track operation history for the labware, which is required for some optimiser features.

The `AutoWorklist` class is defined as an equivalent to `EvoWorklist`, but with additional functionality to enable this tracking and optimisation. Its use is identical to
`EvoWorklist`, except that a waste_location and cleaner_location must be specified, which allow us to generate advanced worklist Wash commands (as the basic worklist W command does not work in conjunction with advanced worklist Aspirate and Dispense). All functionality present in `EvoWorklist` is present and unchanged; you can use `.transfer` just as usual, and the corresponding operations will not be optimised.

A `TransferOperation` class is defined, which captures all the relevant details for a transfer from one well to another, including the plate and position of the source
and destination wells, the volume, and the liquid class. Each `TransferOperation` has a unique ID. The `auto_transfer` method, which is equivalent to `transfer` but adds optimisation, produces one `TransferOperation` for each well it is passed, recording the details appropriate to that well. It then adds all the ops to a set `pending_ops` which records the operations defined by the user but not yet optimised and recorded to the worklist.

An `AdvancedLabware` (and equivalent `AdvancedTrough`) class is defined, which, in addition to requiring the `location` property, also track which `TransferOperation`s touch each of its wells.
When a `TransferOperation` is created, it records the latest `TransferOperation` referenced at the relevant source and destination wells as `source_dep` and `dest_dep` (for source and destination dependency). An operation will only be considered for optimisation if both of these ancestor operations have already been completed. This ensures that all operations on a particular well occur in the order they are defined in code, the same order that they would occur in if `transfer` were used instead. For a practical example, if an `auto_transfer` dispensing diluent to a well occurs before one which dispenses sample to that well, diluent will always be dispensed before sample.

Details of the pending operations can be be accessed with `wl.report`. Operations will be optimised and committed to the worklist whenever `commit()` is called, or when the with block exits.

Volume and composition values of labware are not updated until operations are actually written to the worklist, when `commit()` is called. This ensures that the volume tracking always accurately reflects the actual operations applied, and allows the use of volume tracking to 'sanity-check' the optimiser - volumes are only updated by `robotools` `evo_aspirate` and `evo_dispense` commands called internally, and never modified by the optimiser directly. However, sometimes a user may wish to get the volume or composition of a well after some operations have been queued, but before a commit. To do this, `AdvancedLabware` and `AdvancedTrough` have a `latest_volume` and `latest_composition` method, which will get the volume or composition as it will be after any pending ops have been applied.

The particulars of how operations are optimised is rather complex, however it is the intention of this package that, as long as a user understands and trusts the `TransferOperation` creation and dependency constraint, they do not need to understand the means by which operations are grouped - they can be confident that all operations will be executed in the order they defined.


### Optimisation Details
The optimiser uses a greedy algorithm; at each step it selects the available operations which can complete the most operations in the least movements of the LiHa. This does not guarantee global optimality - it may sometimes be more efficient to select fewer operations for one step to allow a more efficient grouping for a later step. However, considering such possibilities adds significant design and computational complexity, and in preliminary testing, greedy optimisation is very good; little further benefit would be achieved from ensuring global optimality.

Further details of how the algorithm works are available in [`optimisation_details.md`](optimisation_details.md).