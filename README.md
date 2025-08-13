# Evo_Pipetting_Optimiser

## Introduction
A frequent challenge in using the Tecan EVO is the flawed optimisation that EVOware conducts for basic worklist commands (A and D),
as produced by `robotools.EvoWorklist.transfer()`. The strategy it selects can often make pipetting slower, and B; commands 
`EvoWorklist.commit()` must be inserted between commands to stop the EVO from rearranging them inefficiently.

Additionally, a common use case for robotools is dilution/processing/preparation of several samples in a plate. In such cases,
the easiest way to translate the process to Python and robotools is to loop through each sample, and define the pipetting steps which 
must be conducted for that sample to be processed. Unfortunately, this leads to inefficient worklist commands, which pipette one sample at
a time. Previously, careful design and more complicated code was employed to pre-calculate the operations required for all samples,
then make fewer calls to `transfer()`, ideally only one (covering all samples) for each step in the process. This allowed several samples to
be pipetted at once, at the cost of making code more difficult to write and understand. Additionally, it could still fall victim to poor
optimisation from the EVOware side, and miss potential speed.

Thus an optimiser which tracks all operations a user wishes to conduct, before automatically grouping them into the most efficient order,
is desired

### In scope
Optimisation of pipetting operations for an 8-tip LiHa on the Tecan EVO

Provision of an interface with minimal differences to existing robotools `EvoWorklist` and `transfer` funtionality so that operation is
understandable to existing users

Maintainence of pipetting order; no operations on a particular well shall be conducted in a different order to that which they were specified in code

Support for the existing composition tracking and reporting functionality in robotools

### Out of scope
Other liquid handlers or arms.

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

The file  [`example_basic_row_column.py`](examples/example_basic_row_column.py) shows how we can import and use functions from the package.


## Architecture
This package is designed first to track the operations which a user wishes to conduct on all defined Labware, then rearrange the order of these operations
to achieve maximum pipetting efficiency, before adding them to a worklist with Advanced Worklist Commands (Aspirate and Dispense). This allows us to 
precisely define which tips should be used in which wells, and prevent EVOware's built in optimiser from changing our optimised order.

Some additional properties must be defined to allow us to automatically generate advanced worklist commands. E.g. each labware in use must have a location tuple
(grid, site) defined.

The `AutoWorklist` class is defined as an equivalent to `EvoWorklist`, but with additional functionality to enable this tracking and optimisation. Its use is identical to
`EvoWorklist`, except that a waste_location and cleaner_location must be specified, which allow us to generate advanced worklist Wash commands (as the basic worklist W command does not work in conjunction with advanced worklist Aspirate and Dispense) All functionality present in `EvoWorklist` is present and unchanged; you can use `.transfer` just as usual, and the corresponding operations will not be optimised.

A `TransferOperation` class is defined, which captures all the relevant details for a transfer from one well to another, including the plate and position of the source
and destination wells, the volume, and the liquid class. Each `TransferOperation` has a unique ID. The `auto_transfer` method, which is equivalent to `transfer` but adds optimisation, produces one `TransferOperation` for each well it is passed, recording the details appropriate to that well. It then adds all the ops to a set `pending_ops` which records the operations defined by the user but not yet optimised and recorded to the worklist.

An `AdvancedLabware` class is defined, which, in addition to requiring the `location` property, also tracks the last `TransferOperation` to touch each of its wells.
When a `TransferOperation` is created, it records the latest `TransferOperation` referenced at the relevant source and destination wells as `source_dep` and `dest_dep`. An operation will only be considered for optimisation if both of these dependent operations have already been completed. This ensures that all operations on a particular well occur in the order they are defined in code, the same order that they would occur in if `transfer` were used instead. This ensures that, if an `auto_transfer` dispensing diluent to a well occurs before one which dispenses sample, diluent will always be dispensed before sample.

It is assumed that Troughs will only be aspirated from, never dispensed to, so their composition will remain unchanged. Therefore, operation tracking for Troughs is not defined. A `Trough` can be used as the source for an `auto_transfer` (as long as it has the location property set), but not the Destination.

Details of the pending operations can be printed by calling `report_ops`. Operations will be optimised and committed to the worklist whenever `commit()` is called, or when the worklist is saved.

The particulars of how operations are optimised is rather complex, however it is the intention of this package that, as long as a user understands and trusts the `TransferOperation` creation and dependency constraint, they do not need to understand the means by which operations are grouped - they can be confident that all operations will be executed in the order they defined.

