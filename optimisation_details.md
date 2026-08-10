# evo_pipetting_optimiser
## Optimisation Algorithm Details

The optimiser uses a graph representation of labware, columns, rows, and offsets, then calculates a flow through this graph to find a pipetting solution.

For the operations

![Report showing 4 pending operations](figures/demo_1.png)

The following graph is produced

![Figure showing a graph with nodes representing the pipetting rows and columns](figures/optimiser%20graph%201.png)

We use this graph to calculate pipetting operations that can be conducted in a single move of the pipetting arm, and find the operation with maximal efficiency.

The left 4 columns are the 'source' side. The graph begins with an 's' node, which is needed to calculate a flow. This connects to a node for each column in each labware. As the pipetting arm has 8 tips in a column, it can only access 1 column in 1 labware at a time. So this is the first layer of division.

Each of these columns has (usually 8) rows. The rows that have pending pipetting operations have nodes created. The associated labware + column node has an edge going to the rows contained within that column. Here, smp column 0 has pipetting operations in row 0 and row 2 (A01 and C01).

Each column in a plate may be accessed with the tips offset in the y direction. With the default offset of 0, each tip is above its corresponding row. With an offset of +7, tip 8 is above row A. With an offset of -7, tip 1 is above row H. To show a simpler graph for this example, offset_limit_up and offset_limit_down are each set to 2 in the labware objects, which limits the possible offsets from -2 to +2.

Edges are created from each row, to each offset which can address this row. Each of these edges thus represents a tip being assigned 