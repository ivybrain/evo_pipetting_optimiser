# evo_pipetting_optimiser
## Optimisation Algorithm Details

The optimiser uses a graph representation of labware, columns, rows, and offsets, then calculates a flow through this graph to find a pipetting solution.

For the operations

![Report showing 4 pending operations](figures/demo_1.png)

The following graph is produced

![Figure showing a graph with nodes representing the pipetting rows and columns](figures/optimiser%20graph%201.png)

We use this graph to calculate pipetting operations that can be conducted in a single move of the pipetting arm, and find the operation with maximal efficiency.

The left 4 columns are the 'source' side. The graph begins with a single source node labelled 's', which is needed to calculate a flow. This connects to a node for each column in each labware. As the pipetting arm has 8 tips in a column, it can only access 1 column in 1 labware at a time. So this is the first layer of division.

Each of these columns has (usually 8) rows. The rows that have pending pipetting operations have nodes created. The associated labware + column node has an edge going to the rows contained within that column. Here, smp column 0 has pipetting operations in row 0 and row 2 (A01 and C01). The other columns have only 1 row relevant.

Each column in a plate may be accessed with the tips offset in the y direction. With the default offset of 0, each tip is above its corresponding row. With an offset of +7, tip 8 is above row A. With an offset of -7, tip 1 is above row H. To show a simpler graph for this example, offset_limit_up and offset_limit_down are each set to 2 in the labware objects, which limits the possible offsets from -2 to +2.

Edges are created from each row, to each offset which can address this row. Each of these edges thus represents a tip being assigned to a particular row. The graph shows all possible assignments as grey edges, and the edges that were actually selected by the algorithm in red. 'smp c0r0' and 'smp c0r2' can both be aspirated with an offset of 0, so they have red edges to the 'smp c0 ofst 0' node. As we are aspirating row 0 and row 2 with an offset of 0, this will use tip 0 and tip 2. This node represents a single 'aspirate'.

The 5th column also represents an offset, but on the 'dispense' side. From 'aspirate' c0 ofst 0, 1 edge goes to 'dilplate c1 ofst 0' (the transfer from A01 to A02). However the other edge goes to a different offset, because we are transferring to a different row - C01 to B02. As tip 2 aspirated C01, and it now needs to dispense to B02, it must do this with an offset of -1, so the edge goes to 'dilplate c1 ofst -1'. Thus, this single aspirate node links to 2 dispense nodes, and 2 dispenses are required.

These dispense nodes then have edges back to the respective rows of the dispense plate. Here, edges to rows other than the actual target may also appear highlighted - this is not meaningful, since the algorithm only needs to ensure sufficient capacity from the correct offset, not a specific edge selection. No decision is made based on the selected edges in this column.

These rows are then linked to nodes representing the column of the dilution plate, which is in turn linked to the the 't' node, called the sink. The source and sink are used only for the flow algorithm, as a shared 'origin' and 'destination' for all nodes even if they are on different plates.

### Flow calculation
Above, we explain how nodes and edges are constructed, and what they represent. To come up with a 'flow' through this graph (a selection of edges), we assign different edges weights, corresponding to how preferable they are, and then use an algorithm called `max_flow_min_cost` to calculate a flow.

![Figure showing a graph with nodes representing the pipetting rows and columns plus weights shown](figures/optimiser%20graph%20weights.png)
In this figure, each edge is labelled with 3 numbers, separated by slashes. The first is the 'flow' - what the algorithm has decided to put through this edge. You will see that the highlighted edges have a flow >= 1. 1 unit of flow represents 1 op - an edge with a flow of 1 will complete 1 op if it is selected. The second number is the capacity - the maximum flow an edge could take, corresponding to the maximum number of ops this edge can address. You will see that the edge s -> smp c0 has a flow of 2 and a capacity of 2, because 2 ops are completable from this column. 

The third number is the weight. The edges between the 3 outer columns have no weights, as they do not influence the moves of the actual arm - they merely help with construction of the graph. The central edges between the 'source offset' and 'destination offset' columns have a weight in the range of 0-15. This is an 'offset penalty' applied to keep offsets low and close together. 


The first penalty calculates the difference between the offsets required to complete the pipetting - this varies based on rows used. Pipetting from row A to row A will be doable with an offset of 0, but pipetting from row A to row D requires an offset of at least 3. However, moving the arm further with an offset >3 is counterproductive - the op was already doable with offest 3, and a higher offset means the arm can address less additional wells in the same movement, meaning it could restrict the selection of other ops. Thus we incentivise against offsets greater than necessary by applying a penalty equal to actual offset minus the smallest required offset. `u_offset` and `v_offset` refer to the source offset and destination offset respectively. 

```
offset_difference = abs(u_offset - v_offset)
largest_offset = max(abs(u_offset), abs(v_offset))

if largest_offset > offset_difference:
    offset_penalty += largest_offset - offset_difference
```

The second penalty encourages the sum of the offsets to be as close to 0 as possible - this ensures that we stay as close to offset 0 as possible when transferring between similar rows, again maximising the other ops that can share in a movement.

```
# We want to encourage offsets closer to 0 
offset_penalty += abs(u_offset + v_offset)
```

The weights between row nodes and offset nodes are 0 or negative. The calculation here is 
`-16 * number_of_row_edges`, where number_of_row_edges is the number of edges from that offset to corresponding rows - i.e. the number of different operations that can be addressed with a single offset. We want to maximise this, as offsets that connect to the maximum number of rows can complete that number of ops with a single arm movement.

The `max_flow_min_cost` algorithm will first maximise the flow - selecting every possible op. It will then minimise the cost - complete these ops with the minimum sum of (flow * weight) across all edges.

A key advantage of this approach is the ability to realise shared nodes at both the source and the destination, and consider these equally, including at the same time. Consider another example, with ops:
![Figure showing 8 ops from 3 columns to 2 columns](figures/demo_multiplex.png)

The operations here go from 3 columns on one plate to 2 columns from another plate. However the rows are misaligned such source column 2 goes to 2 different destination columns. A naive approach might aspirate ops 0-3 directly, then do 1 aspirate and 2 dispenses to cover ops 3-5, then another direct transfer for ops 6-7, taking 3 aspirates and 4 dispenses. The optimiser can do this in 3 aspirates and 2 dispenses. Let's look at the graph generated.

![Figure showing the graph represengint the ops above](figures/graph_multiplex_example.png)

As before, red edges indicate edges with flow, grey edges are possible offsets that were not selected by the flow algorithm. Edge labels are removed for legibility. Offsets are limited to reduce the number of visible nodes.

Here, it is obvious that the dispense node `c0 ofst 0`, representing column 0 offset 0 for the dilution plate, has the most flow going through it. So, the algorithm chooses this node. This node can dispense to 4 rows with a single offset. However, it has edges to 2 aspirate nodes, so doing all 4 of these dispense rows would require 2 aspirates.

We create a list of edges going into this node, and sort them by the number of ops assigned to that edge. We look at the first edge in the list - this is the edge `c0 ofst 0 -> c0 ofst 0`. This contains ops 0-2. We add the operations 0-2 to our move.

We now have a valid move with 3 operations. However we still have empty tips, so we want to look for other operations we can combine to use the other tips. We select the next edge from the list, `c1 ofst -3 -> c0 ofst 0`. This edge contains op 3. We add this to the move, so now we have 2 aspirates and 1 dispense covering ops 0-3.

However unlike the previous node, `c1 ofst -3` has other out edges - meaning that more than just op 3 can be aspirated with this offset. The edge `c1 ofst -3 -> c1 ofst -4` contains ops 4 and 5. We add this edge to our list. This edge is now the only item in the list, so we process it next, and add ops 4 and 5, with a dispense at `c1 ofst -4`. This node also has another in-edge - it is not highlighted, as it was not assigned flow, but flow only influences which node we choose first - when building a move we consider all possible edges. The edge `c2 ofst -6 -> c1 ofst -4` contains ops 7 and 8. As it is pipetting rows A and B to rows C and D, the difference in the offsets must be +2 to line the rows up correctly. We add this final aspirate to the move, and now we have a finished move covering all 8 ops with 3 aspirates and 2 dispenses.

The algorithm also loops through all other nodes with flow, and calculates a move with that as a starting point, giving a list of available moves. It then chooses the move that accomplishes the most ops, given

The fact that we consider edges even without flow raises the question of if we even need the flow calculation at all. 

Without the weights, the selection of edges through the graph would be completely random. This would always produce a *possible* set of pipetting operations, correctly assigning rows and offsets, but it will not be efficient. The described weights influence the algorithm to choose edges that are the most likely to enable other ops to share the touched nodes.

This might seem contrived - the weights are a guide to selecting popular nodes. The obvious way would be to count the number of nodes touched, and minimise this - this would by definition select the nodes that can fulfil the most edges (ops) at a time. However, counting selected nodes, or edges, becomes a boolean programming problem - a node should be counted equally if it has 1 flow or 100 flow, and not counted if it has 0 flow - it is boolean, off or on. This is a very difficult type of problem to solve computationally - known as NP-hard. However, the weights allow us to use the 'simplex' algorithm, which is under the hood of `max_flow_min_cost`. This is a linear programming technique, where the weight can be represented as a linear equation - the sum of the weights multiplied by the flow is in fact a linear equation. Boolean inclusion/exclusion cannot be represented in a linear equation. Linear equations are much easier computationally, achieving polynomial time on average. Thus, this allows the optimiser to repeatedly calculate good-enough assignments at each step of the process in a reasonable time.