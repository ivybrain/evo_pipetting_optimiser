from matplotlib import pyplot as plt
import networkx as nx
from .Move import Move
from functools import reduce
import heapq

class OptimisationError(Exception):
    pass

def source_key(op):
    return ("s", op.source.name, op.source_pos[1], op.liquid_class)


def dest_key(op):
    return ("t", op.destination.name, op.dest_pos[1], op.liquid_class)


class OpGraph:
    def __init__(self, ops, tip_ops_since_wash={i: [] for i in range(8)}, node=None):
        """
        Init a graph from a set of ops
        Graph used to decide which col in which source and dest to pipette
        Graph has 6 'columns'
        source node
        source labware rows
        all possible offsets for source labware
        all possible offsets for dest labware
        dest labware rows
        sink node
        """
        super().__init__()

        self.node = node
        self.tip_ops_since_wash = tip_ops_since_wash
        G = nx.DiGraph()
        self.G = G

        scaling = 1
        # demand = len(ops)
        G.add_node("s")
        G.add_node("t")

        self.asp_nodes = {}
        self.disp_nodes = {}
        source_nodes = {}
        dest_nodes = {}
        self.op_groups = {}
        self.op_to_group = {}

        for op in ops:

            source_nodes[source_key(op)] = True

            if ("s", source_key(op)) in G.edges:
                G.edges[("s", source_key(op))]["capacity"] += scaling
            else:
                G.add_edge("s", source_key(op), capacity=scaling, weight=0)

            dest_nodes[dest_key(op)] = True
            if (dest_key(op), "t") in G.edges:
                G.edges[(dest_key(op), "t")]["capacity"] += scaling
            else:
                G.add_edge(dest_key(op), "t", capacity=scaling, weight=0)

            if source_key(op) not in self.asp_nodes:
                self.asp_nodes[source_key(op)] = 0
            if dest_key(op) not in self.disp_nodes:
                self.disp_nodes[dest_key(op)] = 0

            if (source_key(op), dest_key(op)) not in G.edges:
                self.asp_nodes[source_key(op)] += 1
                self.disp_nodes[dest_key(op)] += 1

            # Add groups representing edges from source to dest
            # but made up of a subgraph to calculate tip use and cost
            if (source_key(op), dest_key(op)) in self.op_groups:
                self.op_groups[(source_key(op), dest_key(op))].append(op)
            else:

                self.op_groups[(source_key(op), dest_key(op))] = [op]

            self.op_to_group[op] = (source_key(op), dest_key(op))

        return

    def get_moves(self):

        # Create a copy of the graph to populate with the subgraphs
        G = self.G.copy()

        # Create subgraph representing offsets
        offset_edges = []
        source_offset_in_edges = {}
        dest_offset_out_edges = {}

        for (source_col, dest_col), past_ops in self.op_groups.items():
            scaling = 1
            asp_nodes = {}
            disp_nodes = {}
            group_edges = []
            source_row_nodes = {}
            dest_row_nodes = {}
            for op in past_ops:

                # Calculate the feasible tips for this op given the previous wash state of the tips
                free_tips = set()
                tips_with_wash = set()
                for tip, past_ops in self.tip_ops_since_wash.items():

                    # Check if there's a specific allowed tip only
                    if op.tip is not None and op.tip != tip:
                        continue

                    # Check we have not reached the max use limit for this tip
                    if len(past_ops) >= op.worklist.wash_params["max_tip_uses"]:
                        tips_with_wash.add(tip)
                        continue


                    # If tip is unused, we can use it
                    if not past_ops:
                        free_tips.add(tip)
                        continue

                    # If tip has only ops with wash=None, we can use
                    if all(past_op.wash_scheme == None for past_op in past_ops):
                        free_tips.add(tip)
                        continue

                    # If the current op and prior ops on the tip have smartwash enabled, we can check if composition is the same
                    if op.wash_scheme == "smart" and all(
                        past_op.wash_scheme == "smart" for past_op in past_ops
                    ):

                        past_touched = reduce(
                            lambda x, y: x | y.dest_composition | y.source_composition,
                            past_ops,
                            {},
                        ).items()

                        source_comp = op.source_composition
                        dest_comp = op.dest_composition
                        wash_needed = False

                        for prev_key, prev_conc in past_touched:
                            # If the previous ops touched a liquid that is not present in our source, we need to wash to not contaminate it
                            if prev_key not in source_comp:
                                wash_needed = True
                                break

                            # If last op touched a well signficantly more concentrated than this one, wash to not contaminate with the higher conc
                            if prev_conc > 1.2 * source_comp[prev_key]:
                                wash_needed = True
                                break

                        if not wash_needed:
                            free_tips.add(tip)
                            continue

                    tips_with_wash.add(tip)

                source_row_key = (*source_col, op.source_pos[0])
                source_row_nodes[source_row_key] = True

                if (source_col, source_row_key) in G.edges:
                    G.edges[(source_col, source_row_key)]["capacity"] += scaling
                else:
                    G.add_edge(source_col, source_row_key, capacity=scaling, weight=0)

                dest_row_key = (*dest_col, op.dest_pos[0])
                dest_row_nodes[dest_row_key] = True
                if (dest_row_key, dest_col) in G.edges:
                    G.edges[(dest_row_key, dest_col)]["capacity"] += scaling
                else:
                    G.add_edge(dest_row_key, dest_col, capacity=scaling, weight=0)

                def add_offset_edge(
                    source_offset,
                    dest_offset,
                    source_capacity,
                    dest_capacity,
                    tip=0,
                    wash_needed=False,
                ):
                    source_offset_key = ("s_ofst", *source_col, source_offset)
                    dest_offset_key = ("d_ofst", *dest_col, dest_offset)



                    if source_offset_key not in asp_nodes:
                        asp_nodes[source_offset_key] = 0
                    if dest_offset_key not in disp_nodes:
                        disp_nodes[dest_offset_key] = 0

                    if (source_row_key, source_offset_key) not in G.edges:
                        asp_nodes[source_offset_key] += 1
                    if (dest_row_key, dest_offset_key) not in G.edges:
                        disp_nodes[dest_offset_key] += 1

                    if source_offset_key not in source_offset_in_edges:
                        source_offset_in_edges[source_offset_key] = [source_row_key]
                    else:
                        source_offset_in_edges[source_offset_key].append(source_row_key)

                    if dest_offset_key not in dest_offset_out_edges:
                        dest_offset_out_edges[dest_offset_key] = [dest_row_key]
                    else:
                        dest_offset_out_edges[dest_offset_key].append(dest_row_key)

                    G.add_edge(
                        source_row_key,
                        source_offset_key,
                        capacity=source_capacity,
                        weight=0,
                    )

                    # Set weight based on how many incident edges the nodes at each end have
                    if (source_offset_key, dest_offset_key) in G.edges:
                        G.edges[(source_offset_key, dest_offset_key)][
                            "capacity"
                        ] += scaling
                        G.edges[(source_offset_key, dest_offset_key)]["op_tip"][
                            op
                        ] = tip
                        G.edges[(source_offset_key, dest_offset_key)]["wash_needed"] = (
                            G.edges[(source_offset_key, dest_offset_key)]["wash_needed"]
                            or wash_needed
                        )

                    else:
                        group_edges.append((source_offset_key, dest_offset_key))
                        G.add_edge(
                            source_offset_key,
                            dest_offset_key,
                            capacity=scaling,
                            weight=0,
                            op_tip={op: tip},
                            wash_needed=wash_needed,
                        )

                    G.add_edge(
                        dest_offset_key,
                        dest_row_key,
                        capacity=dest_capacity,
                        weight=0,
                    )

                # Check if we have virtual rows
                source_virtual_rows = getattr(op.source, "virtual_rows", 0) or 0
                dest_virtual_rows = getattr(op.destination, "virtual_rows", 0) or 0
                if source_virtual_rows > op.source_pos[0]:
                    source_offset_range = [0]
                    if dest_virtual_rows > op.dest_pos[0]:
                        dest_offset_range = [0]
                    else:
                        # tip = row - offset
                        # offset = row - tip
                        dest_offset_range = [
                            op.dest_pos[0] - tip
                            for tip in (free_tips.union(tips_with_wash))
                        ]

                else:
                    source_offset_range = [
                        op.source_pos[0] - tip
                        for tip in (free_tips.union(tips_with_wash))
                    ]
                    if dest_virtual_rows > op.dest_pos[0]:
                        dest_offset_range = [0]
                    else:
                        dest_offset_range = [None]

                for source_offset in source_offset_range:
                    for dest_offset in dest_offset_range:
                        if dest_offset is None:
                            dest_offset = source_offset + (
                                op.dest_pos[0] - op.source_pos[0]
                            )

                        # Check if this offset is feasible for the dest row
                        source_limit_up = -op.source.offset_limit_up
                        source_limit_down = op.source.offset_limit_down
                        dest_limit_up = -op.destination.offset_limit_up
                        dest_limit_down = op.destination.offset_limit_down

                        if not (source_limit_up <= source_offset <= source_limit_down):
                            continue

                        if not (dest_limit_up <= dest_offset <= dest_limit_down):
                            continue

                        source_tip = op.source_pos[0] - source_offset
                        dest_tip = op.dest_pos[0] - dest_offset

                        tip = max(source_tip, dest_tip)
                        wash_needed = tip in tips_with_wash

                        if tip not in tips_with_wash and tip not in free_tips:
                            raise OptimisationError(f"Tip {tip} is neither in tips_with_wash nor free_tips")

                        if not (0 <= source_tip <= 7):
                            raise OptimisationError("Tip out of range")
                        if not (0 <= dest_tip <= 7):
                            raise OptimisationError("Tip out of range")
                        if not (source_tip == 0 or dest_tip == 0 or source_tip == dest_tip):
                            raise OptimisationError("Tip mismatch")

                        add_offset_edge(
                            source_offset,
                            dest_offset,
                            max(1, source_virtual_rows),
                            max(1, dest_virtual_rows),
                            tip=max(source_tip, dest_tip),
                            wash_needed=wash_needed,
                        )

                    # Add edges from rows to possible offsets

            offset_edges += group_edges

        # Reweight offset edges
        for u, v in offset_edges:
            # Penalty for unneeded offsets
            offset_penalty = 0
            # If offsets the same, should be 0 as we're then doing straight transfer
            # Abs offset should be below the difference in offsets
            u_offset = u[-1]
            v_offset = v[-1]

            offset_difference = abs(u_offset - v_offset)
            largest_offset = max(abs(u_offset), abs(v_offset))

            if largest_offset > offset_difference:
                offset_penalty += largest_offset - offset_difference

            # We want to encourage offsets closer to 0 
            offset_penalty += abs(u_offset + v_offset)

            in_edges = source_offset_in_edges[u]
            for edge in in_edges:
                G.edges[edge, u]["weight"] = -len(in_edges) * 16

            new_edges = dest_offset_out_edges[v]
            for edge in new_edges:
                G.edges[v, edge]["weight"] = -len(new_edges) * 16

            G.edges[u, v]["weight"] = (
                # small penalty for offset further from 0
                offset_penalty
            )

        moves = []

        flow_dict = nx.algorithms.max_flow_min_cost(G, "s", "t")
        flow_value = sum(flow_dict["s"].values())

        for u, v in G.edges():
            # try:
            G[u][v]["flow"] = flow_dict[u][v]

        

        selectable_nodes = [
            (sum(flow_dict[n].values()), n)
            for n in G.nodes
            if (n[0] == "s_ofst" or n[0] == "d_ofst")
            and sum(flow_dict[n].values()) > 0
        ]
        selectable_nodes.sort(key=lambda x: -x[0])

        # nodes_in_moves = []

        # Loop through all selectable nodes and create a 'move' from each node as a starting point
        for node_score, node_choice in selectable_nodes:

            # If we've already included this node in a previous move, skip it
            # if node_choice in nodes_in_moves:
            #     continue

            nodes_this_move = [node_choice]

            # Initialise a possible move associated with this node
            move = Move()
            moves.append(move)

            # Dict of (source, col, lc): [ops]
            aspirates = move.aspirates
            # Dict of (dest, col): [ops]
            dispenses = move.dispenses
            asp_ops = move.ops
            tips_used = move.tips
            potential_ops = []

            if node_choice[0] == "s_ofst":
                # Pre-select edges matching the source offset node for efficiency
                selectable_edges = list(G.out_edges(node_choice))
            else:
                selectable_edges = G.in_edges(node_choice)

            # How we sort and prioritise the edges
            def edge_sort_key(edge):
                return (G.edges[*edge]["wash_needed"],
                -len(G.edges[*edge]["op_tip"]),
                min(G.edges[*edge]["op_tip"].keys()))

            edge_heap = []

            # Add the selectable edges to a heap
            for edge in selectable_edges:
                heapq.heappush(edge_heap, (edge_sort_key(edge), edge))

                                           
            new_ops = None

            while edge_heap:

                _, (u,v) = heapq.heappop(edge_heap)

                (_, _, source, col, _, offset) = u

                if G.edges[u, v]["wash_needed"]:
                    move.wash_first = True
                    move.tip_ops_since_wash = self.tip_ops_since_wash

                potential_ops, potential_tips = zip(*G[u][v]["op_tip"].items())
                if any([op in asp_ops for op in potential_ops]):
                    continue

                if len(potential_ops) == 0:
                    continue
                if any(tip in tips_used for tip in potential_tips):
                    continue

                tips_used += potential_tips

                asp_ops += potential_ops
                if (source, col, offset) in aspirates:
                    aspirates[source, col, offset] += potential_ops
                else:
                    aspirates[source, col, offset] = list(potential_ops)

                (_, _, dest, col, _, offset) = v

                if (dest, col, offset) in dispenses:
                    dispenses[dest, col, offset] += potential_ops
                else:
                    dispenses[dest, col, offset] = list(potential_ops)

                new_ops = potential_ops

                # Add whichever node is on the other end of this edge to the list of nodes touched by this move
                # And add any of its other edges to the list of edges selectable by this move
                
                if node_choice == u:
                    # If we had chosen a source node, then a dest node is being added
                    new_node = v
                    # Add the in-edges for this dest node (except current edge)
                    new_edges = list(G.in_edges(new_node))
                    new_edges.remove((u, v))
                else:
                    # Otherwise, a source node is being added
                    new_node = u
                    # Add the out-edges for this source node (except current edge)
                    new_edges = list(G.out_edges(new_node))
                    new_edges.remove((u,v))

                # Add the new edges to heap to be processed next by smallest key
                for edge in new_edges:
                    heapq.heappush(edge_heap, (edge_sort_key(edge), edge))

                nodes_this_move.append(new_node)
                #nodes_in_moves.append(new_node)

                # If we have the potential to add new ops, (we have less than 8 selected, and more than the number selected
                # were included in a flow)
                # then continue looping to add more ops from selectable edges. Otherwise, end the loop
                if not (new_ops and len(move.tips) < 8 and flow_value > len(potential_ops)):
                    break

            move.calc_score()


        moves.sort(reverse=True)
        return moves


def show_graph(G):
    """
    Show the graph in an interactive window. Useful for debugging and checking that edges are correctly set to possible moves.
    If the graph has flow values set, edges with flow will be highlighted red and have flow/capacity/cost values shown

    To use this, I set a breakpoint in the Opgraph method and then call it from the debugging console with 
    show_graph(G)
    Note: The plt.show method does not work well inside a threaded approach. If you have issues, replace the
    `executor.map` in AutoWorklist.py with a plain map'
    """
    G = G.copy()

    reachable_dict = nx.shortest_path_length(G, source="s")
    reachable_nodes = set(reachable_dict.keys())

    all_nodes = set(G.nodes)
    unreachable_nodes = all_nodes - reachable_nodes

    # Step 3: Remove unreachable nodes
    # We must convert the iterator to a list to avoid RuntimeError during iteration
    G.remove_nodes_from(list(unreachable_nodes))

    pos = nx.bfs_layout(G, start="s")

    # Identify edges with positive flow
    active_edges = [
        (u, v) for u, v in G.edges() if "flow" in G[u][v] and G[u][v]["flow"] > 0
    ]

    # Draw nodes and all edges faintly
    nx.draw_networkx_nodes(G, pos, alpha=0.75)
    nx.draw_networkx_edges(G, pos, alpha=0.2)  # faint background edges

    # Draw active flow edges prominently
    nx.draw_networkx_edges(G, pos, edgelist=active_edges, edge_color="r", width=2)

    # Extract the flow values for labeling
    flow_labels = nx.get_edge_attributes(G, "flow")
    capacity_labels = nx.get_edge_attributes(G, "capacity")
    weight_labels = nx.get_edge_attributes(G, "weight")

    edge_labels = {
        key: f"{flow_labels[key]}/{capacity_labels[key]}/{weight_labels[key]}"
        for key in flow_labels.keys()
        # if key in active_edges
    }

    node_labels = {}
    for node in G.nodes:
        if len(node) == 1:
            node_labels[node] = str(node)
        if len(node) == 4:
            # Cols node is 
            # s, source_name, source_col, liquid_class OR
            # t, dest_name, dest_col, liquid_class
            node_labels[node] = f"{node[1]} c{node[2]}"

        if len(node) == 5:
            # Rows node is
            # s, source_name, source_col, liquid_class, source_row OR
            # t, dest_name, dest_col, liquid_class, dest_row
            node_labels[node] = f"c{node[2]}r{node[4]}"

        if len(node) == 6:
            # Offset node is 
            # s_ofst, s, source_name, source_col, liquid_class, offset OR
            # d_ofst, t, dest_name, dest_col, liquid_class, offset
            node_labels[node] = f"c{node[3]} ofst {node[5]}"

    # Draw the labels on the edges
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    nx.draw_networkx_labels(G, pos, labels=node_labels)

    plt.show()
