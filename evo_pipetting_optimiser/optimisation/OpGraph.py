from matplotlib import pyplot as plt
import networkx as nx
from .Move import Move
from functools import reduce


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

    def get_moves(
        self, occupied_tips=set(), selected_node=None, move=None, selected_ops=set()
    ):

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
                if op in selected_ops:
                    continue

                # Calculate the feasible tips for this op given the previous wash state of the tips
                free_tips = set()
                tips_with_wash = set()
                for tip, past_ops in self.tip_ops_since_wash.items():

                    # Check we have not reached the max use limit for this tip
                    if len(past_ops) >= op.worklist.wash_params["max_tip_uses"]:
                        tips_with_wash.add(tip)
                        continue

                    # If tip is occupied, skip it
                    if tip in occupied_tips:
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

                    # If we're finding new ops to add to an existing move, only process the selected node
                    if selected_node and (
                        selected_node != source_offset_key
                        and selected_node != dest_offset_key
                    ):
                        return

                    if source_offset_key not in asp_nodes:
                        asp_nodes[source_offset_key] = 0
                    if dest_offset_key not in disp_nodes:
                        disp_nodes[dest_offset_key] = 0

                    if (source_row_key, source_offset_key) not in G.edges:
                        asp_nodes[source_offset_key] += 1
                    if (dest_row_key, dest_offset_key) not in G.edges:
                        disp_nodes[dest_offset_key] += 1

                    if source_offset_key not in source_offset_in_edges:
                        source_offset_in_edges[source_offset_key] = 1
                    else:
                        source_offset_in_edges[source_offset_key] += 1

                    if dest_offset_key not in dest_offset_out_edges:
                        dest_offset_out_edges[dest_offset_key] = 1
                    else:
                        dest_offset_out_edges[dest_offset_key] += 1

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

                        assert tip in tips_with_wash or tip in free_tips

                        assert 0 <= source_tip <= 7, "Tip out of range"
                        assert 0 <= dest_tip <= 7, "Tip out of range"
                        assert (
                            source_tip == 0 or dest_tip == 0 or source_tip == dest_tip
                        ), "Tip mismatch"

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
            # Add 7 to offsets to they're moved into the range 0-15
            u_offset = u[-1]
            v_offset = v[-1]

            offset_difference = abs(u_offset - v_offset)
            largest_offset = max(abs(u_offset), abs(v_offset))

            if largest_offset > offset_difference:
                offset_penalty = 1

            G.edges[u, v]["weight"] = (
                # small penalty for offset further from 0
                offset_penalty
                - source_offset_in_edges[u]
                - dest_offset_out_edges[v]
            )

        moves = []

        flow_dict = nx.algorithms.max_flow_min_cost(G, "s", "t")
        flow_value = sum(flow_dict["s"].values())

        for u, v in G.edges():
            # try:
            G[u][v]["flow"] = flow_dict[u][v]

        if selected_node:
            # If we are trying to expand an existing move, check the selected node is still in the graph
            if selected_node not in G:
                return
            selectable_nodes = [(0, selected_node)]

        else:
            selectable_nodes = [
                (sum(flow_dict[n].values()), n)
                for n in G.nodes
                if (n[0] == "s_ofst" or n[0] == "d_ofst")
                and sum(flow_dict[n].values()) > 0
            ]
            selectable_nodes.sort(key=lambda x: -x[0])

        for node_score, node_choice in selectable_nodes:

            # Initialise a possible move associated with this node
            if not selected_node:
                move = Move()
                moves.append(move)

            # Dict of (source, col, lc): [ops]
            aspirates = move.aspirates
            # Dict of (dest, col): [ops]
            dispenses = move.dispenses
            asp_ops = move.ops
            tips_used = move.tips
            new_ops = []

            if node_choice[0] == "s_ofst":
                # Pre-select edges matching the source offset node for efficiency
                selectable_edges = list(G.out_edges(node_choice))
            else:
                selectable_edges = G.in_edges(node_choice)

            selectable_edges = sorted(
                selectable_edges, key=lambda x: G.edges[*x]["wash_needed"]
            )

            for u, v in selectable_edges:

                (_, _, source, col, _, offset) = u

                if G.edges[u, v]["wash_needed"]:
                    move.wash_first = True
                    move.tip_ops_since_wash = self.tip_ops_since_wash

                new_ops, new_tips = zip(*G[u][v]["op_tip"].items())
                if any([op in asp_ops for op in new_ops]):
                    continue

                if len(new_ops) == 0:
                    continue
                if any(tip in tips_used for tip in new_tips):
                    continue

                tips_used += new_tips

                asp_ops += new_ops
                if (source, col, offset) in aspirates:
                    aspirates[source, col, offset] += new_ops
                else:
                    aspirates[source, col, offset] = list(new_ops)

                (_, _, dest, col, _, offset) = v

                if (dest, col, offset) in dispenses:
                    dispenses[dest, col, offset] += new_ops
                else:
                    dispenses[dest, col, offset] = list(new_ops)
                break

            move.calc_score()

            # Given these tips used, calculate other edges that can be added
            if new_ops and len(move.tips) < 8 and flow_value > len(new_ops):
                self.get_moves(tips_used, node_choice, move, selected_ops=set(asp_ops))
                pass

            #

        moves.sort(reverse=True)
        return moves


def show_graph(G):
    """
    Show the graph in an interactive window. Useful for debugging and checking that edges are correctly set to possible moves.
    If the graph has flow values set, edges with flow will be highlighted red and have flow/capacity/cost values shown
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
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, alpha=0.1)  # faint background edges

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

    node_labels = {node: str(node) for node in G.nodes}

    # Draw the labels on the edges
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    nx.draw_networkx_labels(G, pos, labels=node_labels)

    plt.show()
