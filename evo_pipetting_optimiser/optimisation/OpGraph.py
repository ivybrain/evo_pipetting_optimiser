from matplotlib import pyplot as plt
import networkx as nx
from .Move import Move
from functools import reduce
import heapq

# By default, we only track edges between offset nodes, as this is all that is used to calculate moves
# Set DEBUG=True to also track edges for source/dest rows and columns - this is useful
# to show the full context of the graph in show_graph()
# Leave False for routine use - speeds up execution ~10%
DEBUG = True

class OptimisationError(Exception):
    pass

def source_key(op):
    return ("s", op.source.name, op.source_pos[1], op.liquid_class)


def dest_key(op):
    return ("t", op.destination.name, op.dest_pos[1], op.liquid_class)


def _add_edge(edges, out_adj, in_adj, u, v, **attrs):
    """
    Add or update an edge (u, v) with the given attributes.
    Replacement for networkx.DiGraph using plain dicts that achieves a ~25% speedup.
    Mirrors networkx's add_edge semantics: re-adding an existing edge updates
    its attributes but does not duplicate the adjacency entries used by
    out_adj/in_adj lookups.
    """
    is_new = (u, v) not in edges
    edges[(u, v)] = attrs
    if is_new:
        out_adj.setdefault(u, []).append((u, v))
        in_adj.setdefault(v, []).append((u, v))


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
        self.op_groups = {}
        self.op_to_group = {}

        # If debug, track source row and column edges
        if DEBUG:
            self.edges = {}
            self.out_adj = {}
            self.in_adj = {}

        for op in ops:

            if DEBUG:
                if ("s", source_key(op)) not in self.edges:
                    _add_edge(self.edges, self.out_adj, self.in_adj, "s", source_key(op))
                if (dest_key(op), "t") not in self.edges:
                    _add_edge(self.edges, self.out_adj, self.in_adj, dest_key(op), "t")

            # Add groups representing edges from source to dest
            # but made up of a subgraph to calculate tip use and cost
            if (source_key(op), dest_key(op)) in self.op_groups:
                self.op_groups[(source_key(op), dest_key(op))].append(op)
            else:

                self.op_groups[(source_key(op), dest_key(op))] = [op]

            self.op_to_group[op] = (source_key(op), dest_key(op))

        return

    def get_moves(self):

        # Only the source_offset <-> dest_offset edges are ever read by the
        # move-selection loop below, so that's the only graph we build unless
        # DEBUG is set, in which case we also add the row/column edges that
        # show_graph() needs for a legible visualisation
        if DEBUG:
            G = dict(self.edges)
            out_adj = {k: list(v) for k, v in self.out_adj.items()}
            in_adj = {k: list(v) for k, v in self.in_adj.items()}
        else:
            G = {}
            out_adj = {}
            in_adj = {}

        # Use a dict to track the nodes we add - this gives us set dedup behaviour
        # But retains deterministic order
        offset_nodes = {}

        for (source_col, dest_col), group_ops in self.op_groups.items():
            for op in group_ops:

                # Calculate the feasible tips for this op given the previous wash state of the tips
                free_tips = set()
                tips_with_wash = set()

                # We need to look at all past ops, across tips, since last wash, to see if there is a tip type mounted
                # If there has been a tip type mounted, we can't mount another type without wash
                tip_mounted = None
                for past_ops in self.tip_ops_since_wash.values():
                    if past_ops:
                        tip_mounted = tip_mounted or past_ops[0].tip_index

                for tip, past_ops in self.tip_ops_since_wash.items():

                    # Check if there's a specific allowed tip only
                    if op.tip is not None and op.tip != tip:
                        continue

                    if op.allowed_tips and tip not in op.allowed_tips:
                        continue

                    # Check we have not reached the max use limit for this tip
                    if len(past_ops) >= op.worklist.wash_params["max_tip_uses"]:
                        tips_with_wash.add(tip)
                        continue

                    # Check that we have not previously mounted a tip that doesn't match this tip type
                    if tip_mounted is not None and tip_mounted != op.tip_index:
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

                if DEBUG:
                    source_row_key = (*source_col, op.source_pos[0])
                    if (source_col, source_row_key) not in G:
                        _add_edge(G, out_adj, in_adj, source_col, source_row_key)

                    dest_row_key = (*dest_col, op.dest_pos[0])
                    if (dest_row_key, dest_col) not in G:
                        _add_edge(G, out_adj, in_adj, dest_row_key, dest_col)

                def add_offset_edge(
                    source_offset,
                    dest_offset,
                    tip=0,
                    wash_needed=False,
                ):
                    source_offset_key = ("s_ofst", *source_col, source_offset)
                    dest_offset_key = ("d_ofst", *dest_col, dest_offset)

                    if DEBUG:
                        _add_edge(G, out_adj, in_adj, source_row_key, source_offset_key)

                    if (source_offset_key, dest_offset_key) in G:
                        G[(source_offset_key, dest_offset_key)]["op_tip"][op] = tip
                        G[(source_offset_key, dest_offset_key)]["wash_needed"] = (
                            G[(source_offset_key, dest_offset_key)]["wash_needed"]
                            or wash_needed
                        )

                    else:
                        _add_edge(
                            G, out_adj, in_adj,
                            source_offset_key,
                            dest_offset_key,
                            op_tip={op: tip},
                            wash_needed=wash_needed,
                        )

                    if DEBUG:
                        _add_edge(G, out_adj, in_adj, dest_offset_key, dest_row_key)

                    offset_nodes[source_offset_key] = None
                    offset_nodes[dest_offset_key] = None

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
                            tip=max(source_tip, dest_tip),
                            wash_needed=wash_needed,
                        )


        

        moves = []


        # nodes_in_moves = []

        # Loop through all selectable nodes and create a 'move' from each node as a starting point
        for node_choice in offset_nodes:

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
                selectable_edges = list(out_adj.get(node_choice, []))
            else:
                selectable_edges = in_adj.get(node_choice, [])

            # How we sort and prioritise the edges
            def edge_sort_key(edge):

                return (G[edge]["wash_needed"],
                -len(G[edge]["op_tip"]))

            

            edge_heap = []

            # Add the selectable edges to a heap
            for edge in selectable_edges:
                heapq.heappush(edge_heap, (edge_sort_key(edge), edge))

                                           
            new_ops = None

            while edge_heap:

                _, (u,v) = heapq.heappop(edge_heap)

                (_, _, source, col, _, offset) = u

                if G[u, v]["wash_needed"]:
                    move.wash_first = True
                    move.tip_ops_since_wash = self.tip_ops_since_wash

                potential_ops, potential_tips = zip(*G[u, v]["op_tip"].items())
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
                    new_edges = list(in_adj.get(new_node, []))
                    new_edges.remove((u, v))
                else:
                    # Otherwise, a source node is being added
                    new_node = u
                    # Add the out-edges for this source node (except current edge)
                    new_edges = list(out_adj.get(new_node, []))
                    new_edges.remove((u,v))

                # Add the new edges to heap to be processed next by smallest key
                for edge in new_edges:
                    heapq.heappush(edge_heap, (edge_sort_key(edge), edge))

                nodes_this_move.append(new_node)
                #nodes_in_moves.append(new_node)

                # If we have the potential to add new ops, (we have less than 8 selected, and more than the number selected
                # were included in a flow)
                # then continue looping to add more ops from selectable edges. Otherwise, end the loop
                if not (new_ops and len(move.tips) < 8):
                    break

            move.calc_score()


        moves.sort(reverse=True)
        return moves


def show_graph(edges):
    """
    Show the graph in an interactive window. Useful for debugging and checking that edges are correctly set to possible moves.

    `edges` is the plain {(u, v): attrs} dict built in get_moves() (also called `G` there) -
    this rebuilds a throwaway networkx graph from it purely for visualisation.

    To use this, I set a breakpoint in the Opgraph method and then call it from the debugging console with
    show_graph(G)
    Note: The plt.show method does not work well inside a threaded approach. If you have issues, replace the
    `executor.map` in AutoWorklist.py with a plain map'
    """
    G = nx.DiGraph()
    for (u, v), attrs in edges.items():
        G.add_edge(u, v, **attrs)

    

    pos = nx.bfs_layout(G, start="s")


    # Draw nodes and all edges 
    nx.draw_networkx_nodes(G, pos, alpha=0.75, nodelist=[n for n in G.nodes if n not in ["s", "t"]])
    nx.draw_networkx_edges(G, pos, alpha=0.2, edgelist=[e for e in G.edges if e[0] != "s" and e[1] != "t"]) 

    edges_of_interest = [
        (('s_ofst', 's', 'smp', 0, 'Water free dispense', 0), ('d_ofst', 't', 'dilplate', 0, 'Water free dispense', 0)),
        (('s_ofst', 's', 'smp', 1, 'Water free dispense', -3), ('d_ofst', 't', 'dilplate', 0, 'Water free dispense', 0)),
        (('s_ofst', 's', 'smp', 1, 'Water free dispense', -3), ('d_ofst', 't', 'dilplate', 1, 'Water free dispense', -4)),
        (('s_ofst', 's', 'smp', 2, 'Water free dispense', -6), ('d_ofst', 't', 'dilplate', 1, 'Water free dispense', -4)),
    ]

    # Draw specific edges of interest
    nx.draw_networkx_edges(G, pos, alpha=0.75, edge_color='r', width=2, edgelist=edges_of_interest) 



    # Extract the flow values for labeling
    op_labels = nx.get_edge_attributes(G, "op_tip")

    edge_labels = {
        key: ",".join(str(op.id) for op in op_tip.keys())
        for key, op_tip in op_labels.items()
        if key in edges_of_interest
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
