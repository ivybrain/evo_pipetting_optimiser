from .group_ops import group_ops
from .group_by import init_search
from copy import copy, deepcopy
import cProfile, pstats
import time
import networkx as nx
from collections import deque
from sortedcontainers import SortedSet, SortedKeyList
from matplotlib import pyplot as plt
import itertools


class OffsetLimitException(Exception):
    pass


# Abstract
class PipettingGraph(nx.DiGraph):
    def show(self):
        G = self.copy()

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


class TipGraph(PipettingGraph):

    def __init__(self, ops=None):
        """
        Init a graph from a set of ops
        Graph used to decide which rows to pipette and which tips to use
        Between a single col source and single col dest
        Graph has 6 'columns'
        source node
        source labware rows
        all possible offsets for source labware
        all possible offsets for dest labware
        dest labware rows
        sink node
        """
        super().__init__()

        if ops:
            G = self
            scaling = 1
            demand = scaling * min(8, len(ops))
            G.add_node("s", demand=-demand)
            G.add_node("t", demand=demand)

            self.asp_nodes = {}
            self.disp_nodes = {}
            source_row_nodes = {}
            dest_row_nodes = {}
            for op in ops:

                source_row_key = (*source_key(op), op.source_pos[0])
                source_row_nodes[source_row_key] = True

                if ("s", source_row_key) in G.edges:
                    G.edges[("s", source_row_key)]["capacity"] += scaling
                else:
                    G.add_edge("s", source_row_key, capacity=scaling, weight=0)

                dest_row_key = (*dest_key(op), op.dest_pos[0])
                dest_row_nodes[dest_row_key] = True
                if (dest_row_key, "t") in G.edges:
                    G.edges[(dest_row_key, "t")]["capacity"] += scaling
                else:
                    G.add_edge(dest_row_key, "t", capacity=scaling, weight=0)

                for source_offset in range(op.source_pos[0] - 7, op.source_pos[0] + 1):

                    # Add edges from rows to possible offsets

                    dest_offset = source_offset + (op.dest_pos[0] - op.source_pos[0])

                    # Check if this offset is feasible for the dest row
                    if dest_offset > op.dest_pos[0] or dest_offset < op.dest_pos[0] - 7:
                        continue

                    source_offset_key = ("s_ofst", *source_key(op), source_offset)
                    dest_offset_key = ("d_ofst", *dest_key(op), dest_offset)
                    if source_offset_key not in self.asp_nodes:
                        self.asp_nodes[source_offset_key] = 0
                    if dest_offset_key not in self.disp_nodes:
                        self.disp_nodes[dest_offset_key] = 0

                    if (source_row_key, source_offset_key) not in G.edges:
                        self.asp_nodes[source_offset_key] += 1
                    if (dest_row_key, dest_offset_key) not in G.edges:
                        self.disp_nodes[dest_offset_key] += 1
                    G.add_edge(
                        source_row_key,
                        source_offset_key,
                        capacity=scaling,
                        weight=0,
                    )

                    # Set weight based on how many incident edges the nodes at each end have
                    if (source_offset_key, dest_offset_key) in G.edges:
                        G.edges[(source_offset_key, dest_offset_key)][
                            "capacity"
                        ] += scaling
                        G.edges[(source_offset_key, dest_offset_key)]["ops"].append(op)

                    else:
                        G.add_edge(
                            source_offset_key,
                            dest_offset_key,
                            capacity=scaling,
                            weight=0,
                            ops=[op],
                        )

                    G.add_edge(
                        dest_offset_key,
                        dest_row_key,
                        capacity=scaling,
                        weight=0,
                    )

            # Re-weight source row to source offset edges based on how many sources have edges going to the same offset
            # This incentivises choosing edges to offsets that have as many compatible rows as possible
            for source_node in source_row_nodes.keys():
                for source_node_offset, source_count in self.asp_nodes.items():
                    if (source_node, source_node_offset) in G.edges:
                        G.edges[source_node, source_node_offset]["weight"] = (
                            9 - source_count
                        )

            # Same for dest
            for dest_node in dest_row_nodes.keys():
                for dest_node_offset, dest_count in self.disp_nodes.items():
                    if (dest_node_offset, dest_node) in G.edges:
                        G.edges[dest_node_offset, dest_node]["weight"] = 9 - dest_count

            # Add weight for central edges that's 16 - the number of out edges from offset - number of in edges to dest offset
            # for source_node, source_count in asp_nodes.items():
            #     for dest_node, dest_count in disp_nodes.items():
            #         if (source_node, dest_node) in G.edges:
            #             G.edges[source_node, dest_node]["weight"] = (
            #                 17 - source_count - dest_count
            #             )

    def get_move(self):
        G = self

        flow_cost, flow_dict = nx.network_simplex(G)

        # Dict of (source, row, offset): [ops]
        aspirates = {}
        # Dict of (dest, row, offset): ops
        dispenses = {}
        for u, v in G.edges():
            try:
                G[u][v]["flow"] = flow_dict[u][v]
                if flow_dict[u][v] > 0 and u in self.asp_nodes:
                    (_, source, row, _, offset) = u
                    if (source, row, offset) in aspirates:
                        aspirates[source, row, offset] += G[u][v]["ops"]
                    else:
                        aspirates[source, row, offset] = copy(G[u][v]["ops"])

                if flow_dict[u][v] > 0 and v in self.disp_nodes:
                    (_, dest, row, _, offset) = v
                    if (dest, row, offset) in dispenses:
                        dispenses[dest, row, offset] += G[u][v]["ops"]
                    else:
                        dispenses[dest, row, offset] = copy(G[u][v]["ops"])

            except KeyError:
                G[u][v]["flow"] = 0

        print(flow_cost, binary_cost(G, flow_dict))
        return aspirates, dispenses

    def gen_empty(self):
        pass


class ColGraph(PipettingGraph):
    def __init__(self, ops=None):
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

        if ops:
            G = self
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

            # Populate each group with the subgraph
            self.group_asp_nodes = {}
            self.group_disp_nodes = {}
            self.group_edges = {}
            for (source_col, dest_col), ops in self.op_groups.items():
                self.create_subgraph(source_col, dest_col, ops)

            # max_capacity = 0
            # for u, v in self.op_groups:
            #     max_capacity = max(max_capacity, G[u][v]["capacity"])

            # for u, v in self.op_groups:
            #     G[u][v]["weight"] = 1 + (max_capacity - G[u][v]["capacity"])
            #     # perfect cost would be 1 / capcaity but that doesn't work with simplex as it integerfies everything
            #     # Potentially make an integer only approximation

            return

    # def remove_ops(self, ops):
    #     for op in ops:
    #         self.remove_edge(self.op_to_edge[op])

    def create_subgraph(self, source_col, dest_col, ops):
        G = self
        scaling = 1

        self.group_asp_nodes[(source_col, dest_col)] = {}
        asp_nodes = self.group_asp_nodes[(source_col, dest_col)]
        self.group_disp_nodes[(source_col, dest_col)] = {}
        disp_nodes = self.group_disp_nodes[(source_col, dest_col)]
        self.group_edges[(source_col, dest_col)] = []
        group_edges = self.group_edges[(source_col, dest_col)]
        source_row_nodes = {}
        dest_row_nodes = {}
        for op in ops:

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

            # TODO: Add trough handling

            def add_offset_edge(
                source_offset, dest_offset, source_capacity, dest_capacity
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
                G.add_edge(
                    source_row_key,
                    source_offset_key,
                    capacity=source_capacity,
                    weight=0,
                )

                # Set weight based on how many incident edges the nodes at each end have
                if (source_offset_key, dest_offset_key) in G.edges:
                    G.edges[(source_offset_key, dest_offset_key)]["capacity"] += scaling
                    G.edges[(source_offset_key, dest_offset_key)]["ops"].append(op)

                else:
                    group_edges.append((source_offset_key, dest_offset_key))
                    G.add_edge(
                        source_offset_key,
                        dest_offset_key,
                        capacity=scaling,
                        weight=0,
                        ops=[op],
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
                    dest_offset_range = range(op.dest_pos[0] - 7, op.dest_pos[0] + 1)

            else:
                source_offset_range = range(op.source_pos[0] - 7, op.source_pos[0] + 1)
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
                    if dest_offset > op.dest_pos[0] or dest_offset < op.dest_pos[0] - 7:
                        continue

                    source_tip = op.source_pos[0] - source_offset
                    dest_tip = op.dest_pos[0] - dest_offset

                    assert 0 <= source_tip <= 7, "Tip out of range"
                    assert 0 <= dest_tip <= 7, "Tip out of range"

                    add_offset_edge(
                        source_offset,
                        dest_offset,
                        max(1, source_virtual_rows),
                        max(1, dest_virtual_rows),
                    )

                # Add edges from rows to possible offsets

        # Re-weight source row to source offset edges based on how many sources have edges going to the same offset
        # This incentivises choosing edges to offsets that have as many compatible rows as possible
        for source_node in source_row_nodes.keys():
            for source_node_offset, source_count in asp_nodes.items():
                if (source_node, source_node_offset) in G.edges:
                    G.edges[source_node, source_node_offset]["weight"] = (
                        9 - source_count
                    )

        # Same for dest
        for dest_node in dest_row_nodes.keys():
            for dest_node_offset, dest_count in disp_nodes.items():
                if (dest_node_offset, dest_node) in G.edges:
                    G.edges[dest_node_offset, dest_node]["weight"] = 9 - dest_count

    def get_move(self):
        G = self

        max_flow = nx.algorithms.maximum_flow_value(G, "s", "t")

        G.nodes["s"]["demand"] = -max_flow
        G.nodes["t"]["demand"] = max_flow

        flow_cost, flow_dict = nx.network_simplex(G)

        # Dict of (source, col, lc): [ops]
        aspirates = {}
        # Dict of (dest, col): [ops]
        dispenses = {}
        asp_ops = []
        disp_ops = []
        for u, v in G.edges():
            # try:
            G[u][v]["flow"] = flow_dict[u][v]

        op_groups_sorted = sorted(self.op_groups.items(), key=lambda x: -len(x[1]))

        # NOT THIS as it doesn't account for a one-to-many.
        # Choose single asp node or single disp node
        edge_choices = [
            ((u, v), G.edges[u, v])
            for u, v in G.edges
            if u[0] == "s_ofst" and v[0] == "d_ofst" and flow_dict[u][v] > 0
        ]

        edge_choices.sort(key=lambda x: (-x[1]["flow"], -len(x[1]["ops"])))
        # For now, make single choice
        edge_choice = edge_choices[0]

        (u, v) = edge_choice[0]
        (_, _, source, s_col, _, s_offset) = u
        (_, _, dest, d_col, _, d_offset) = v
        ops = G[u][v]["ops"]
        aspirates[source, s_col, s_offset] = ops
        dispenses[dest, d_col, d_offset] = ops

        print(
            f"Selecting ops. Moves {len(aspirates) + len(dispenses)}, ops {len(ops)}, cost {flow_cost}"
        )

        return aspirates, dispenses, ops


def binary_cost(G, flow_dict):
    cost = 0
    for u, v in G.edges:
        if G.edges[u, v]["weight"] > 0 and flow_dict[u][v] > 0:
            cost += 1
    return cost


class Node:
    """
    Node in the problem graph
    """

    def __init__(self, parent, *args):

        self.parent = parent

        if not parent:
            (pending_ops,) = args
            self.init_root(pending_ops)

            # nx.draw(self.pending_graph, with_labels=True)
            # plt.show()

            # nx.draw(self.open_graph, with_labels=True)
            # plt.show()

        else:
            (pending_ops, open_ops, open_graph, move, cost) = args

            self.open_graph = open_graph
            self.pending_ops = pending_ops
            self.open_ops = open_ops
            self.depth = parent.depth + 1

            self.cost = cost
            self.move = move

        self.fscore = self.cost + self.heuristic_cost_estimate()

        return

    def init_root(self, pending_ops):
        self.pending_ops = pending_ops

        self.open_ops = set(
            [
                op
                for op in pending_ops
                if op.source_dep not in pending_ops and op.dest_dep not in pending_ops
            ]
        )

        # self.pending_graph = TipGraph(pending_ops)
        self.open_graph = ColGraph(self.open_ops)
        # self.open_graph = TipGraph()

        self.move = None
        self.cost = 0
        self.depth = 0

        return

    def get_neighbours(self):

        children = []

        aspirates, dispenses, ops = self.open_graph.get_move()

        # self.open_graph.show()

        new_pending = copy(self.pending_ops)

        new_pending.difference_update(set(ops))
        all_open_ops = set(
            [
                op
                for op in new_pending
                if op.source_dep not in new_pending and op.dest_dep not in new_pending
            ]
        )

        # Optimisation: do by copy and modification of old graph rather than rebuild
        new_graph = ColGraph(all_open_ops)

        new_node = Node(
            self,
            new_pending,
            all_open_ops,
            new_graph,
            (aspirates, dispenses),
            len(aspirates) + len(dispenses),
        )
        children.append(new_node)
        return children

    def heuristic_cost_estimate(self):
        score = 0

        return score

    def key(self):
        # If pending_ops is the same, the graph must have the same nodes
        return (
            self.fscore,
            tuple(self.pending_ops),
            tuple(self.open_graph.edges),
        )

    def __hash__(self):
        return hash(self.key())

    # Hashes will collide whenever edges and vertices are the same - we then use isomorphic to check equality
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False

        if self.key() != other.key():
            return False

        iso = nx.is_isomorphic(self.open_graph, other.open_graph)
        return iso

    def __repr__(self):
        return f"Node score {self.fscore}"


def source_key(op):
    return ("s", op.source.name, op.source_pos[1], op.liquid_class)


def dest_key(op):
    return ("t", op.destination.name, op.dest_pos[1], op.liquid_class)


def init_search(pending_ops):

    initial_node = Node(None, pending_ops)
    open_nodes = SortedSet([initial_node], key=lambda node: (node.fscore, -node.depth))
    generated_nodes = set([initial_node])
    node = None
    max_len = 0
    prune_count = 0
    while len(open_nodes) > 0:
        max_len = max(max_len, len(open_nodes))

        node = open_nodes.pop(0)

        if len(node.open_graph) == 0:
            break

        children = node.get_neighbours()
        # print(
        #     f"Depth {node.depth}, score {node.fscore}, queue {len(open_nodes)}, new_children {len(children)}, pruned {prune_count}"
        # )

        for child in children:
            if child in generated_nodes:
                prune_count += 1
                continue
            else:
                generated_nodes.add(child)
                open_nodes.add(child)

    path = [node]
    while node.parent:
        path.append(node.parent)
        node = node.parent

    path.reverse()
    moves = []
    for node in path:

        if node.move:
            moves.append(node.move)
            # print(node.move)

        # nx.draw(node.open_graph, with_labels=True)
        # plt.show()
        continue

    return moves


def make_plan(pending_ops):
    """
    Create a list of pipetting actions from a set of pending ops
    Actions are a tuple of
    (Type, [ops], [tips])
    Where type is one of "Aspirate", "Dispense", "Wash"
    """

    # Count the aspirates, dispenses, and washes we use
    # for performance tracking
    asp_count = 0
    disp_count = 0
    wash_count = 0

    actions = []

    # Create a copy of pending ops set that we can mutate
    pending_ops = copy(pending_ops)

    completed_ops = set()

    labware_copies = {}

    def labware_copy(labware):
        if labware.name not in labware_copies:
            labware_copies[labware.name] = deepcopy(labware)

        return labware_copies[labware.name]

    start_time = time.time()

    prev_tip_op = {}
    # profiler = cProfile.Profile()
    # profiler.enable()

    moves = init_search(pending_ops)

    for aspirates, dispenses in moves:
        ops = []
        tips_selected = []
        for (labware, col, offset), asp_ops in aspirates.items():
            ops += asp_ops
            # Calculate tip from offset
            for op in asp_ops:
                # If source is a trough, don't assign a tip for now. Assign  from dest
                if getattr(op.source, "virtual_rows", 0) or 0 > op.source_pos[0]:
                    # If dest is also a trough, just assign tips sequentially
                    if getattr(op.destination, "virtual_rows", 0) or 0 > op.dest_pos[0]:
                        tips_selected.append(len(tips_selected) - 1)
                    else:
                        # Find the dest offset used for this op and use it to calculate tip
                        for (_, _, dest_offset), disp_ops in dispenses.items():
                            if op in disp_ops:
                                break

                        tips_selected.append(op.dest_pos[0] - dest_offset)
                else:
                    tips_selected.append(op.source_pos[0] - offset)

        op_tip = {op: tip + 1 for op, tip in zip(ops, tips_selected)}

        # Check if we need to wash any tips
        wash_needed = False
        for tip, op in zip(tips_selected, ops):
            # Standard or decontamination mean we require a wash
            if op.wash_scheme in [1, "D"] and tip in prev_tip_op:
                wash_needed = True

            if op.wash_scheme == "smart" and tip in prev_tip_op:
                past_op = prev_tip_op[tip]

                past_source = labware_copy(past_op.source)
                past_dest = labware_copy(past_op.destination)
                source = labware_copy(op.source)
                dest = labware_copy(op.destination)

                past_source_comp = past_source.get_well_composition(
                    past_source.wells[past_op.source_pos]
                )

                past_dest_comp = past_dest.get_well_composition(
                    past_dest.wells[past_op.dest_pos]
                )

                source_comp = source.get_well_composition(source.wells[op.source_pos])

                past_touched = list(past_source_comp.items()) + list(
                    past_dest_comp.items()
                )

                for prev_key, prev_conc in past_touched:
                    # If the previous op touched a liquid that is not present in our source, we need to wash to not contaminate it
                    if prev_key not in source_comp:
                        wash_needed = True
                        break

                    # If last op touched a well with more than twice the conc of the source, wash to not contaminate with the higher conc
                    if prev_conc > 2 * source_comp[prev_key]:
                        wash_needed = True
                        break

                # If we've already determined we need a wash, no point checking the other tips
                if wash_needed:
                    break

            prev_tip_op[tip] = op

        if wash_needed:
            actions.append(("Wash", ops, list(op_tip.values())))
            wash_count += 1

        # Sort the source list and dest list by the tips used
        # This just makes sure the pipetting occurs in an order that's less confusing visually,
        # starting from tip 1 to tip 8
        group_sort_key = lambda ops: min([op_tip[op] for op in ops])

        for (labware, col, offset), asp_ops in aspirates.items():

            # Sort the group by assigned tip
            asp_ops.sort(key=lambda op: op_tip[op])

            actions.append(("Aspirate", asp_ops, [op_tip[op] for op in asp_ops]))

            # Remove liquid from our virtual copy of the plates so we can track composition

            virtual_source = labware_copy(asp_ops[0].source)
            virtual_source.remove(
                [virtual_source.wells[op.source_pos] for op in asp_ops],
                [op.volume for op in asp_ops],
            )

            asp_count += 1

        for (labware, col, offset), disp_ops in dispenses.items():

            # Sort the group by assigned tip
            disp_ops.sort(key=lambda op: op_tip[op])

            actions.append(("Dispense", disp_ops, [op_tip[op] for op in disp_ops]))

            # Add liquid to our virtual copy of the plates so we can track composition
            virtual_dest = labware_copy(disp_ops[0].destination)
            compositions = [
                labware_copy(op.source).get_well_composition(
                    op.source.wells[op.source_pos]
                )
                for op in disp_ops
            ]
            virtual_dest.add(
                [virtual_dest.wells[op.dest_pos] for op in disp_ops],
                [op.volume for op in disp_ops],
                compositions=compositions,
            )

            disp_count += 1

        selected_ops = ops
        pending_ops.difference_update(selected_ops)
        completed_ops.update(selected_ops)

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats("program.cprof")

    cost = asp_count + disp_count + (wash_count * 2)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")

    return actions, cost, (asp_count, disp_count, wash_count)
