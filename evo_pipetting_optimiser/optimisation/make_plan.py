from .group_ops import group_ops
from .group_by import init_search
from copy import copy, deepcopy
import cProfile, pstats
import time
import networkx as nx
from sortedcontainers import SortedSet
from matplotlib import pyplot as plt
from ..utils import longest_consecutive_sequence
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)


class OffsetLimitException(Exception):
    pass


class Move:
    def __init__(self):
        self.aspirates = {}
        self.dispenses = {}
        self.ops = []
        self.tips = []

        self.wash_first = False

    def calc_score(self):
        # Score the efficiency of the move
        self.score = (len(self.ops) * 2) / (
            len(self.aspirates) + len(self.dispenses) + (2 if self.wash_first else 0)
        )

    def key(self):
        asp_ops = tuple(op for set in self.aspirates.values() for op in set)
        disp_ops = tuple(op for set in self.dispenses.values() for op in set)

        return (self.score, asp_ops, disp_ops)

    def tuple(self):
        return self.score, self.aspirates, self.dispenses, self.ops, self.tips

    def __repr__(self):
        return str(self.tuple())

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self.key() == other.key()

    def __hash__(self):
        return hash(self.key)

    def __lt__(self, other):
        return self.key() < other.key()

    def __gt__(self, other):
        return self.key() > other.key()


def show_graph(G):
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

    # def remove_ops(self, ops):
    #     for op in ops:
    #         self.remove_edge(self.op_to_edge[op])

    def check_wash(self, ops, tips):
        """
        Check if a wash is needed given parent ops
        """
        if not (self.node.parent and self.node.parent.move):
            return False

        return True
        # if all(tip in )

        return True
        prev_ops, prev_tips = self.node.parent.move[-2:]
        prev_tip_op = {tip: op for op, tip in zip(prev_ops, prev_tips)}
        for op, tip in zip(ops, tips):
            # Standard or decontamination mean we require a wash
            if op.wash_scheme in [1, "D"] and tip in prev_tip_op:
                return True

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

    def get_moves(
        self, occupied_tips=set(), selected_node=None, move=None, selected_ops=set()
    ):

        # Create a copy of the graph to populate with the subgraphs
        G = self.G.copy()

        # Create subgraph representing offsets
        offset_edges = []
        source_offset_out_edges = {}
        dest_offset_in_edges = {}

        for (source_col, dest_col), ops in self.op_groups.items():
            scaling = 1
            asp_nodes = {}
            disp_nodes = {}
            group_edges = []
            source_row_nodes = {}
            dest_row_nodes = {}
            for op in ops:
                if op in selected_ops:
                    continue

                # Calculate the feasible tips for this op given the previous wash state of the tips
                free_tips = set()
                tips_with_wash = set()
                for tip, ops in self.tip_ops_since_wash.items():
                    # If tip is occupied, skip it
                    if tip in occupied_tips:
                        continue

                    # If tip is unused, we can use it
                    if not ops:
                        free_tips.add(tip)
                        continue

                    # If tip has only ops with wash=None, we can use
                    if all(op.wash_scheme == None for op in ops):
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

                # TODO: Add trough handling

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

                    if source_offset_key not in source_offset_out_edges:
                        source_offset_out_edges[source_offset_key] = 1
                    else:
                        source_offset_out_edges[source_offset_key] += 1

                    if dest_offset_key not in dest_offset_in_edges:
                        dest_offset_in_edges[dest_offset_key] = 1
                    else:
                        dest_offset_in_edges[dest_offset_key] += 1

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
            G.edges[u, v]["weight"] = (
                0 - source_offset_out_edges[u] - dest_offset_in_edges[v]
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

        moves.sort(key=lambda move: move.score, reverse=True)
        return moves


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
        self.open_graph = OpGraph(self.open_ops)
        self.open_graph.node = self
        # self.open_graph = TipGraph()

        self.move = None
        self.cost = 0
        self.depth = 0

        return

    @property
    def unused_tips(self):
        return [i for i in range(8) if not self.open_graph.tip_ops_since_wash[i]]

    def get_neighbours(self):
        children = []

        # If we're not the first node, we can consider a wash

        tip_ops_since_wash = {t: [] for t in range(8)}
        if self.depth > 0:
            # Find all the ops conducted and tips used since last wash
            ops = []
            tips = []
            n = self
            while True:
                ops += n.move.ops
                tips += n.move.tips

                for op, tip in zip(n.move.ops, n.move.tips):
                    tip_ops_since_wash[tip].append(op)

                if n.move.wash_first or not (n.parent and n.parent.move):
                    break

                n = n.parent

            self.open_graph.tip_ops_since_wash = tip_ops_since_wash

        moves = self.open_graph.get_moves()
        moves = moves[:1]

        # self.open_graph.show()

        for move in moves:
            score, aspirates, dispenses, ops, tips = move.tuple()

            new_pending = copy(self.pending_ops)

            new_pending.difference_update(set(ops))
            all_open_ops = set(
                [
                    op
                    for op in new_pending
                    if op.source_dep not in new_pending
                    and op.dest_dep not in new_pending
                ]
            )

            new_node = Node(
                self,
                new_pending,
                all_open_ops,
                OpGraph(all_open_ops),
                move,
                self.cost + len(aspirates) + len(dispenses),
            )
            new_node.open_graph.node = new_node
            children.append(new_node)

        if not children:
            raise Exception(
                "Failed to solve. Perhaps you are trying to pipette wells with impossible offsets?"
            )

        return children

    def heuristic_cost_estimate(self):
        # op_cost = 0

        # for n_ops in list(self.open_graph.asp_nodes.values()) + list(
        #     self.open_graph.disp_nodes.values()
        # ):
        #     op_cost += (n_ops // 8) + 1

        op_cost = (len(self.pending_ops) * 18) / 8

        longest_seq = longest_consecutive_sequence(self.unused_tips)
        if len(self.pending_ops) > longest_seq:
            # Give a slight preference for having more contiguous tips
            op_cost -= (len(self.unused_tips) / 5) + (longest_seq / 5)

        return op_cost - (self.depth)

    def key(self):
        # If pending_ops is the same, the graph must have the same nodes
        return (self.fscore, tuple(self.pending_ops), len(self.unused_tips))

    def compare_key(self):
        return (
            self.fscore,
            len(tuple(self.pending_ops)),
            -self.depth,
            -longest_consecutive_sequence(self.unused_tips),
            -len(self.unused_tips),
            self.move,
        )

    def __lt__(self, other):
        return self.compare_key() < other.compare_key()

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False

        if self.key() != other.key():
            return False

        return True

    def __repr__(self):
        return f"Node score {self.fscore}"


def source_key(op):
    return ("s", op.source.name, op.source_pos[1], op.liquid_class)


def dest_key(op):
    return ("t", op.destination.name, op.dest_pos[1], op.liquid_class)


def init_search(pending_ops, report=True):

    # TODO: if certain time has elapsed, reweght to depth first

    initial_node = Node(None, pending_ops)
    open_nodes = SortedSet([initial_node])
    generated_nodes = set([initial_node])
    node = None
    max_len = 0
    prune_count = 0

    if report:
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        task = progress.add_task("Optimising...", total=len(pending_ops))
        progress.start()

    while len(open_nodes) > 0:
        max_len = max(max_len, len(open_nodes))

        node = open_nodes.pop(0)

        if report and node.move:
            progress.update(task, advance=len(node.move.ops))

        if len(node.open_ops) == 0:
            break

        children = node.get_neighbours()
        # print(
        #     f"Depth {node.depth}, score {node.fscore}, queue {len(open_nodes)}, new_children {len(children)}, pruned {prune_count}"
        # )

        for child in children:
            if child in generated_nodes:
                prune_count += 1
                continue

            generated_nodes.add(child)
            open_nodes.add(child)

    if report:
        progress.stop()

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


def make_plan(pending_ops, report=True):
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

    # profiler = cProfile.Profile()
    # profiler.enable()

    moves = init_search(pending_ops, report=report)

    for move in moves:

        if move.wash_first:
            tips = []
            ops = []
            for t, o in move.tip_ops_since_wash.items():
                if o:
                    tips.append(t)
                ops += o
            actions.append(("Wash", ops, [tip + 1 for tip in tips]))
            wash_count += 1

        # Otherwise, we have these fields
        _, aspirates, dispenses, ops, tips_selected = move.tuple()

        op_tip = {op: tip + 1 for op, tip in zip(ops, tips_selected)}

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

    return actions, cost, (asp_count, disp_count, wash_count)
