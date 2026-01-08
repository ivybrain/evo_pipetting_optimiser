from .group_ops import group_ops
from .group_by import init_search
from copy import copy, deepcopy
import cProfile, pstats
import time
import networkx as nx
from collections import deque
from sortedcontainers import SortedSet, SortedKeyList
from matplotlib import pyplot as plt
from networkx.algorithms.approximation import maxcut
from networkx.algorithms.flow import shortest_augmenting_path

OFFSET_EXPLORE_ORDER = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7]


class OffsetLimitException(Exception):
    pass


def add_op(graph, op):
    edge_key = source_key(op), dest_key(op)

    if edge_key in graph.edges:
        graph.edges[edge_key]["ops"].append(op)
        return

    graph.add_edge(source_key(op), dest_key(op), ops=[op])


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
            (pending_graph, open_graph, pending_ops, move, cost) = args

            self.pending_graph = pending_graph
            self.open_graph = open_graph
            self.pending_ops = pending_ops
            self.depth = parent.depth + 1

            self.cost = cost
            self.move = move

        self.fscore = self.cost + self.heuristic_cost_estimate()

        return

    def init_root(self, pending_ops):
        self.pending_ops = pending_ops

        self.pending_graph = nx.DiGraph()
        self.open_graph = nx.Graph()

        self.move = None
        self.cost = 0
        self.depth = 0

        # self.open_ops = set(
        #     [
        #         op
        #         for op in pending_ops
        #         if op.source_dep not in pending_ops and op.dest_dep not in pending_ops
        #     ]
        # )

        # for op in pending_ops:

        #     add_op(self.pending_graph, op)

        #     if op.source_dep not in pending_ops and op.dest_dep not in pending_ops:
        #         add_op(self.open_graph, op)
        #         self.open_ops.add(op)

        G = self.pending_graph
        scaling = 1  # least common multiple of 1..8
        demand = scaling * min(8, len(self.pending_ops))
        G.add_node("s", demand=-demand)
        G.add_node("t", demand=demand)
        for op in pending_ops:

            source_row_key = (*source_key(op), op.source_pos[0])

            if ("s", source_row_key) in G.edges:
                G.edges[("s", source_row_key)]["capacity"] += scaling
            else:
                G.add_edge("s", source_row_key, capacity=scaling, weight=0)

            dest_row_key = (*dest_key(op), op.dest_pos[0])
            if (dest_row_key, "t") in G.edges:
                G.edges[(dest_row_key, "t")]["capacity"] += scaling
            else:
                G.add_edge(dest_row_key, "t", capacity=scaling, weight=0)

            # G.add_edge(source_row_key, dest_row_key, capacity=1, weight=0)

            for source_offset in OFFSET_EXPLORE_ORDER:

                # Add edges from rows to possible offsets

                # Check if this offset is feasible for the source row
                if (
                    source_offset > op.source_pos[0]
                    or source_offset < op.source_pos[0] - 7
                ):
                    continue

                dest_offset = source_offset + (op.dest_pos[0] - op.source_pos[0])

                # Check if this offset is feasible for the dest row
                if dest_offset > op.dest_pos[0] or dest_offset < op.dest_pos[0] - 7:
                    continue

                source_offset_key = ("ofst", *source_key(op), source_offset)
                dest_offset_key = ("ofst", *dest_key(op), dest_offset)
                G.add_edge(
                    source_row_key,
                    source_offset_key,
                    capacity=scaling,
                    weight=0,
                )

                if (source_offset_key, dest_offset_key) in G.edges:
                    G.edges[(source_offset_key, dest_offset_key)]["capacity"] += scaling
                    G.edges[(source_offset_key, dest_offset_key)]["weight"] = 1
                else:
                    G.add_edge(
                        source_offset_key,
                        dest_offset_key,
                        capacity=scaling,
                        weight=1,
                    )

                G.add_edge(
                    dest_offset_key,
                    dest_row_key,
                    capacity=scaling,
                    weight=0,
                )

        flow_value = nx.maximum_flow_value(G, "s", "t")
        print(flow_value)

        G = shortest_augmenting_path(G, "s", "t")

        # print(flow_dict)

        # # 2. Add the results back to the graph as edge attributes
        # for u, v in G.edges():
        #     try:
        #         G[u][v]["flow"] = flow_dict[u][v]
        #     except KeyError:
        #         G[u][v]["flow"] = 0

        # print(
        #     sum(
        #         [
        #             G.edges[e]["weight"] / scaling if G.edges[e]["flow"] > 0 else 0
        #             for e in G.edges
        #         ]
        #     )
        # )
        # print(nx.cost_of_flow(G, flow_dict))

        pos = nx.bfs_layout(G, start="s")

        # Identify edges with positive flow
        active_edges = [(u, v) for u, v in G.edges() if G[u][v]["flow"] > 0]

        # Draw nodes and all edges faintly
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos, alpha=0.1)  # faint background edges

        # Draw active flow edges prominently
        nx.draw_networkx_edges(G, pos, edgelist=active_edges, edge_color="r", width=2)

        # Extract the flow values for labeling
        flow_labels = nx.get_edge_attributes(G, "flow")
        capacity_labels = nx.get_edge_attributes(G, "capacity")

        edge_labels = {
            key: f"{flow_labels[key]/scaling}/{capacity_labels[key]/scaling}"
            for key in flow_labels.keys()
            if key in active_edges
        }

        node_labels = {node: (node[-1]) for node in G.nodes}

        # Draw the labels on the edges
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        nx.draw_networkx_labels(G, pos, labels=node_labels)

        plt.show()

        return

    def get_neighbours(self):

        components = nx.connected_components(self.open_graph)

        children = []

        for c in components:

            subgraph = self.open_graph.subgraph(c)

            cut_value, partition = maxcut.one_exchange(subgraph)
            print(cut_value)
            print(partition[0])
            print()
            print(partition[1])

            nx.draw(subgraph, with_labels=True)
            plt.show()

            min_cut_edges = []

            min_cut_edges = nx.minimum_edge_cut(subgraph)
            min_cut_value = len(min_cut_edges)

            if min_cut_value == 0:
                return []

            G_cut = self.open_graph.copy()
            G_cut.remove_edges_from(min_cut_edges)
            isolates = list(nx.isolates(G_cut))
            G_cut.remove_nodes_from(isolates)

            components = nx.connected_components(G_cut)

            cost = self.cost + 1

            child_pending = self.pending_graph.copy()

            # Adjust the graph with the old ops removed and new added
            child_pending.remove_edges_from(min_cut_edges)
            isolates = list(nx.isolates(child_pending))
            child_pending.remove_nodes_from(isolates)

            child_open = G_cut

            ops_this_cut = []
            if min_cut_edges:
                for edge in min_cut_edges:
                    ops = self.open_graph.edges[edge]["ops"]
                    ops_this_cut += ops

            new_pending = self.pending_ops.difference(set(ops_this_cut))
            new_open = set()

            # Add to the new ops any newly opened ops after completing these ones
            # ops_to_add =
            for op in ops_this_cut:
                for n_op in op.dep_children:
                    # Check both the source and dest deps are satisfied
                    if (
                        n_op.source_dep not in new_pending
                        and n_op.dest_dep not in new_pending
                    ):
                        add_op(child_open, n_op)
                        new_open.add(n_op)

            child = Node(
                self, child_pending, child_open, new_pending, ops_this_cut, cost
            )

            children.append(child)

        return children

    def heuristic_cost_estimate(self):
        score = 0

        for edge in self.pending_graph.edges:
            score += (len(self.pending_graph.edges[edge]["ops"]) // 8) + 1

        return score

    def key(self):
        # If pending_ops is the same, the graph must have the same nodes
        return (
            self.fscore,
            tuple(self.pending_ops),
            tuple(sorted(self.pending_graph.edges)),
        )

    def __hash__(self):
        return hash(self.key())

    # Hashes will collide whenever edges and vertices are the same - we then use isomorphic to check equality
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False

        if self.key() != other.key():
            return False

        iso = nx.is_isomorphic(self.pending_graph, other.pending_graph)
        return iso

    def __repr__(self):
        return f"Node score {self.fscore}"

    def create_row_graph(self):
        self.row_graph = nx.DiGraph()
        G = self.row_graph

        for op in self.move:

            G.add_edge("s", f"s{op.source_pos[0]}", capacity=1)
            G.add_edge(f"d{op.dest_pos[0]}", "t", capacity=1)

            key = f"s{op.source_pos[0]}", f"d{op.dest_pos[0]}"

            if key in self.row_graph.edges:
                self.row_graph.edges[key]["ops"].append(op)
                continue

            self.row_graph.add_edge(*key, ops=[op], capacity=1)

        for edge in self.row_graph.edges:
            if "ops" in self.row_graph.edges[edge]:
                print(
                    edge,
                    self.row_graph.edges[edge]["ops"],
                )

        flow_value, flow_dict = nx.maximum_flow(G, "s", "t")
        print(flow_value)
        print(flow_dict)

        nx.draw(self.row_graph, with_labels=True)
        plt.show()

        return


def source_key(op):
    return (op.source.name, op.source_pos[1], op.liquid_class)


def dest_key(op):
    return (op.destination.name, op.dest_pos[1], op.liquid_class)


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

        if len(node.pending_graph) == 0:
            break

        children = node.get_neighbours()
        print(
            f"Depth {node.depth}, score {node.fscore}, queue {len(open_nodes)}, new_children {len(children)}, pruned {prune_count}"
        )

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
            print((node, source_key(node.move[0]), dest_key(node.move[0]), node.move))

        # nx.draw(node.open_graph, with_labels=True)
        # plt.show()
        continue

    for node in path[1:]:
        node.create_row_graph()

    pass


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

    # Repeat until we have no more operations pending
    while len(pending_ops) > 0:
        # Group the operations. See group_ops method for details

        best_groupings = init_search(pending_ops)

        # TO implement: All groups that do not contain newly completed ops are still valid
        # keep in memory and skip calculation for next

        # In the simple case, we can just take the single best group. However, if the best group doesn't use all 8 tips,
        # We can select other groups to use the remaining tips
        # Before washing them all together

        # Track how many tips we've used across the currently selected groups
        tips_used = 0
        # Track the selected groups, and the ops among those selected groups
        selected_groups = []
        selected_ops = []

        # Store the cost of executing the second best group available
        # This is a benchmark for whether to include subsequent groups
        # e.g. if it's more efficient to select the second best group and do an additional wash,
        # compared to adding an unefficient group to the remaining tips,
        # Then don't add any more groups, and leave the remaining tips empty
        # second best cost is the number of ((aspirates + dispenses) + 2 (as a wash takes two movements into the waste then cleaner)),
        # divided by the number of ops achieved by that group
        second_best_cost = (best_groupings[1][0] + 2) / len(best_groupings[1][2])

        (steps, group_type, ops, target_groups, tips_selected) = best_groupings[0]

        selected_groups.append((group_type, ops, target_groups, tips_selected))
        selected_ops += ops

        if len(selected_ops) == 0:
            raise OffsetLimitException(
                "No valid groups to select. This probably means you're trying to transfer between two plates with offset restrictions that don't allow this transfer"
            )

        # Process the groups into a list of sources to aspirate and a list of destinations to dispense
        source_list = []
        dest_list = []
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

        for group_type, ops, target_groups, tips_selected in selected_groups:

            if group_type == "source":
                # If we've grouped by source, we can aspirate all ops in the group at once
                source_list += [ops]
                # Destinations will depend on the subgroups selected, one for each subgroup
                dest_list += [group[0] for group in target_groups]

            else:
                # If we've grouped by destination, we can dispense all ops in the group at once
                dest_list += [ops]
                # Sources will depend on the subgroups selected, one for each subgroup
                source_list += [group[0] for group in target_groups]

        # Sort the source list and dest list by the tips used
        # This just makes sure the pipetting occurs in an order that's less confusing visually,
        # starting from tip 1 to tip 8
        group_sort_key = lambda ops: min([op_tip[op] for op in ops])
        source_list.sort(key=group_sort_key)
        dest_list.sort(key=group_sort_key)

        for source_group in source_list:

            # Sort the group by assigned tip
            source_group.sort(key=lambda op: op_tip[op])

            actions.append(
                ("Aspirate", source_group, [op_tip[op] for op in source_group])
            )

            # Remove liquid from our virtual copy of the plates so we can track composition

            virtual_source = labware_copy(source_group[0].source)
            virtual_source.remove(
                [virtual_source.wells[op.source_pos] for op in source_group],
                [op.volume for op in source_group],
            )

            asp_count += 1

        for dest_group in dest_list:

            # Sort the group by assigned tip
            dest_group.sort(key=lambda op: op_tip[op])

            actions.append(("Dispense", dest_group, [op_tip[op] for op in dest_group]))

            # Add liquid to our virtual copy of the plates so we can track composition
            virtual_dest = labware_copy(dest_group[0].destination)
            compositions = [
                labware_copy(op.source).get_well_composition(
                    op.source.wells[op.source_pos]
                )
                for op in dest_group
            ]
            virtual_dest.add(
                [virtual_dest.wells[op.dest_pos] for op in dest_group],
                [op.volume for op in dest_group],
                compositions=compositions,
            )

            disp_count += 1

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
