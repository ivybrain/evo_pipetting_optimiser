from .group_ops import group_ops
from .group_by import init_search
from copy import copy, deepcopy
import cProfile, pstats
import time
import networkx as nx
from collections import deque
from sortedcontainers import SortedKeyList
from matplotlib import pyplot as plt


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

            self.cost = cost
            self.move = move

        self.fscore = self.cost + self.heuristic_cost_estimate()

        return

    def init_root(self, pending_ops):
        self.pending_ops = pending_ops

        self.pending_graph = nx.Graph()
        self.open_graph = nx.Graph()

        self.cost = 0

        self.open_ops = set(
            [
                op
                for op in pending_ops
                if op.source_dep not in pending_ops and op.dest_dep not in pending_ops
            ]
        )

        for op in pending_ops:

            add_op(self.pending_graph, op)

            if op.source_dep not in pending_ops and op.dest_dep not in pending_ops:
                add_op(self.open_graph, op)
                self.open_ops.add(op)

    def get_neighbours(self):

        components = nx.connected_components(self.open_graph)

        children = []

        for c in components:

            subgraph = self.open_graph.subgraph(c)

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
                new_open.update(op.dep_children)
                for n_op in op.dep_children:
                    add_op(child_open, n_op)

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

    def __repr__(self):
        return f"Node score{self.fscore}"


def source_key(op):
    return (op.source.name, op.source_pos[1], op.liquid_class)


def dest_key(op):
    return (op.destination.name, op.dest_pos[1], op.liquid_class)


def init_search(pending_ops):

    initial_node = Node(None, pending_ops)
    open_nodes = SortedKeyList([initial_node], key=lambda node: node.fscore)
    node = None
    while len(open_nodes) > 0:
        node = open_nodes.pop(0)

        if len(node.pending_graph) == 0:
            break

        children = node.get_neighbours()

        for child in children:
            open_nodes.add(child)

    path = [node]
    while node.parent:
        path.append(node.parent)
        node = node.parent

    path.reverse()

    for node in path:
        # nx.draw(node.open_graph, with_labels=True)
        # plt.show()
        continue

    """
    When the other side has been generated for a group, re-score it based on this
    
    oo graph from primary to secondary, edges where a group
    """

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
