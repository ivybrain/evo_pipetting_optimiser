from copy import copy, deepcopy
import cProfile, pstats
import time
from sortedcontainers import SortedSet
from .OpGraph import OpGraph

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
            raise OffsetLimitException(
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


def init_search(pending_ops, report=True):

    initial_node = Node(None, pending_ops)
    open_nodes = SortedSet([initial_node])
    generated_nodes = set([initial_node])
    node = None
    max_len = 0
    prune_count = 0

    if report:
        progress = Progress(
            TextColumn("[bold pink]{task.description}"),
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
                    # Give the wash command with the last op on the tip
                    # We will use the wash scheme from this one
                    ops.append(o[-1])

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
