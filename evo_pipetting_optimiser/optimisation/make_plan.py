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
    Used to enable A-star search where we can look ahead to find the best action several moves in advance
    A-star is not currently used for performance reasons - we only select the best node at each step
    Retained in case better optimisation decisions are needed in the future
    """

    def __init__(self, parent, *args):

        self.parent = parent

        if not parent:
            (pending_ops,) = args
            self.pending_ops = pending_ops

            # Open ops are those currently executable (do not have dependencies)
            self.open_ops = set(
                [
                    op
                    for op in pending_ops
                    if op.source_dep not in pending_ops
                    and op.dest_dep not in pending_ops
                ]
            )

            # Initialise the graph representing the open ops
            self.open_graph = OpGraph(self.open_ops)
            self.open_graph.node = self

            self.move = None
            self.cost = 0
            self.depth = 0

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

    @property
    def unused_tips(self):
        return [i for i in range(8) if not self.open_graph.tip_ops_since_wash[i]]

    def get_neighbours(self):
        """
        Get the liquid handler moves that we can make from the current state, and the resulting nodes (states) from those moves
        We currently only select the 1 best move and create one new state (greedy approach)
        """

        children = []

        # Create a dictionary tracking all the ops that each tip has conducted since its lart wash
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
        # select only one best move (moves are pre-sorted by the get_moves method)
        moves = moves[:1]

        for move in moves:

            # Get the details of this move
            score, aspirates, dispenses, ops, tips = move.tuple()

            # The new node's pending operations will not include the ops that were just comlpeted
            new_pending = copy(self.pending_ops)
            new_pending.difference_update(set(ops))

            # Calculate which ops are 'open' (executable straight away, without dependencies) from the new pending ops
            # This may include new ops now that some ops are no longer pending (complete)
            new_open_ops = set(
                [
                    op
                    for op in new_pending
                    if op.source_dep not in new_pending
                    and op.dest_dep not in new_pending
                ]
            )

            # New cost is the old cost, plus how many aspirates, dispenses, and wash steps this move took
            new_cost = (
                self.cost
                + len(aspirates)
                + len(dispenses)
                + (2 if move.wash_first else 0)
            )

            # Create the new node with the new pending and open states

            new_node = Node(
                self, new_pending, new_open_ops, OpGraph(new_open_ops), move, new_cost
            )
            new_node.open_graph.node = new_node
            children.append(new_node)

        if not children:
            # If we can't find a move, then there are no valid moves to complete any of the remaining ops
            raise OffsetLimitException(
                "Failed to solve. Perhaps you are trying to pipette wells with impossible offsets?"
            )

        return children

    def heuristic_cost_estimate(self):
        """
        Used to evaluate which node to choose next for Astar
        Not currently relevant as we only have one node at a time
        """

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
            TextColumn("[bright_magenta]{task.description}"),
            BarColumn(finished_style="bar.complete"),
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

            asp_count += 1

        for (labware, col, offset), disp_ops in dispenses.items():

            # Sort the group by assigned tip
            disp_ops.sort(key=lambda op: op_tip[op])

            actions.append(("Dispense", disp_ops, [op_tip[op] for op in disp_ops]))

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
