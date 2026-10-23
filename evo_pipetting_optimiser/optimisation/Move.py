class Move:
    """
    Class defining a single move of the pipetting arm (aspirate and dispense), what tips are used, and what ops are included
    """

    def __init__(self):
        self.aspirates = {}
        self.dispenses = {}
        self.ops = []
        self.tips = []
        self.score = 0

        self.wash_first = False

    def calc_score(self):
        # Score the efficiency of the move, penalising moves that need more
        # distinct aspirate/dispense groups relative to the ops they cover
        num_actions = len(self.aspirates) + len(self.dispenses)

        # 0.25 weight on actions tested to perform well on range of use cases
        self.score = (
            len(self.ops) - (1 if self.wash_first else 0) - 0.25 * num_actions
        )

    def key(self):
        asp_ops = tuple(op for set in self.aspirates.values() for op in set)
        disp_ops = tuple(op for set in self.dispenses.values() for op in set)

        return (-self.score, len(asp_ops) + len(disp_ops), asp_ops, disp_ops)

    def tuple(self):
        return self.score, self.aspirates, self.dispenses, self.ops, self.tips

    def __repr__(self):
        return str(self.tuple())

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __lt__(self, other):
        return self.key() > other.key()

    def __gt__(self, other):
        return self.key() < other.key()
