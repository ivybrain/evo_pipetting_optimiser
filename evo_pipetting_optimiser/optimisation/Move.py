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
        # Score the efficiency of the move
        self.score = (len(self.ops) * 2) / (
            len(self.aspirates) + len(self.dispenses) + (2 if self.wash_first else 0)
        )

    def key(self):
        asp_ops = tuple(op for set in self.aspirates.values() for op in set)
        disp_ops = tuple(op for set in self.dispenses.values() for op in set)

        return (-self.score, asp_ops, disp_ops)

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
        return self.key() > other.key()

    def __gt__(self, other):
        return self.key() < other.key()
