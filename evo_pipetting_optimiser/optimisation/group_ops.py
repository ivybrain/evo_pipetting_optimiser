import itertools
from collections import deque
from robotools import *
from ..utils import *
import numpy as np
from ..robotools_extensions import TransferOperation

import time
from copy import copy
from .group_by import group_by


def group_ops(pending_ops):
    """
    A group is a 5-tuple of
    (
        pipetting steps required
        'source' or 'destination'
        the ops included across groups
        the group details, which is a 3-tuple of (ops, tip-op assignment for primary, tip-op assignment for secondary ),
        the overall tips used
    )
    """

    # Get the best groups of ops when grouping by source,
    # and grouping by destination
    # See group_by method for more details
    best_groupings = group_by(pending_ops, "source")
    best_groupings += group_by(pending_ops, "destination")

    # Sort the groups by best cost, then earliest op, then the most ops achieved
    def group_sort_key(group):
        (steps, _, ops, _, _) = group
        # Cost is the number of steps (aspirates + dispenses) needed for the group,
        # divided by the number of ops (well transfers) achieved
        cost = steps / len(ops)
        return (cost, min([op.id for op in ops]), -1 * len(ops))

    best_groupings.sort(key=group_sort_key)
    return best_groupings
