def group_movments_needed(op_set, field):
    """
    Group operations by the specified field (source or destination),
    The column, and the liquid class. We group like this because these are the constraints
    To aspirating (or dispensing) something in one step
    """
    group_dict = {}
    for op in op_set:

        if field == "source":
            key = (op.source.name, op.source_pos[1], op.liquid_class)
        else:
            key = (op.destination.name, op.dest_pos[1], op.liquid_class)
        if key not in group_dict:
            group_dict[key] = []

        group_dict[key].append(op)

    return group_dict


def check_offset_limits(op, tip):
    """
    Check if we have violated an offset limit on the source or destination, and how much by
    Inputs: an operation, and the tip we have assigned to that operation
    Outputs: tuple of up_limit_violation, down_limit_violation
    up_limit_violation - if < 0 the tips are too far up. Need to add this value to tip used (lower tip number will be used)
    down_limit_violation - if > 0 the tips are too far down. Need to add this value to tip used (higher tip number will be used)
    """

    source_limit_up = getattr(op.source, "offset_limit_up", None)
    source_limit_down = getattr(op.source, "offset_limit_down", None)
    dest_limit_up = op.destination.offset_limit_up
    dest_limit_down = op.destination.offset_limit_down

    source_offset = op.source_pos[0] - tip
    dest_offset = op.dest_pos[0] - tip

    def check_limit(offset, limit, subtract=False):
        # Check if we have exceeded the offset limit allowed
        if limit is None:
            return 0
        if subtract:
            return offset - limit
        return offset + limit

    source_up_check = check_limit(source_offset, source_limit_up)
    source_down_check = check_limit(source_offset, source_limit_down, True)
    dest_up_check = check_limit(dest_offset, dest_limit_up)
    dest_down_check = check_limit(dest_offset, dest_limit_down, True)

    return min(source_up_check, dest_up_check), max(source_down_check, dest_down_check)


def check_sublist(main_list, sub_list):
    if not sub_list:
        return True
    if len(sub_list) > len(main_list):
        return False
    for i in range(len(main_list) - len(sub_list) + 1):
        if main_list[i : i + len(sub_list)] == sub_list:
            return True
    return False
